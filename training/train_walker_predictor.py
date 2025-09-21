#!/usr/bin/env python3
"""
Pedestrian velocity prediction model training script
Train using sf_dataset.csv data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path

# Add project root directory to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model_def import WalkerSpeedPredictor, WalkerSpeedPredictorV2


class WalkerDataset(Dataset):
    """Pedestrian dataset class"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class WalkerTrainer:
    """Pedestrian velocity prediction model trainer"""
    
    def __init__(self, model_name='WalkerSpeedPredictor', config=None):
        self.model_name = model_name
        self.config = config or self._get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize model
        if model_name == 'WalkerSpeedPredictor':
            self.model = WalkerSpeedPredictor(
                input_dim=self.config['input_dim'],
                hidden_dims=self.config['hidden_dims'],
                output_dim=self.config['output_dim'],
                dropout_rate=self.config['dropout_rate']
            )
        elif model_name == 'WalkerSpeedPredictorV2':
            self.model = WalkerSpeedPredictorV2(
                input_dim=self.config['input_dim'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['output_dim'],
                num_layers=self.config['num_layers'],
                dropout_rate=self.config['dropout_rate']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def _get_default_config(self):
        """Default configuration"""
        return {
            'input_dim': 7,
            'output_dim': 2,
            'hidden_dims': [128, 128, 64],
            'hidden_dim': 128,
            'num_layers': 4,
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 512,
            'num_epochs': 100,
            'patience': 20,
            'test_size': 0.2,
            'val_size': 0.1
        }
    
    def load_data(self, csv_path='../assets/sf_dataset.csv'):
        """加载和预处理数据"""
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        
        # 特征和目标
        feature_cols = ['car_x', 'car_y', 'car_v', 'walker_x', 'walker_y', 'walker_vx', 'walker_vy']
        target_cols = ['next_walker_vx', 'next_walker_vy']
        
        X = df[feature_cols].values
        y = df[target_cols].values
        
        print(f"Features shape: {X.shape}")
        print(f"Targets shape: {y.shape}")
        
        # 数据统计
        print("\nFeature statistics:")
        for i, col in enumerate(feature_cols):
            print(f"{col}: mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}, "
                  f"min={X[:, i].min():.4f}, max={X[:, i].max():.4f}")
        
        print("\nTarget statistics:")
        for i, col in enumerate(target_cols):
            print(f"{col}: mean={y[:, i].mean():.4f}, std={y[:, i].std():.4f}, "
                  f"min={y[:, i].min():.4f}, max={y[:, i].max():.4f}")
        
        # 数据分割
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config['val_size'], random_state=42
        )
        
        # 特征标准化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # 创建数据集
        self.train_dataset = WalkerDataset(X_train_scaled, y_train_scaled)
        self.val_dataset = WalkerDataset(X_val_scaled, y_val_scaled)
        self.test_dataset = WalkerDataset(X_test_scaled, y_test_scaled)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        print(f"\nData split:")
        print(f"Train: {len(self.train_dataset)} samples")
        print(f"Validation: {len(self.val_dataset)} samples")
        print(f"Test: {len(self.test_dataset)} samples")
        
        return X_test, y_test  # 返回原始测试数据用于最终评估
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for features, targets in tqdm(self.train_loader, desc="Training"):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in tqdm(self.val_loader, desc="Validation"):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """训练模型"""
        print(f"\nStarting training for {self.config['num_epochs']} epochs...")
        print(f"Model: {self.model_name}")
        print(f"Config: {self.config}")
        
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_model('../assets/best_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['num_epochs']}: "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"Best Val Loss: {self.best_val_loss:.6f}")
            
            # 早停
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # 加载最佳模型
        self.load_model('../assets/best_model.pth')
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        print("\nEvaluating model on test set...")
        self.model.eval()
        
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for features, targets_batch in self.test_loader:
                features = features.to(self.device)
                targets_batch = targets_batch.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets_batch)
                
                total_loss += loss.item()
                predictions.append(outputs.cpu().numpy())
                targets.append(targets_batch.cpu().numpy())
        
        # 反标准化预测结果
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        
        predictions_original = self.scaler_y.inverse_transform(predictions)
        targets_original = self.scaler_y.inverse_transform(targets)
        
        # 计算指标
        mse = np.mean((predictions_original - targets_original) ** 2)
        mae = np.mean(np.abs(predictions_original - targets_original))
        rmse = np.sqrt(mse)
        
        print(f"Test Results:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        # 分别计算x和y方向的误差
        mse_x = np.mean((predictions_original[:, 0] - targets_original[:, 0]) ** 2)
        mse_y = np.mean((predictions_original[:, 1] - targets_original[:, 1]) ** 2)
        mae_x = np.mean(np.abs(predictions_original[:, 0] - targets_original[:, 0]))
        mae_y = np.mean(np.abs(predictions_original[:, 1] - targets_original[:, 1]))
        
        print(f"\nPer-dimension results:")
        print(f"X-direction: MSE={mse_x:.6f}, MAE={mae_x:.6f}")
        print(f"Y-direction: MSE={mse_y:.6f}, MAE={mae_y:.6f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mse_x': mse_x,
            'mse_y': mse_y,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'predictions': predictions_original,
            'targets': targets_original
        }
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {**self.config, 'model_name': self.model_name},
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
    
    def plot_training_history(self, save_path='../logs/training_history.png'):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Training History (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training history saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Walker Speed Predictor')
    parser.add_argument('--model', type=str, default='WalkerSpeedPredictor',
                       choices=['WalkerSpeedPredictor', 'WalkerSpeedPredictorV2'],
                       help='Model architecture to use')
    parser.add_argument('--data', type=str, default='../assets/sf_dataset.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='../assets/walker_speed_predictor.pth',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'input_dim': 7,
        'output_dim': 2,
        'hidden_dims': [128, 128, 64],
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout_rate': 0.1,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'patience': args.patience,
        'test_size': 0.2,
        'val_size': 0.1
    }
    
    # 创建训练器
    trainer = WalkerTrainer(model_name=args.model, config=config)
    
    # 加载数据
    X_test, y_test = trainer.load_data(args.data)
    
    # 训练模型
    trainer.train()
    
    # 评估模型
    results = trainer.evaluate(X_test, y_test)
    
    # 保存最终模型
    trainer.save_model(args.save_path)
    print(f"Model saved to {args.save_path}")
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 保存结果
    results_path = args.save_path.replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        # 转换numpy类型为Python原生类型
        results_dict = {k: float(v) if hasattr(v, 'item') else v 
                       for k, v in results.items() if k not in ['predictions', 'targets']}
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
