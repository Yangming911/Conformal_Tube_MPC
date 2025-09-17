"""
测试训练好的行人速度预测模型
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.predictor import WalkerActionPredictor
from models.model_def import WalkerSpeedPredictor, WalkerSpeedPredictorV2


def test_model_loading():
    """测试模型加载"""
    print("Testing model loading...")
    
    try:
        # 测试新模型加载
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        predictor = WalkerActionPredictor('../assets/walker_speed_predictor.pth', device=device)
        print("✓ Model loaded successfully")
        print(f"  - Is legacy model: {predictor.is_legacy_model}")
        print(f"  - Has scaler_X: {predictor.scaler_X is not None}")
        print(f"  - Has scaler_y: {predictor.scaler_y is not None}")
        return predictor
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def test_prediction(predictor):
    """测试预测功能"""
    print("\nTesting prediction...")
    
    if predictor is None:
        print("✗ No predictor available")
        return
    
    # 测试数据
    test_cases = [
        {
            'name': 'Normal case',
            'car_x': 10.0, 'car_y': 12.0, 'car_v': 5.0,
            'walker_x': 30.0, 'walker_y': 5.0, 'walker_vx': 0.0, 'walker_vy': 0.0
        },
        {
            'name': 'High speed case',
            'car_x': 15.0, 'car_y': 12.0, 'car_v': 12.0,
            'walker_x': 25.0, 'walker_y': 8.0, 'walker_vx': 0.1, 'walker_vy': 1.0
        },
        {
            'name': 'Close encounter',
            'car_x': 20.0, 'car_y': 12.0, 'car_v': 8.0,
            'walker_x': 22.0, 'walker_y': 10.0, 'walker_vx': -0.2, 'walker_vy': 0.5
        }
    ]
    
    for case in test_cases:
        try:
            vx, vy = predictor.predict(
                case['car_x'], case['car_y'], case['car_v'],
                case['walker_x'], case['walker_y'], case['walker_vx'], case['walker_vy']
            )
            print(f"✓ {case['name']}: vx={vx:.4f}, vy={vy:.4f}")
        except Exception as e:
            print(f"✗ {case['name']}: Error - {e}")


def test_batch_prediction(predictor):
    """测试批量预测"""
    print("\nTesting batch prediction...")
    
    if predictor is None:
        print("✗ No predictor available")
        return
    
    # 生成测试数据
    n_samples = 100
    np.random.seed(42)
    
    car_x = np.random.uniform(0, 50, n_samples)
    car_y = np.full(n_samples, 12.0)
    car_v = np.random.uniform(1, 15, n_samples)
    walker_x = np.random.uniform(20, 40, n_samples)
    walker_y = np.random.uniform(0, 15, n_samples)
    walker_vx = np.random.uniform(-1, 1, n_samples)
    walker_vy = np.random.uniform(0, 2, n_samples)
    
    predictions = []
    for i in range(n_samples):
        vx, vy = predictor.predict(
            car_x[i], car_y[i], car_v[i],
            walker_x[i], walker_y[i], walker_vx[i], walker_vy[i]
        )
        predictions.append([vx, vy])
    
    predictions = np.array(predictions)
    
    print(f"✓ Batch prediction completed for {n_samples} samples")
    print(f"  - vx range: [{predictions[:, 0].min():.4f}, {predictions[:, 0].max():.4f}]")
    print(f"  - vy range: [{predictions[:, 1].min():.4f}, {predictions[:, 1].max():.4f}]")
    print(f"  - vx mean: {predictions[:, 0].mean():.4f}, std: {predictions[:, 0].std():.4f}")
    print(f"  - vy mean: {predictions[:, 1].mean():.4f}, std: {predictions[:, 1].std():.4f}")


def test_model_architecture():
    """测试模型架构"""
    print("\nTesting model architectures...")
    
    # 测试基础模型
    try:
        model1 = WalkerSpeedPredictor(input_dim=7, hidden_dims=[128, 64], output_dim=2)
        x = torch.randn(10, 7)
        y = model1(x)
        print(f"✓ WalkerSpeedPredictor: input {x.shape} -> output {y.shape}")
    except Exception as e:
        print(f"✗ WalkerSpeedPredictor error: {e}")
    
    # 测试改进模型
    try:
        model2 = WalkerSpeedPredictorV2(input_dim=7, hidden_dim=128, output_dim=2)
        x = torch.randn(10, 7)
        y = model2(x)
        print(f"✓ WalkerSpeedPredictorV2: input {x.shape} -> output {y.shape}")
    except Exception as e:
        print(f"✗ WalkerSpeedPredictorV2 error: {e}")


def test_data_compatibility():
    """测试数据兼容性"""
    print("\nTesting data compatibility...")
    
    try:
        # 检查数据集
        df = pd.read_csv('../assets/sf_dataset.csv')
        print(f"✓ Dataset loaded: {df.shape}")
        
        # 检查特征列
        feature_cols = ['car_x', 'car_y', 'car_v', 'walker_x', 'walker_y', 'walker_vx', 'walker_vy']
        target_cols = ['next_walker_vx', 'next_walker_vy']
        
        missing_features = [col for col in feature_cols if col not in df.columns]
        missing_targets = [col for col in target_cols if col not in df.columns]
        
        if missing_features:
            print(f"✗ Missing features: {missing_features}")
        else:
            print("✓ All required features present")
        
        if missing_targets:
            print(f"✗ Missing targets: {missing_targets}")
        else:
            print("✓ All required targets present")
        
        # 检查数据范围
        print("\nData ranges:")
        for col in feature_cols + target_cols:
            if col in df.columns:
                print(f"  {col}: [{df[col].min():.4f}, {df[col].max():.4f}]")
        
    except Exception as e:
        print(f"✗ Data compatibility error: {e}")


def main():
    """主测试函数"""
    print("=" * 50)
    print("Walker Speed Predictor Model Test")
    print("=" * 50)
    
    # 测试数据兼容性
    test_data_compatibility()
    
    # 测试模型架构
    test_model_architecture()
    
    # 测试模型加载
    predictor = test_model_loading()
    
    # 测试预测功能
    test_prediction(predictor)
    
    # 测试批量预测
    test_batch_prediction(predictor)
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
