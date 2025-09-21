import torch
import os
import numpy as np
from models.model_def import WalkerSpeedPredictor, WalkerSpeedPredictorV2

class WalkerActionPredictor:
    """Pedestrian action predictor - compatible with old and new version models"""
    
    def __init__(self, model_path='assets/walker_speed_predictor.pth', device='cpu'):
        self.device = device
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.is_legacy_model = False
        
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load model, supports old and new formats"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Try loading new format (includes scaler)
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                # New format
                config = checkpoint.get('config', {})
                model_name = config.get('model_name', 'WalkerSpeedPredictor')
                
                if model_name == 'WalkerSpeedPredictorV2':
                    self.model = WalkerSpeedPredictorV2(
                        input_dim=config.get('input_dim', 7),
                        hidden_dim=config.get('hidden_dim', 128),
                        output_dim=config.get('output_dim', 2),
                        num_layers=config.get('num_layers', 4),
                        dropout_rate=config.get('dropout_rate', 0.1)
                    )
                else:
                    self.model = WalkerSpeedPredictor(
                        input_dim=config.get('input_dim', 7),
                        hidden_dims=config.get('hidden_dims', [128, 128, 64]),
                        output_dim=config.get('output_dim', 2),
                        dropout_rate=config.get('dropout_rate', 0.1)
                    )
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.scaler_X = checkpoint.get('scaler_X')
                self.scaler_y = checkpoint.get('scaler_y')
                self.is_legacy_model = False
            else:
                # Old format
                self.model = WalkerSpeedPredictor()
                self.model.load_state_dict(checkpoint)
                self.is_legacy_model = True
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy):
        """
        Predict pedestrian next step velocity
        
        Args:
            car_x, car_y, car_v: Vehicle position and velocity
            walker_x, walker_y, walker_vx, walker_vy: Pedestrian position and velocity
        
        Returns:
            tuple: (next_walker_vx, next_walker_vy)
        """
        # Prepare input features
        if self.is_legacy_model:
            # 旧模型只使用car_v和walker_y
            features = np.array([[car_v, walker_y]], dtype=np.float32)
        else:
            # 新模型使用所有7个特征
            features = np.array([[car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy]], 
                              dtype=np.float32)
            
            # 标准化特征
            if self.scaler_X is not None:
                features = self.scaler_X.transform(features)
        
        # 预测
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(features_tensor).cpu().numpy()
        
        # 反标准化预测结果
        if not self.is_legacy_model and self.scaler_y is not None:
            predictions = self.scaler_y.inverse_transform(predictions)
        
        return predictions[0][0], predictions[0][1]
    
    def predict_legacy(self, car_speed, walker_y):
        """
        兼容旧版本的预测接口
        注意：这个方法假设使用的是旧模型格式
        """
        if not self.is_legacy_model:
            raise ValueError("Legacy predict method can only be used with legacy models")
        
        inp = torch.tensor([[car_speed, walker_y]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            vx, vy = self.model(inp).cpu().numpy()[0]
        return vx, vy
