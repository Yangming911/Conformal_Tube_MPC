import torch
import os
from models.model_def import WalkerSpeedPredictor

class WalkerActionPredictor:
    def __init__(self, model_path='assets/walker_speed_predictor.pth', device='cpu'):
        self.model = WalkerSpeedPredictor()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

    def predict(self, car_speed, walker_y):
        inp = torch.tensor([[car_speed, walker_y]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            vx, vy = self.model(inp).cpu().numpy()[0]
        return vx, vy
