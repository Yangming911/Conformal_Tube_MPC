import torch
import torch.nn as nn
import torch.nn.functional as F

class WalkerSpeedPredictor(nn.Module):
    """
    Pedestrian velocity prediction neural network
    Input: [car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy] (7D)
    Output: [next_walker_vx, next_walker_vy] (2D)
    """
    def __init__(self, input_dim=7, hidden_dims=[128, 128, 64], output_dim=2, dropout_rate=0.1):
        super(WalkerSpeedPredictor, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class WalkerSpeedPredictorV2(nn.Module):
    """
    Improved pedestrian velocity prediction network using residual connections and attention mechanism
    """
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=2, num_layers=4, dropout_rate=0.1):
        super(WalkerSpeedPredictorV2, self).__init__()
        
        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_layers)
        ])
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout_rate)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 自注意力 (需要添加序列维度)
        x_attn = x.unsqueeze(0)  # [1, batch_size, hidden_dim]
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x_attn.squeeze(0)  # [batch_size, hidden_dim]
        
        # 输出
        return self.output_layer(x)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out += residual
        out = F.relu(out)
        return out
