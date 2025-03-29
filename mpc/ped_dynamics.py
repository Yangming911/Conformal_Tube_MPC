import numpy as np
import torch
from models.model_def import WalkerSpeedPredictor
from models.conformal_grid import get_eta

# 初始化模型
model = WalkerSpeedPredictor()
model.load_state_dict(torch.load("assets/walker_speed_predictor.pth"))
model.eval()

def forward_ped(y0, u_seq):
    """
    输入 walker 初始位置 y0 = [x, y]
         和 car 控制序列 u_seq
    输出 List[(lo, hi)]: 每一步的最坏情况可达管（box）
    """
    y_lo = np.array(y0)  # 下界
    y_hi = np.array(y0)  # 上界
    tube = []

    for car_u in u_seq:
        car_speed = car_u  # assume ∆t = 1

        # 对 tube box 的所有点使用 conservative 传播
        walker_y_lo = y_lo[1]
        walker_y_hi = y_hi[1]

        # 用最保守点生成 a_pred 和 η（用上下边界中偏大的）
        input_tensor_lo = torch.tensor([[car_speed, walker_y_lo]], dtype=torch.float32)
        input_tensor_hi = torch.tensor([[car_speed, walker_y_hi]], dtype=torch.float32)

        with torch.no_grad():
            a_pred_lo = model(input_tensor_lo).numpy()[0]
            a_pred_hi = model(input_tensor_hi).numpy()[0]

        eta_lo = get_eta(car_speed, a_pred_lo[0], a_pred_lo[1], walker_y_lo)
        eta_hi = get_eta(car_speed, a_pred_hi[0], a_pred_hi[1], walker_y_hi)

        # 最坏情况传播：分别传播 lo 和 hi
        a_min = np.minimum(a_pred_lo - eta_lo, a_pred_hi - eta_hi)
        a_max = np.maximum(a_pred_lo + eta_lo, a_pred_hi + eta_hi)

        y_lo = y_lo + a_min
        y_hi = y_hi + a_max

        tube.append((y_lo.copy(), y_hi.copy()))

    return tube

def forward_ped_trace(y0, u_seq):
    """
    输入 walker 初始位置 y0 = [x, y]
         和 car 控制序列 u_seq
    输出 List[(y)]: 每一步预测可达点（trace）
    """
    y = np.array(y0)
    trace = []

    for car_u in u_seq:
        car_speed = car_u  # assume ∆t = 1

        input_tensor = torch.tensor([[car_speed, y[1]]], dtype=torch.float32)
        with torch.no_grad():
            a_pred = model(input_tensor).numpy()[0]
        y = y + a_pred
        trace.append(y.copy())

    return trace