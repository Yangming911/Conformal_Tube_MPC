import numpy as np
import torch
import sys
from pathlib import Path
 
# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.predictor import WalkerActionPredictor
from models.conformal_grid import get_eta
import utils.constants as C
# 初始化模型
predictor = WalkerActionPredictor(model_path="assets/best_model.pth", device="cpu")

def forward_ped(y0, u_seq, car_x0=0.0, car_y=12.0, cp_alpha=0.85):
    """
    输入 walker 初始位置 y0 = [x, y]
         和 car 控制序列 u_seq
    输出 List[(lo, hi)]: 每一步的最坏情况可达管（box）
    """
    y_lo = np.array(y0)  # 下界
    y_hi = np.array(y0)  # 上界
    tube = []
    
    # 累积车辆位置
    car_x_accum = car_x0

    for car_u in u_seq:
        car_speed = car_u  # assume ∆t = 1
        car_x_accum += car_speed * C.dt  # 累积车辆x位置

        # 对 tube box 的所有点使用 conservative 传播
        walker_x_lo, walker_y_lo = y_lo[0], y_lo[1]
        walker_x_hi, walker_y_hi = y_hi[0], y_hi[1]

        # 用最保守点生成 a_pred 和 η（用上下边界中偏大的）
        # 使用新的predictor接口
        a_pred_lo = predictor.predict(
            car_x=car_x_accum, car_y=car_y, car_v=car_speed,
            walker_x=walker_x_lo, walker_y=walker_y_lo,
            walker_vx=0.0, walker_vy=0.0  # 简化假设当前速度为0
        )
        a_pred_hi = predictor.predict(
            car_x=car_x_accum, car_y=car_y, car_v=car_speed,
            walker_x=walker_x_hi, walker_y=walker_y_hi,
            walker_vx=0.0, walker_vy=0.0  # 简化假设当前速度为0
        )
        
        # 将元组转换为numpy数组
        a_pred_lo = np.array([a_pred_lo[0], a_pred_lo[1]])
        a_pred_hi = np.array([a_pred_hi[0], a_pred_hi[1]])

        # 获取eta值
        eta_lo = get_eta(car_x=car_x_accum, car_v=car_speed, walker_x=walker_x_lo, walker_y=walker_y_lo,
                         walker_vx=0.0, walker_vy=0.0, cp_alpha=cp_alpha)
        eta_hi = get_eta(car_x=car_x_accum, car_v=car_speed, walker_x=walker_x_hi, walker_y=walker_y_hi,
                         walker_vx=0.0, walker_vy=0.0, cp_alpha=cp_alpha)

        # 最坏情况传播：分别传播 lo 和 hi
        a_min = np.minimum(a_pred_lo - eta_lo, a_pred_hi - eta_hi)
        a_max = np.maximum(a_pred_lo + eta_lo, a_pred_hi + eta_hi)

        y_lo = y_lo + a_min * C.dt  # 正确使用 x = x + v*dt
        y_hi = y_hi + a_max * C.dt  # 正确使用 x = x + v*dt

        tube.append((y_lo.copy(), y_hi.copy()))

    return tube

def forward_ped_trace(y0, u_seq, car_x0=0.0, car_y=12.0, cp_alpha=0.85):
    """
    输入 walker 初始位置 y0 = [x, y]
         和 car 控制序列 u_seq
    输出 List[(y)]: 每一步预测可达点（trace）
    """
    y = np.array(y0)
    trace = []
    
    # 累积车辆位置
    car_x_accum = car_x0

    for car_u in u_seq:
        car_speed = car_u  # u_seq中的值是速度
        car_x_accum += car_speed * C.dt  # 累积车辆x位置

        # 使用新的predictor接口
        a_pred = predictor.predict(
            car_x=car_x_accum, car_y=car_y, car_v=car_speed,
            walker_x=y[0], walker_y=y[1],
            walker_vx=0.0, walker_vy=0.0  # 简化假设当前速度为0
        )
        # 将元组转换为numpy数组
        a_pred = np.array([a_pred[0], a_pred[1]])
        y = y + a_pred * C.dt  # 正确使用 x = x + v*dt
        trace.append(y.copy())

    return trace