import numpy as np
import torch
from models.model_def import WalkerSpeedPredictor
from models.conformal_grid import get_eta_batch

# 初始化模型
model = WalkerSpeedPredictor()
model.load_state_dict(torch.load("assets/walker_speed_predictor.pth"))
model.eval()


def forward_ped_batch(y0, u_seqs):
    """
    并行处理多个控制序列的 pedestrian tube 推演
    Args:
        y0: shape (2,) 初始位置
        u_seqs: shape (N, T) 控制序列批次

    Returns:
        tubes: List[List[(lo, hi)]]  每个样本对应一个 tube 序列
    """
    u_seqs = np.atleast_2d(u_seqs)  # 保证至少是 (N, T)
    N, T = u_seqs.shape

    tubes = [[] for _ in range(N)]
    y_lo = np.tile(np.array(y0), (N, 1))  # (N, 2)
    y_hi = np.tile(np.array(y0), (N, 1))  # (N, 2)

    for t in range(T):
        car_speed = u_seqs[:, t]  # (N,)

        walker_y_lo = y_lo[:, 1]
        walker_y_hi = y_hi[:, 1]

        # 构造批量输入
        input_lo = torch.tensor(np.stack([car_speed, walker_y_lo], axis=1), dtype=torch.float32)
        input_hi = torch.tensor(np.stack([car_speed, walker_y_hi], axis=1), dtype=torch.float32)

        with torch.no_grad():
            a_pred_lo = model(input_lo).numpy()  # (N, 2)
            a_pred_hi = model(input_hi).numpy()  # (N, 2)

        # eta_lo = np.array([get_eta(cs, a[0], a[1], y) for cs, a, y in zip(car_speed, a_pred_lo, walker_y_lo)])
        # eta_hi = np.array([get_eta(cs, a[0], a[1], y) for cs, a, y in zip(car_speed, a_pred_hi, walker_y_hi)])
        eta_lo = get_eta_batch(car_speed, a_pred_lo[:, 0], a_pred_lo[:, 1], walker_y_lo)
        eta_hi = get_eta_batch(car_speed, a_pred_hi[:, 0], a_pred_hi[:, 1], walker_y_hi)

        a_min = np.minimum(a_pred_lo - eta_lo, a_pred_hi - eta_hi)
        a_max = np.maximum(a_pred_lo + eta_lo, a_pred_hi + eta_hi)

        y_lo = y_lo + a_min
        y_hi = y_hi + a_max

        for i in range(N):
            tubes[i].append((y_lo[i].copy(), y_hi[i].copy()))

    return tubes

def forward_ped_trace_batch(y0, u_seqs):
    """
    批量版本的行人轨迹预测函数。
    Args:
        y0: (2,) 初始位置
        u_seqs: (N, T) 控制序列

    Returns:
        traces: (N, T, 2) 每个样本的轨迹
    """
    N, T = u_seqs.shape
    y = np.tile(np.array(y0), (N, 1))  # shape (N, 2)
    traces = np.zeros((N, T, 2))

    for t in range(T):
        car_speeds = u_seqs[:, t]
        walker_ys = y[:, 1]

        inputs = torch.tensor(np.stack([car_speeds, walker_ys], axis=1), dtype=torch.float32)
        with torch.no_grad():
            a_preds = model(inputs).numpy()  # (N, 2)

        y = y + a_preds
        traces[:, t, :] = y

    return traces
