import numpy as np
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.constants as C

def forward_car(x0, u_seq):
    """
    给定初始位置 x0 和控制序列 u_seq，生成未来 T 步的位置轨迹
    Args:
        x0: float, 当前车的位置
        u_seq: ndarray, 控制序列 u_0, ..., u_{T-1} (速度值)
    Returns:
        List[float]，车每一步的 x_t
    """
    x = x0
    traj = []
    for u in u_seq:
        x = x + u * C.dt  # 正确使用 x = x + v*dt
        traj.append(x)
    return traj
