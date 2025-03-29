import numpy as np

def forward_car(x0, u_seq):
    """
    给定初始位置 x0 和控制序列 u_seq，生成未来 T 步的位置轨迹
    Args:
        x0: float, 当前车的位置
        u_seq: ndarray, 控制序列 u_0, ..., u_{T-1}
    Returns:
        List[float]，车每一步的 x_t
    """
    x = x0
    traj = []
    for u in u_seq:
        x = x + u
        traj.append(x)
    return traj
