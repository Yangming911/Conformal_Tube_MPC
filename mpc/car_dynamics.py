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

def forward_car_batch(x0, u_seqs):
    """
    Args:
        x0: float，初始位置
        u_seqs: ndarray of shape (N, T)，每行一组控制序列
    Returns:
        trajs: ndarray of shape (N, T)，每行是 car 的位置轨迹
    """
    u_seqs = np.atleast_2d(u_seqs)  # 自动变成 (N, T)
    x0s = np.full((u_seqs.shape[0],), x0)
    return np.cumsum(u_seqs, axis=1) + x0s[:, None]

