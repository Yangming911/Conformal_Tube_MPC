import numpy as np

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

