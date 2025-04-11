import numpy as np
from mpc.car_dynamics_batch import forward_car_batch
from mpc.ped_dynamics_batch import forward_ped_batch
from mpc.tube_utils import is_tube_safe
from mpc.tube_utils import batch_is_tube_safe


def mpc_control_parallel(x0, y0, T=10, N=100, d_safe=10.0):
    """
    Tube-based MPC 控制器（批量版本）
    
    Args:
        x0: float，当前汽车位置
        y0: ndarray shape (2,) 当前行人位置
        T: int，MPC 预测步长
        N: int，采样控制序列数量
        d_safe: float，安全距离

    Returns:
        u0: float，下一个时刻的控制输入
    """
    # 首先尝试全速是否安全
    u_full_speed = np.full((1, T), 15.0)  # shape (1, T)
    x_seq = forward_car_batch(x0, u_full_speed)[0]     # (T,)
    y_tube = forward_ped_batch(y0, u_full_speed)[0]    # List[(lo, hi)]

    if is_tube_safe(x_seq, y_tube, d_safe):
        return 15.0

    # 采样 N 组控制序列，shape: (N, T)
    u_seqs = np.random.uniform(low=0.0, high=15.0, size=(N, T))
    x_seqs = forward_car_batch(x0, u_seqs)            # shape: (N, T)
    y_tubes = forward_ped_batch(y0, u_seqs)           # List[List[(lo, hi)]]

    best_u = None
    best_score = -np.inf

    # for i in range(N):
    #     if is_tube_safe(x_seqs[i], y_tubes[i], d_safe):
    #         score = np.mean(u_seqs[i])  # 越快越好
    #         if score > best_score:
    #             best_score = score
    #             best_u = u_seqs[i, 0]

    # return best_u if best_u is not None else 0.0

    safe_flags = batch_is_tube_safe(x_seqs, y_tubes, d_safe)
    valid_indices = np.where(safe_flags)[0]

    if len(valid_indices) > 0:
        best_i = valid_indices[np.argmax(np.mean(u_seqs[valid_indices], axis=1))]
        best_u = u_seqs[best_i, 0]
    else:
        best_u = 0.0
        
    return best_u if best_u is not None else 0.0

