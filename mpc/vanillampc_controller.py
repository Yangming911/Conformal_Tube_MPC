import numpy as np
from mpc.car_dynamics import forward_car
from mpc.ped_dynamics import forward_ped_trace
from mpc.tube_utils import is_trace_safe

def sample_control_sequence(T, u_min=0.0, u_max=15.0):
    """采样一组长度为 T 的控制序列"""
    return np.random.uniform(low=u_min, high=u_max, size=T)

def mpc_control_vanilla(x0, y0, T=10, N=100, d_safe=0.5):
    """
    Tube-based MPC 控制器（基于蒙特卡洛采样）
    
    Args:
        x0: 当前汽车位置（float，1D）
        y0: 当前行人位置 [x, y]（2D）
        T: MPC 预测步长
        N: 采样控制序列数量
        d_safe: 安全距离阈值

    Returns:
        u0: 下一时刻的控制量（float）
    """
    best_u = None
    best_score = -np.inf

    # 优化：如果[15, 15, ..., 15]是安全的，直接返回
    if is_trace_safe(forward_car(x0, np.full(T, 15.0)), forward_ped_trace(y0, np.full(T, 15.0)), d_safe):
        return 15.0
        
    for _ in range(N):
        u_seq = sample_control_sequence(T)
        x_trace = forward_car(x0, u_seq)           # List of car x positions
        y_trace = forward_ped_trace(y0, x_trace)          # List of sets (e.g., box or sample cloud)

        if not is_trace_safe(x_trace, y_trace, d_safe):
            continue

        # score = -np.mean((u_seq - 15)**2)         # 奖励靠近最大速度
        score = np.mean(u_seq)
        if score > best_score:
            best_score = score
            best_u = u_seq[0]
            
    # 没有可行解则保守控制
    return best_u if best_u is not None else 0.0

