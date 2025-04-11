from mpc.car_dynamics_batch import forward_car_batch
from mpc.ped_dynamics_batch import forward_ped_trace_batch
from mpc.tube_utils import is_trace_safe
from mpc.tube_utils import batch_is_trace_safe
import numpy as np

def mpc_control_vanilla_batch(x0, y0, T=10, N=100, d_safe=10.0):
    """
    批量版 vanilla MPC 控制器，使用行人轨迹而非 tube
    """
    # Fast check: all-15 sequence
    u_full_speed = np.full((1, T), 15.0)
    x_trace = forward_car_batch(x0, u_full_speed)[0]
    y_trace = forward_ped_trace_batch(y0, u_full_speed)[0]
    if is_trace_safe(x_trace, y_trace, d_safe):
        return 15.0

    # 采样控制序列
    u_seqs = np.random.uniform(0.0, 15.0, size=(N, T))
    x_traces = forward_car_batch(x0, u_seqs)             # (N, T)
    y_traces = forward_ped_trace_batch(y0, u_seqs)       # (N, T, 2)

    safe_flags = batch_is_trace_safe(x_traces, y_traces, d_safe)
    valid_indices = np.where(safe_flags)[0]

    if len(valid_indices) > 0:
        best_i = valid_indices[np.argmax(np.mean(u_seqs[valid_indices], axis=1))]
        return u_seqs[best_i, 0]
    else:
        return 0.0
