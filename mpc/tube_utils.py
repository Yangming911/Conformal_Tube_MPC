import numpy as np

def is_tube_safe(x_seq, y_tube, d_safe=10.0, car_y=320.0):
    """
    判断未来 T 步车和行人 tube 是否保持安全距离。
    """
    assert len(x_seq) == len(y_tube)

    for t in range(len(x_seq)):
        x = np.array([x_seq[t], car_y])
        y_lo, y_hi = y_tube[t]
        y_clipped = np.clip(x, y_lo, y_hi)
        dist = np.linalg.norm(x - y_clipped)
        if dist < d_safe:
            # print(f"Collision at step {t}! Distance = {dist}")
            return False
    return True

import numpy as np

def batch_is_tube_safe(x_seqs, y_tubes, d_safe=10.0, car_y=320.0):
    """
    批量判断 N 条 (T 步) 的 car-pedestrian tube 是否安全。
    
    Args:
        x_seqs: ndarray (N, T)，每条控制序列对应的 car x 轨迹
        y_tubes: List[List[(lo, hi)]], shape (N, T)
        d_safe: float，安全距离
        car_y: float，汽车固定纵坐标

    Returns:
        safe_flags: ndarray (N,) of bool，是否安全
    """
    N, T = x_seqs.shape
    safe_flags = np.ones(N, dtype=bool)

    for i in range(N):
        for t in range(T):
            x = np.array([x_seqs[i, t], car_y])
            y_lo, y_hi = y_tubes[i][t]

            # clip x to box
            y_clipped = np.clip(x, y_lo, y_hi)
            dist = np.linalg.norm(x - y_clipped)

            if dist < d_safe:
                safe_flags[i] = False
                break  # Early exit for this sample

    return safe_flags


def is_trace_safe(x_seq, y_seq, d_safe=10.0, car_y=320.0):
    """
    判断未来 T 步车和行人 trace 是否保持安全距离。
    """
    # print(x_seq,y_seq)
    assert len(x_seq) == len(y_seq)

    for t in range(len(x_seq)):
        x = np.array([x_seq[t], 320.0])
        y = np.array(y_seq[t])
        dist = np.linalg.norm(x - y)
        if dist < d_safe:
            return False
    return True

def batch_is_trace_safe(x_seqs, y_traces, d_safe=10.0, car_y=320.0):
    """
    批量判断是否安全：每组 car 轨迹 x_seq 与对应行人轨迹 y_trace
    Args:
        x_seqs: (N, T)
        y_traces: (N, T, 2)
    Returns:
        safe_flags: (N,) bool
    """
    N, T = x_seqs.shape
    car_pos = np.stack([x_seqs, np.full_like(x_seqs, car_y)], axis=-1)  # (N, T, 2)
    dist = np.linalg.norm(car_pos - y_traces, axis=-1)  # (N, T)
    min_dist = np.min(dist, axis=1)  # (N,)
    return min_dist >= d_safe
