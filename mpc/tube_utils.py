import numpy as np

def is_tube_safe(x_seq, y_tube, d_safe=0.5, car_y=12.0):
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

def is_trace_safe(x_seq, y_seq, d_safe=0.5, car_y=12.0):
    """
    判断未来 T 步车和行人 trace 是否保持安全距离。
    """
    # print(x_seq,y_seq)
    assert len(x_seq) == len(y_seq)

    for t in range(len(x_seq)):
        x = np.array([x_seq[t], car_y])
        y = np.array(y_seq[t])
        dist = np.linalg.norm(x - y)
        if dist < d_safe:
            return False
    return True