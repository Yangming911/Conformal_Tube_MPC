import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpc.ped_dynamics import forward_ped  
from mpc.car_dynamics import forward_car  
from mpc.tube_utils import is_tube_safe 

def test_forward_ped_like_sim():
    # y0 = np.array([400.0, 200.0])   # walker 起点
    y0 = np.array([400.0, 200.0])   # walker 中点
    x0 = 370                        # car 起点
    T = 10                         # 预测步长


    # u_seq = np.full(T, 7.0)
    # 使用 normal 分布近似中等车速
    u_seq = np.clip(np.random.normal(loc=5.0, scale=1.5, size=T), 0.0, 15.0)

    x_seq = forward_car(x0, u_seq)

    tube = forward_ped(y0, u_seq, cp_alpha=0.85)
    print("Is tube safe?", is_tube_safe(x_seq, tube, d_safe=5.0, car_y=210.0))
     
    # ✅ 打印每一步 tube 的大小
    print(f"Tube length: {len(tube)} (expected: {T})")
    for t, (lo, hi) in enumerate(tube):
        center = (lo + hi) / 2
        size = hi - lo
        print(f"Step {t}:")
        print(f"  Center = {np.round(center, 2)}")
        print(f"  Size   = {np.round(size, 3)}")

    # 📈 可视化
    centers = [(lo + hi) / 2 for (lo, hi) in tube]
    xs = [y0[0]] + [c[0] for c in centers]
    ys = [y0[1]] + [c[1] for c in centers]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(y0[0], y0[1], marker='*', color='blue', markersize=10, label='Pedestrain Initial Position')
    ax.plot(x_seq[2], 210, marker='*', color='red', markersize=10, label='Car Initial Position')
    ax.plot(xs, ys, marker='o', linestyle='-', color='blue', label='Tube Center Trajectory')

    for (lo, hi) in tube:
        width = hi[0] - lo[0]
        height = hi[1] - lo[1]
        rect = patches.Rectangle((lo[0], lo[1]), width, height,
                                 linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.2)
        ax.add_patch(rect)

    # 画出 car 的未来位置轨迹（y 固定为车道中心）

    ax.plot(x_seq[2:8], [210]*len(x_seq[2:8]), color='red', linestyle='-', marker='s',
        markersize=5, label='Car Trajectory')
    # 可选图例
    # ax.plot([], [], marker='s', color='red', linestyle='None', label='Car Position')

    ax.set_title("Predicted Walker Trajectory with Reachable Tubes", fontsize=14)
    ax.set_xlabel("$y_{\parallel}$", fontsize=12)
    ax.set_ylabel("$y_{\perp}$ ", fontsize=12)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_forward_ped_like_sim()
