import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


CP_GRID_PATH = "assets/conformal_grid.pkl"
CP_COUNT_PATH = "assets/conformal_grid_counts.pkl"


def load_grid(path: str = CP_GRID_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Conformal grid not found at {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_counts(path: str = CP_COUNT_PATH):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_eta_vs_car_v(
    dim: str = 'x',
    car_x_bin: int = 0,
    walker_y_bin: int = 0,
    past_vx_bin: int = 2,
    past_vy_bin: int = 2,
    grid_path: str = CP_GRID_PATH,
):
    """
    6D 网格切片：固定 car_x / walker_y / past_vx / past_vy，沿 car_v (0..14) 画 η_x 或 η_y。
    """
    grid = load_grid(grid_path)
    K_CAR_V = 15
    eta_vals = []
    for j in range(K_CAR_V):
        key = (car_x_bin, j, 0, walker_y_bin, past_vx_bin, past_vy_bin)
        eta = grid.get(key, (0.0, 0.0))
        eta_vals.append(eta[0] if dim == 'x' else eta[1])

    xs = np.arange(K_CAR_V)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, eta_vals, marker='o')
    plt.xlabel('car_v bin (0..14)')
    plt.ylabel(f'eta_{dim}')
    plt.title(f'eta_{dim} vs car_v | car_x={car_x_bin}, wy={walker_y_bin}, pvx={past_vx_bin}, pvy={past_vy_bin}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_eta_heatmap_carx_carv(
    dim: str = 'x',
    walker_y_bin: int = 0,
    past_vx_bin: int = 2,
    past_vy_bin: int = 2,
    grid_path: str = CP_GRID_PATH,
    counts_overlay: bool = True,
):
    """
    6D 网格切片：固定 walker_y / past_vx / past_vy，绘制 (car_x, car_v) 的 η 热力图。
    """
    grid = load_grid(grid_path)
    counts = load_counts() if counts_overlay else None

    K_CAR_X = 40
    K_CAR_V = 15
    Z = np.zeros((K_CAR_X, K_CAR_V), dtype=float)

    for i in range(K_CAR_X):
        for j in range(K_CAR_V):
            key = (i, j, 0, walker_y_bin, past_vx_bin, past_vy_bin)
            eta = grid.get(key, (0.0, 0.0))
            Z[i, j] = eta[0] if dim == 'x' else eta[1]

    plt.figure(figsize=(9, 6))
    im = plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(im, label=f'eta_{dim}')
    plt.xlabel('car_v bin (0..14)')
    plt.ylabel('car_x bin (0..39)')
    plt.title(f'eta_{dim} heatmap over (car_x, car_v) | wy={walker_y_bin}, pvx={past_vx_bin}, pvy={past_vy_bin}')

    if counts is not None:
        # 仅在低计数时标记
        for i in range(K_CAR_X):
            for j in range(K_CAR_V):
                key = (i, j, 0, walker_y_bin, past_vx_bin, past_vy_bin)
                c = counts.get(key, 0)
                if c < 20:
                    plt.text(j, i, str(c), color='white', ha='center', va='center', fontsize=6)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 示例：按 car_v 绘制一个切片
    plot_eta_vs_car_v(dim='x', car_x_bin=10, walker_y_bin=3, past_vx_bin=2, past_vy_bin=2)
    # 示例：绘制 (car_x, car_v) 热力图
    plot_eta_heatmap_carx_carv(dim='y', walker_y_bin=3, past_vx_bin=2, past_vy_bin=2)
