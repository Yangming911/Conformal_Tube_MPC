# visualizer/conformal_viz.py

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

CP_GRID_PATH = "assets/conformal_grid.pkl"

def load_grid():
    if not os.path.exists(CP_GRID_PATH):
        raise FileNotFoundError(f"Conformal grid not found at {CP_GRID_PATH}")
    with open(CP_GRID_PATH, 'rb') as f:
        return pickle.load(f)

def plot_eta_vs_car_speed(dim='x', vy_bin=1, walker_bin=0):
    """
    横轴：car speed bin (0-14)
    纵轴：η_x 或 η_y
    walker_bin: 0 = 前段, 1 = 后段
    """
    grid = load_grid()
    eta_list = []
    for i in range(15):  # car speed bins
        key = (i, vy_bin, 0, walker_bin)
        eta = grid.get(key, (0, 0))
        eta_val = eta[0] if dim == 'x' else eta[1]
        eta_list.append(eta_val)

    x = np.linspace(0.5, 14.5, 15)
    plt.figure(figsize=(6, 4))
    plt.plot(x, eta_list, marker='o', label=f'$\\eta_{{{dim}}}$ | zone={walker_bin}, v_y bin={vy_bin}')
    plt.xlabel('Car speed bin')
    plt.ylabel(f'Conformal radius $\\eta_{{{dim}}}$')
    plt.title(f'CP Region $\\eta_{{{dim}}}$ vs Car Speed (walker zone {walker_bin})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
