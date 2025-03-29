import numpy as np
import torch
import pandas as pd
import pickle
import os

from models.model_def import WalkerSpeedPredictor
from envs.simulator import data_collecting

CP_GRID_PATH = "assets/conformal_grid.pkl"
_CP_GRID = None  # 用于 lazy load

def build_conformal_grid(alpha=0.85, num_samples=40000):
    """ 构建 conformal region grid, {(i,j,k,m): (η_x, η_y)} """
    df = pd.DataFrame(data_collecting(num_samples),
                      columns=['car_speed', 'walker_y', 'walker_speed_x', 'walker_speed_y'])
    
    X = df[['car_speed', 'walker_y']].values
    y_true = df[['walker_speed_x', 'walker_speed_y']].values

    
    model = WalkerSpeedPredictor()
    model.load_state_dict(torch.load('assets/walker_speed_predictor.pth'))
    model.eval()
    pred = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()

    
    score_x = np.abs(pred[:, 0] - y_true[:, 0])
    score_y = np.abs(pred[:, 1] - y_true[:, 1])

    # 分区边界
    bins_car = np.linspace(0, 15, 16)
    bins_vy = np.linspace(0, 3, 4)
    bins_vx = np.linspace(0, 1, 2)
    bins_y = [200, 300, 450]

    grid = {}
    bin_count = {}
    for i in range(15):       # car speed
        for j in range(3):    # v_y
            for k in range(1):  # v_x
                for m in range(2):  # walker_y (前/后段)
                    mask = (
                        (X[:, 0] >= bins_car[i]) & (X[:, 0] < bins_car[i+1]) &
                        (pred[:, 1] >= bins_vy[j]) & (pred[:, 1] < bins_vy[j+1]) &
                        (pred[:, 0] >= bins_vx[k]) & (pred[:, 0] < bins_vx[k+1]) &
                        (X[:, 1] >= bins_y[m]) & (X[:, 1] < bins_y[m+1])
                    )
                    count = int(np.sum(mask))
                    bin_count[(i, j, k, m)] = count
                    if np.sum(mask) < 10:
                        grid[(i, j, k, m)] = (0.05, 0.05)  # fallback
                    else:
                        eta_par = np.quantile(score_x[mask], alpha)
                        eta_perp = np.quantile(score_y[mask], alpha)
                        grid[(i, j, k, m)] = (eta_par, eta_perp)
                    # print(f"Grid bin {(i,j,k,m)} has {np.sum(mask)} samples")

    with open("assets/conformal_grid_counts.pkl", "wb") as f:
        pickle.dump(bin_count, f)
    return grid


def get_eta(car_speed, v_x, v_y, walker_y):
    """ 从已加载的 grid 中获取 eta 值 """
    global _CP_GRID
    if _CP_GRID is None:
        if not os.path.exists(CP_GRID_PATH):
            raise FileNotFoundError(
                f"{CP_GRID_PATH} not found. Please run `scripts/gen_cp_grid.py` to generate it."
            )
        with open(CP_GRID_PATH, 'rb') as f:
            _CP_GRID = pickle.load(f)
            


    i = min(int(car_speed), 14)
    j = min(int(v_y), 2)
    k = 0  # v_x 只有一个 bin
    m = 0 if walker_y < 300 else 1
    return _CP_GRID.get((i, j, k, m), (0.5, 0.5))
