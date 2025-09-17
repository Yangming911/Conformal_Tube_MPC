import numpy as np
import torch
import pandas as pd
import pickle
import os
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Tuple

from envs.simulator import collect_dataset
from models.predictor import WalkerActionPredictor
import utils.constants as C
import argparse

CP_GRID_PATH = "assets/conformal_grid.pkl"
_CP_GRID_CACHE = {}  # lazy load cache for different alpha values

# Conformal prediction parameters
DEFAULT_ALPHA = 0.85
DEFAULT_MIN_COUNT_PER_BIN = 20
DEFAULT_FALLBACK_ETA_X = 0.341
DEFAULT_FALLBACK_ETA_Y = 1.605
DEFAULT_CALIB_EPISODES = 2500
DEFAULT_CALIB_STEPS_PER_EPISODE = 10000
DEFAULT_CALIB_SEED = 2025


def _collect_dataframe() -> pd.DataFrame:
    """
    Collect a FRESH calibration dataset distinct from training data.
    Always collects new data and overwrites existing file.
    """
    calib_csv = "assets/sf_dataset_calib.csv"
    df = collect_dataset(
        num_episodes=DEFAULT_CALIB_EPISODES,
        max_steps_per_episode=DEFAULT_CALIB_STEPS_PER_EPISODE,
        seed=DEFAULT_CALIB_SEED,
        save_path=calib_csv,
    )
    return df


def build_conformal_grid(
    alpha: float = DEFAULT_ALPHA,
    model_path: str = "assets/best_model.pth",
    min_count_per_bin: int = DEFAULT_MIN_COUNT_PER_BIN,
    calib_csv_path: str = None,
) -> Dict[Tuple[int, int, int, int, int, int], Tuple[float, float]]:
    """
    构建 conformal region grid, key: (i_car_x, i_car_v, i_walker_x, i_walker_y, i_past_vx, i_past_vy)

    维度建议：40 x 15 x 1 x 10 x 5 x 5
    """

    # Load data. If external calibration CSV is provided, use it; otherwise collect fresh.
    if calib_csv_path is not None:
        df = pd.read_csv(calib_csv_path)
    else:
        df = _collect_dataframe()

    # Prepare features for model prediction using a unified predictor wrapper
    predictor = WalkerActionPredictor(model_path=model_path, device="cpu")

    # Compute model predictions for next-step pedestrian velocity
    preds_vx = []
    preds_vy = []
    for _, row in df.iterrows():
        vx_pred, vy_pred = predictor.predict(
            car_x=row["car_x"],
            car_y=row["car_y"],
            car_v=row["car_v"],
            walker_x=row["walker_x"],
            walker_y=row["walker_y"],
            walker_vx=row["walker_vx"],
            walker_vy=row["walker_vy"],
        )
        preds_vx.append(vx_pred)
        preds_vy.append(vy_pred)

    preds_vx = np.asarray(preds_vx, dtype=np.float32)
    preds_vy = np.asarray(preds_vy, dtype=np.float32)

    # True next-step velocities from dataset
    true_next_vx = df["next_walker_vx"].to_numpy(dtype=np.float32)
    true_next_vy = df["next_walker_vy"].to_numpy(dtype=np.float32)

    # Conformal scores (absolute errors)
    score_x = np.abs(preds_vx - true_next_vx)
    score_y = np.abs(preds_vy - true_next_vy)

    # Define bin edges per constants and requirements
    car_x_edges = np.linspace(float(C.CAR_LEFT_LIMIT), float(C.CAR_RIGHT_LIMIT), 41)  # 40 bins
    car_v_edges = np.linspace(0.0, 15.0, 16)  # 15 bins
    walker_x_edges = np.array([float(df["walker_x"].min()) - 1e-6, float(df["walker_x"].max()) + 1e-6])  # 1 bin
    walker_y_edges = np.linspace(float(C.WALKER_START_Y), float(C.WALKER_DESTINATION_Y), 11)  # 10 bins
    v_edges = np.linspace(-float(C.v_max), float(C.v_max), 6)  # 5 bins for both vx, vy

    # Extract features to bin
    car_x_vals = df["car_x"].to_numpy(dtype=np.float32)
    car_v_vals = df["car_v"].to_numpy(dtype=np.float32)
    walker_x_vals = df["walker_x"].to_numpy(dtype=np.float32)
    walker_y_vals = df["walker_y"].to_numpy(dtype=np.float32)
    past_vx_vals = df["walker_vx"].to_numpy(dtype=np.float32)
    past_vy_vals = df["walker_vy"].to_numpy(dtype=np.float32)

    # Digitize into bin indices
    i_car_x = np.clip(np.digitize(car_x_vals, car_x_edges) - 1, 0, len(car_x_edges) - 2)
    i_car_v = np.clip(np.digitize(car_v_vals, car_v_edges) - 1, 0, len(car_v_edges) - 2)
    i_walker_x = np.zeros_like(i_car_x)  # single bin
    i_walker_y = np.clip(np.digitize(walker_y_vals, walker_y_edges) - 1, 0, len(walker_y_edges) - 2)
    i_past_vx = np.clip(np.digitize(past_vx_vals, v_edges) - 1, 0, len(v_edges) - 2)
    i_past_vy = np.clip(np.digitize(past_vy_vals, v_edges) - 1, 0, len(v_edges) - 2)

    # Aggregate scores per bin and compute quantiles
    grid: Dict[Tuple[int, int, int, int, int, int], Tuple[float, float]] = {}
    bin_count: Dict[Tuple[int, int, int, int, int, int], int] = {}

    # Iterate over unique bins present in data to avoid nested loops over empty bins
    keys = np.stack([i_car_x, i_car_v, i_walker_x, i_walker_y, i_past_vx, i_past_vy], axis=1)
    # Use structured array for unique rows
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)

    for idx, key in enumerate(map(tuple, uniq)):
        mask = inverse == idx
        cnt = int(counts[idx])
        bin_count[key] = cnt
        if cnt < min_count_per_bin:
            grid[key] = (DEFAULT_FALLBACK_ETA_X, DEFAULT_FALLBACK_ETA_Y)
        else:
            eta_x = float(np.quantile(score_x[mask], alpha))
            eta_y = float(np.quantile(score_y[mask], alpha))
            grid[key] = (eta_x, eta_y)

    with open("assets/conformal_grid_counts.pkl", "wb") as f:
        pickle.dump(bin_count, f)

    return grid


def get_eta(car_x: float, car_v: float, walker_x: float, walker_y: float, walker_vx: float, walker_vy: float, cp_alpha: float = DEFAULT_ALPHA):
    """从已加载的 grid 中获取 eta 值，按照 40x15x1x10x5x5 网格索引。"""
    global _CP_GRID_CACHE
    
    # Generate file path based on cp_alpha
    cp_grid_path = f"assets/conformal_grid_{cp_alpha:.2f}.pkl"
    
    # Check if grid is already cached for this cp_alpha
    if cp_alpha not in _CP_GRID_CACHE:
        if not os.path.exists(cp_grid_path):
            raise FileNotFoundError(
                f"{cp_grid_path} not found. Please run the conformal grid generation with alpha={cp_alpha} to generate it."
            )
        with open(cp_grid_path, 'rb') as f:
            _CP_GRID_CACHE[cp_alpha] = pickle.load(f)

    # Build the same bin edges used during grid construction
    car_x_edges = np.linspace(float(C.CAR_LEFT_LIMIT), float(C.CAR_RIGHT_LIMIT), 41)
    car_v_edges = np.linspace(0.0, 15.0, 16)
    walker_x_edges = np.array([walker_x - 1e-3, walker_x + 1e-3])  # single bin placeholder
    walker_y_edges = np.linspace(float(C.WALKER_START_Y), float(C.WALKER_DESTINATION_Y), 11)
    v_edges = np.linspace(-float(C.v_max), float(C.v_max), 6)

    i = int(np.clip(np.digitize([car_x], car_x_edges)[0] - 1, 0, len(car_x_edges) - 2))
    j = int(np.clip(np.digitize([car_v], car_v_edges)[0] - 1, 0, len(car_v_edges) - 2))
    k = 0
    m = int(np.clip(np.digitize([walker_y], walker_y_edges)[0] - 1, 0, len(walker_y_edges) - 2))
    p = int(np.clip(np.digitize([walker_vx], v_edges)[0] - 1, 0, len(v_edges) - 2))
    q = int(np.clip(np.digitize([walker_vy], v_edges)[0] - 1, 0, len(v_edges) - 2))

    return _CP_GRID_CACHE[cp_alpha].get((i, j, k, m, p, q), (0.5, 0.5))


def _cli():
    parser = argparse.ArgumentParser(description="Collect calibration dataset (if missing) and build/save CP grid")
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help='Quantile level for conformal region')
    parser.add_argument('--model_path', type=str, default='assets/best_model.pth', help='Model path for predictions')
    parser.add_argument('--calib_csv', type=str, default='assets/sf_dataset_calib.csv', help='Calibration CSV path')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save CP grid (default: assets/conformal_grid_{alpha:.2f}.pkl)')
    parser.add_argument('--seed', type=int, default=DEFAULT_CALIB_SEED, help='Seed for calibration collection when needed')
    args = parser.parse_args()

    # Always collect fresh calibration data
    print(f"Collecting calibration dataset: episodes={DEFAULT_CALIB_EPISODES}, steps={DEFAULT_CALIB_STEPS_PER_EPISODE}, seed={args.seed}")
    df = collect_dataset(num_episodes=DEFAULT_CALIB_EPISODES, max_steps_per_episode=DEFAULT_CALIB_STEPS_PER_EPISODE, seed=args.seed, save_path=args.calib_csv)
    print(f"Calibration dataset saved to {args.calib_csv} with shape {df.shape}")

    print(f"Generating CP grid with alpha={args.alpha} ...")
    grid = build_conformal_grid(alpha=args.alpha, model_path=args.model_path, calib_csv_path=args.calib_csv)

    # Use default save path if not provided
    if args.save_path is None:
        args.save_path = f"assets/conformal_grid_{args.alpha:.2f}.pkl"
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'wb') as f:
        pickle.dump(grid, f)
    print(f"Conformal grid saved to {args.save_path}")


if __name__ == '__main__':
    _cli()
