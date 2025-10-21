#!/usr/bin/env python3
"""
Collect sequence dataset for control-to-pedestrian-position prediction using the Social Force model.

For each episode (sequence):
- Sample a constant vehicle speed u ~ Uniform[1, 15] m/s and keep it fixed for T steps (uniform speed).
- Set initial vehicle position p_veh0 = (CAR_START_X, CAR_LANE_Y).
- Sample initial pedestrian y uniformly between WALKER_START_Y and WALKER_DESTINATION_Y; set x=WALKER_START_X.
- Roll out T steps with walker_logic_SF to get positions p_ped[1..T].

Outputs
- CSV format: columns `u0..u{T-1}`, `p_veh0_x`, `p_veh0_y`, `p_ped0_x`, `p_ped0_y`,
  then `p_ped1_x`, `p_ped1_y`, ..., `p_ped{T}_x`, `p_ped{T}_y` (compatible with models_control/train.py)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.constants as C
from envs.simulator import _step as sim_step


def simulate_one_sequence(u_seq: np.ndarray, p_veh0: np.ndarray, p_ped0: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    T = u_seq.shape[0]
    # Initialize full simulator state
    state = {
        "car_x": float(p_veh0[0]),
        "car_y": float(p_veh0[1]),
        "car_v": float(u_seq[0, 0]),  # constant per episode; will keep this value
        "walker_x": float(p_ped0[0]),
        "walker_y": float(p_ped0[1]),
        "walker_vx": float(C.WALKER_START_V_X),
        "walker_vy": float(C.WALKER_START_V_Y),
    }

    out = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        # Ensure car_v equals the constant episode speed (u_seq[t,0] is same for all t)
        state["car_v"] = float(u_seq[t, 0])
        next_state, _ = sim_step(state, rng=rng)
        out[t, 0] = next_state["walker_x"]
        out[t, 1] = next_state["walker_y"]
        state = next_state
    return out


def collect_dataset(num_episodes: int, T: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    u_all = np.zeros((num_episodes, T, 1), dtype=np.float32)
    pveh0_all = np.zeros((num_episodes, 2), dtype=np.float32)
    pped0_all = np.zeros((num_episodes, 2), dtype=np.float32)
    pseq_all = np.zeros((num_episodes, T, 2), dtype=np.float32)

    for i in range(num_episodes):
        # Sample a single constant speed and fill the sequence
        speed = float(rng.uniform(1.0, 15.0))
        u_seq = np.full((T, 1), speed, dtype=np.float32)

        p_veh0 = np.array([float(C.CAR_START_X), float(C.CAR_LANE_Y)], dtype=np.float32)
        p_ped0 = np.array([
            float(C.WALKER_START_X),
            float(rng.uniform(C.WALKER_START_Y, C.WALKER_DESTINATION_Y)),
        ], dtype=np.float32)

        p_seq = simulate_one_sequence(u_seq, p_veh0, p_ped0, rng)

        u_all[i] = u_seq
        pveh0_all[i] = p_veh0
        pped0_all[i] = p_ped0
        pseq_all[i] = p_seq

    return u_all, pveh0_all, pped0_all, pseq_all


def save_csv(path: str, u: np.ndarray, p_veh0: np.ndarray, p_ped0: np.ndarray, p_seq: np.ndarray) -> None:
    N, T, _ = u.shape
    rows = {}
    for t in range(T):
        rows[f"u{t}"] = u[:, t, 0]
    rows["p_veh0_x"] = p_veh0[:, 0]
    rows["p_veh0_y"] = p_veh0[:, 1]
    rows["p_ped0_x"] = p_ped0[:, 0]
    rows["p_ped0_y"] = p_ped0[:, 1]
    for t in range(T):
        rows[f"p_ped{t+1}_x"] = p_seq[:, t, 0]
        rows[f"p_ped{t+1}_y"] = p_seq[:, t, 1]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Collect control sequences with constant speed per episode")
    parser.add_argument('--episodes', type=int, default=20000, help='Number of sequences to generate')
    parser.add_argument('--T', type=int, default=10, help='Sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='assets/control_sequences_cp_1021.csv', help='Save path for dataset (CSV)')
    args = parser.parse_args()

    print(f"Generating dataset: episodes={args.episodes}, T={args.T}")
    u, p_veh0, p_ped0, p_seq = collect_dataset(args.episodes, args.T, seed=args.seed)
    print(f"Shapes: u={u.shape}, p_veh0={p_veh0.shape}, p_ped0={p_ped0.shape}, p_seq={p_seq.shape}")

    save_csv(args.save_path, u, p_veh0, p_ped0, p_seq)
    print(f"Saved dataset to {args.save_path}")


if __name__ == '__main__':
    main()


