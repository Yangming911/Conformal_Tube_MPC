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
from envs.simulator import _is_collision as is_collision
from envs.simulator import _done as is_done
from tqdm import tqdm


def simulate_one_sequence(u_seq: np.ndarray, p_veh0: np.ndarray, p_ped0: np.ndarray, rng: np.random.RandomState = None) -> np.ndarray:
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


def simulate_one_case(u_0: float, p_veh0: np.ndarray, p_ped0: np.ndarray, rng: np.random.RandomState, method='random') -> Tuple[np.ndarray, np.ndarray]:
    state = {
        "car_x": float(p_veh0[0]),
        "car_y": float(p_veh0[1]),
        "car_v": float(u_0),  # constant per episode; will keep this value
        "walker_x": float(p_ped0[0]),
        "walker_y": float(p_ped0[1]),
        "walker_vx": float(C.WALKER_START_V_X),
        "walker_vy": float(C.WALKER_START_V_Y),
    }
    u_seq = []
    p_seq = []
    if method == 'random' or method == 'apf':
        u = u_0
    elif method == 'increase':
        u = rng.uniform(1.0, 2.0)
    elif method == 'decrease':
        u = rng.uniform(13.0, 15.0)
    elif method == 'full_speed':
        u = 15.0
    elif method == 'half_speed':
        u = rng.uniform(5.0, 10.0)
        
    while True:
        # Ensure car_v equals the constant episode speed (u_seq[t,0] is same for all t)
        state["car_v"] = float(u)
        next_state, _ = sim_step(state, rng=rng)
        u_seq.append([state["car_v"]])
        p_seq.append([next_state["walker_x"], next_state["walker_y"]])
        state = next_state

        # randomly change u
        if method == 'random':
            u = np.random.uniform(1.0, 15.0)

        # increase u
        elif method == 'increase':
            u += np.random.uniform(0.0, 1.0)
            u = min(u,15.0) if u<15.0 else 1.0

        # decrease u
        elif method == 'decrease':
            u -= np.random.uniform(0.0, 1.0)
            u = max(u,1.0) if u>1.0 else 15.0

        # APF u
        elif method == 'apf':
            from evalAPF_multi import apf_controller_multi_pedestrian
            import copy
            multi_state = copy.deepcopy(state)
            multi_state['walker_x'] = [multi_state['walker_x']]
            multi_state['walker_y'] = [multi_state['walker_y']]
            u = apf_controller_multi_pedestrian(multi_state, goal_x=C.CAR_RIGHT_LIMIT+1)
        
        # full speed u
        elif method == 'full_speed':
            u = 15.0

        # half speed u
        elif method == 'half_speed':
            u = rng.uniform(5.0, 10.0)

        flag = is_done(state)
        # if is_collision(state):
        #     u_seq = []
        #     p_seq = []
        #     u = u_0
        #     state = {
        #         "car_x": float(p_veh0[0]),
        #         "car_y": float(p_veh0[1]),
        #         "car_v": float(u_0),  # constant per episode; will keep this value
        #         "walker_x": float(p_ped0[0]),
        #         "walker_y": float(p_ped0[1]),
        #         "walker_vx": float(C.WALKER_START_V_X),
        #         "walker_vy": float(C.WALKER_START_V_Y),
        #     }
        if flag:
            break
    p_seq = np.expand_dims(np.array(p_seq, dtype=np.float32), axis=-1)
    return np.array(u_seq, dtype=np.float32), p_seq

def simulate_multi_ped(u_0: float, num_pedestrians: int, rng: np.random.RandomState, method='random', p2p=False) -> np.ndarray:
    from envs.simulator import _initial_state_multi_pedestrian as initial_state_multi_pedestrian
    from envs.simulator import _step_multi_pedestrian as sim_step_multi_pedestrian
    from envs.simulator import _done_multi_pedestrian as is_done_multi_pedestrian
    state = initial_state_multi_pedestrian(car_v=u_0, rng=rng, num_pedestrians=num_pedestrians)
    u_seq = []
    p_seq = []
    u = u_0
    while True:
        # Ensure car_v equals the constant episode speed (u_seq[t,0] is same for all t)
        state["car_v"] = float(u)
        next_state, _ = sim_step_multi_pedestrian(state, rng=rng, p2p=p2p)
        u_seq.append([state["car_v"]])
        p_seq.append([next_state["walker_x"], next_state["walker_y"]])
        state = next_state

        # randomly change u
        if method == 'random':
            u = np.random.uniform(1.0, 15.0)

        # increase u
        elif method == 'increase':
            u += np.random.uniform(0.0, 1.0)
            u = min(u,15.0) if u<15.0 else 1.0

        # decrease u
        elif method == 'decrease':
            u -= np.random.uniform(0.0, 1.0)
            u = max(u,1.0) if u>1.0 else 15.0

        # APF u
        elif method == 'apf':
            from evalAPF_multi import apf_controller_multi_pedestrian
            import copy
            multi_state = copy.deepcopy(state)
            u = apf_controller_multi_pedestrian(multi_state, goal_x=C.CAR_RIGHT_LIMIT+1)
        
        # full speed u
        elif method == 'full_speed':
            u = np.random.uniform(13.0, 15.0)
        
        # half speed u
        elif method == 'half_speed':
            u = np.random.uniform(5.0, 10.0)


        flag = is_done_multi_pedestrian(state)
        if flag:
            break

    return np.array(u_seq, dtype=np.float32), np.array(p_seq, dtype=np.float32)

def visual_data(u: np.ndarray, p_veh0: np.ndarray, p_ped0: np.ndarray, p_seq: np.ndarray, num_samples: int = 5) -> None:
    import matplotlib.pyplot as plt

    N = u.shape[0]
    sample_indices = np.random.choice(N, size=num_samples, replace=False)

    plt.figure(figsize=(6, 6))
    for idx in sample_indices:
        u_seq = u[idx]  # [T,1]
        p_veh0_i = p_veh0[idx]  # [2,]
        p_ped0_i = p_ped0[idx]  # [2,]
        p_seq_i = p_seq[idx]  # [T,2]
        T = u_seq.shape[0]
        # Simulate vehicle trajectory
        p_veh = np.zeros((T, 2), dtype=np.float32)
        p_veh[0] = p_veh0_i
        for t in range(1, T):
            p_veh[t, 0] = p_veh[t-1, 0] + u_seq[t-1, 0] * C.dt  # dt=0.1s
            p_veh[t, 1] = p_veh[t-1, 1]
        # Plot
        plt.subplot(3, 2, list(sample_indices).index(idx)+1)
        plt.plot(p_veh[:, 0], p_veh[:, 1], label='Vehicle Trajectory', color='blue')
        plt.plot(p_seq_i[:, 0], p_seq_i[:, 1], label='Pedestrian Trajectory', color='orange')
        plt.scatter(p_veh0_i[0], p_veh0_i[1], color='blue', marker='o', label='Vehicle Start')
        plt.scatter(p_ped0_i[0], p_ped0_i[1], color='orange', marker='x', label='Pedestrian Start')
        plt.title(f'Sample Index: {idx}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('sample_trajectories.png')
    plt.close()

def visual_sequence(u_seq: np.ndarray, p_seq: np.ndarray, index: int) -> None:
    import matplotlib.pyplot as plt
    os.makedirs('trajectories_1027', exist_ok=True)

    T = u_seq.shape[0]
    # Simulate vehicle trajectory
    p_veh = np.zeros((T, 2), dtype=np.float32)
    p_veh[0] = np.array([C.CAR_START_X, C.CAR_LANE_Y], dtype=np.float32)
    for t in range(1, T):
        p_veh[t, 0] = p_veh[t-1, 0] + u_seq[t-1, 0] * C.dt  # dt=0.1s
        p_veh[t, 1] = p_veh[t-1, 1]
    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(p_veh[:, 0], p_veh[:, 1], label='Vehicle Trajectory', color='blue')
    plt.scatter(p_veh[0, 0], p_veh[0, 1], color='blue', marker='o', label='Vehicle Start')
    for i in range(p_seq.shape[-1]):
        plt.plot(p_seq[:, 0,i], p_seq[:, 1,i], label='Pedestrian Trajectory', color='orange')
        plt.scatter(p_seq[0, 0,i], p_seq[0, 1,i], color='orange', marker='x', label='Pedestrian Start')
    plt.title(f'Sequence Visualization')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'trajectories_1027/trajectory_{index}.png')
    plt.close()

def collect_dataset(num_episodes: int, T: int, seed: int = 42, p2p: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    u_all = np.zeros((num_episodes, T, 1), dtype=np.float32)
    pveh0_all = np.zeros((num_episodes, 2), dtype=np.float32)
    pped0_all = np.zeros((num_episodes, 2), dtype=np.float32)
    pseq_all = np.zeros((num_episodes, T, 2), dtype=np.float32)
    cnt = 0

    for i in tqdm(range(num_episodes)):
        # Sample a single constant speed and fill the sequence
        # speed = float(rng.uniform(1.0, 15.0))
        # u_seq = np.full((T, 1), speed, dtype=np.float32)

        # random u_seq
        u_seq = rng.uniform(1.0, 15.0, size=(T, 1)).astype(np.float32)
        u_0 = u_seq[0,0]

        p_veh0 = np.array([float(C.CAR_START_X), float(C.CAR_LANE_Y)], dtype=np.float32)
        p_ped0 = np.array([
            float(C.WALKER_START_X),
            float(rng.uniform(C.WALKER_START_Y, C.CAR_LANE_Y)),
        ], dtype=np.float32)

        # p_seq = simulate_one_sequence(u_seq, p_veh0, p_ped0, rng)

        # u_all[i] = u_seq
        # pveh0_all[i] = p_veh0
        # pped0_all[i] = p_ped0
        # pseq_all[i] = p_seq

        # simulate until done # multi: 0.1, 0, 0, 0.7, 0.2, 0.0;   single: 0.2, 0.0, 0.0, 0.1, 0.7, 0.0;    multi p2p:
        methods = {'random':0.3,'increase':0.0,'decrease':0.0,'apf':0.4,'full_speed':0.3, 'half_speed':0.0}
        method = rng.choice(list(methods.keys()), p=list(methods.values()))
        u_seq,p_seq = simulate_one_case(u_0,p_veh0,p_ped0,rng,method=method)
        # u_seq, p_seq = simulate_multi_ped(u_0, num_pedestrians=rng.randint(1,9), rng=rng, method=method, p2p=p2p)
        visual_sequence(u_seq,p_seq,i) # visualize
        for p in range(p_seq.shape[-1]):
            for j in range(1,len(u_seq)-T):
                u_all[cnt] = np.array(u_seq[j:j+T], dtype=np.float32)

                pveh0_all[cnt] = p_veh0 + np.array([np.cumsum(u_seq[:j+1], axis=0)[-1][-1]*C.dt,0.0])
                pped0_all[cnt] = p_seq[j-1,:,p]
                pseq_all[cnt] = np.array(p_seq[j:j+T,:,p], dtype=np.float32)
                cnt += 1
                if cnt >= num_episodes:
                    break
            if cnt >= num_episodes:
                break
        if cnt >= num_episodes:
            break

    # visualize some samples
    visual_data(u_all, pveh0_all, pped0_all, pseq_all, num_samples=6)
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
    parser.add_argument('--save_path', type=str, default='assets/control_sequences_cp.csv', help='Save path for dataset (CSV)')
    args = parser.parse_args()

    print(f"Generating dataset: episodes={args.episodes}, T={args.T}")
    u, p_veh0, p_ped0, p_seq = collect_dataset(args.episodes, args.T, seed=args.seed)
    print(f"Shapes: u={u.shape}, p_veh0={p_veh0.shape}, p_ped0={p_ped0.shape}, p_seq={p_seq.shape}")

    save_csv(args.save_path, u, p_veh0, p_ped0, p_seq)
    print(f"Saved dataset to {args.save_path}")


if __name__ == '__main__':
    main()


