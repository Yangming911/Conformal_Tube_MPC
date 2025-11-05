#!/usr/bin/env python3
"""
Run multiple simulator episodes and report average vehicle speed and collisions.

Uses envs.simulator's _initial_state, _step, _done, and _is_collision helpers
to ensure consistency with the environment dynamics.
"""

import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import cvxpy as cp

import utils.constants as C
from models_control.model_def import ACPCausalPedestrianPredictor
from tools.eval_runs_scp import plot_trajectory, calculate_smoothness_metrics

import argparse
import numpy as np
from tqdm import tqdm
import warnings
from datetime import datetime
from typing import List, Dict
import copy

from envs import simulator as sim
import time
import json
import torch
import matplotlib.pyplot as plt
import os
from typing import Tuple

@torch.no_grad()
def nn_predict_positions_multi(model: ACPCausalPedestrianPredictor, device: torch.device, past_p_ped0: np.ndarray) -> np.ndarray:
    """Predict for M pedestrians in batch.
    past_p_ped0: [M,T,2]
    Returns: [M,T,2]
    """
    T = past_p_ped0.shape[1]
    M = past_p_ped0.shape[0]
    past_p_ped0_tensor = torch.from_numpy(past_p_ped0.astype(np.float32)).to(device)
    pred = model(past_p_ped0_tensor)  # [M,T,2]
    return pred.cpu().numpy()


def monte_carlo_sampling(
    u_ref: np.ndarray, 
    C_eta: np.ndarray, 
    u_min: float, 
    u_max: float, 
    d_safe: float,
    p_veh_0: np.ndarray,
    p_ped: np.ndarray,
    max_iter: int = 10000,
    d_t: float = 0.1,
) -> np.ndarray:
    """Monte Carlo sampling for ACP problem.
    """
    T = 10
    M = p_ped.shape[0]
    def is_feasible(u_sample: np.ndarray) -> bool:
        """Check if the sample is feasible."""
        L = np.tril(np.ones((T, T)))
        diff_u = L @ u_sample * d_t
        p_veh_x = (diff_u + p_veh_0[0])
        p_veh_y = np.array([p_veh_0[1]] * T).reshape(-1, 1)
        for m in range(M):
            p_ped_x = p_ped[m, :, 0].reshape(-1, 1)
            p_ped_y = p_ped[m, :, 1].reshape(-1, 1)
            dis_y = p_veh_y - p_ped_y
            dis_x = p_veh_x - p_ped_x
            if np.any((dis_x**2 + dis_y**2) <= (C_eta.reshape(-1, 1) + d_safe)**2):
                return False
        return True
    if is_feasible(u_ref):
        return u_ref
    u_samples = []
    for _ in range(max_iter):
        u_sample = np.random.uniform(u_min, u_max, T)
        if is_feasible(u_sample):
            u_samples.append(u_sample)
    u_samples = np.array(u_samples)
    if u_samples.shape[0] == 0:
        return None
    u_opt = u_samples[np.argmin(np.linalg.norm(u_samples - u_ref, axis=1))]
    return u_opt

def optimize_u(u_ref: np.ndarray, 
    C_eta: np.ndarray, 
    u_min: float, 
    u_max: float, 
    d_safe: float, 
    p_veh_0: np.ndarray, 
    p_ped: np.ndarray, 
    max_iter: int = 1000, 
    d_t: float = 0.1, 
) -> np.ndarray:
    """Optimize u using scipy.optimize.minimize.
    """
    from scipy.optimize import minimize
    T = 10
    M = p_ped.shape[0]
    
    # Convert all relevant variables to numpy.float64
    u_ref = np.array(u_ref, dtype=np.float64)
    C_eta = np.array(C_eta, dtype=np.float64)
    u_min_np = np.float64(u_min)
    u_max_np = np.float64(u_max)
    d_safe_np = np.float64(d_safe)
    p_veh_0 = np.array(p_veh_0, dtype=np.float64)
    p_ped = np.array(p_ped, dtype=np.float64)
    d_t_np = np.float64(d_t)
    
    def objective(u_sample: np.ndarray) -> float:
        """Objective function for optimization."""
        return np.linalg.norm(u_sample - u_ref)
    
    def constraint(u_sample: np.ndarray) -> np.ndarray:
        """Constraint function for optimization that returns M*T constraints."""
        # Ensure u_sample is float64
        u_sample = np.array(u_sample, dtype=np.float64)
        
        L = np.tril(np.ones((T, T), dtype=np.float64))
        diff_u = L @ u_sample * d_t_np
        p_veh_x = (diff_u + p_veh_0[0])
        p_veh_y = np.array([p_veh_0[1]] * T, dtype=np.float64).reshape(-1, 1)
        
        # 创建一个数组来存储所有M*T个约束
        constraints = []
        
        for m in range(M):
            p_ped_x = p_ped[m, :, 0].reshape(-1, 1)
            p_ped_y = p_ped[m, :, 1].reshape(-1, 1)
            
            # 计算每个时间步的距离
            for t in range(T):
                # 计算车辆与行人在时间t的距离
                dis_x = p_veh_x[t] - p_ped_x[t]
                dis_y = p_veh_y[t] - p_ped_y[t]
                
                # 确保距离约束
                distance = dis_x**2 + dis_y**2
                constraints.append((distance - (d_safe_np + C_eta[t])**2)[0])
        
        # 转换为numpy.float64数组并返回
        return np.array(constraints, dtype=np.float64)
    
    cons = ({'type': 'ineq', 'fun': constraint})
    
    # Ensure initial guess is float64
    u_ref_float64 = np.array(u_ref, dtype=np.float64)
    
    # 在minimize函数调用时直接指定边界
    bounds = [(u_min_np, u_max_np) for _ in range(len(u_ref_float64))]
    res = minimize(objective, u_ref_float64, method='SLSQP', constraints=cons, bounds=bounds, options={'disp': False, 'maxiter': max_iter, 'ftol': 1e-6, 'eps': 1e-6})
    
    if res.success:
        return res.x
    else:
        return None

    

def run_mpc(num_episodes: int, max_steps_per_episode: int, horizon_T: int, control_T: int, num_pedestrians: int):
    rng = np.random.RandomState(123)
    u_ref = 15.0
    u_min = 0.0
    u_max = 15.0
    d_safe = 2.0
    u_init = 0.1
    method ="monte_carlo" # "monte_carlo"
    p2p = True
    model_path = os.path.join("assets_ACP/control_ped_model.pth")
    error_npy_path = os.path.join("assets_ACP/cp_errors_test.npy")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    hidden_dim = int(checkpoint.get('config', {}).get('hidden_dim', 128))
    num_layers = int(checkpoint.get('config', {}).get('num_layers', 2))
    dropout = float(checkpoint.get('config', {}).get('dropout', 0.1))
    model = ACPCausalPedestrianPredictor(p_dim=2, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    total_steps = 0
    total_plan_time = []
    speed_sum = 0.0
    episodes_with_collision = 0
    unsolver_count = 0
    acceleration_dict = {}
    for episode_idx in tqdm(range(num_episodes), desc="Running episodes"):
        # # Initial state from simulator
        car_speed = float(rng.uniform(1.0, 15.0))
        state = sim._initial_state_multi_pedestrian(car_speed, rng, num_pedestrians)

        # Initial state from file
        collided = False

        u_opt = None
        step_idx = 0
        states = []
        collide_state = None


        initial_eta = 0.85
        error_npy = np.load(error_npy_path)
        C_eta = np.array([np.quantile(error_npy[:,i], initial_eta) for i in range(error_npy.shape[1])])
        eta = [initial_eta]*horizon_T
        while step_idx < max_steps_per_episode:
            if sim._is_collision_multi_pedestrian(state):
                collide_state = state.copy()
                collided = True
            if sim._done_multi_pedestrian(state):
                break
            # 行人轨迹不够10个，continue
            if step_idx < horizon_T:
                u_t = u_ref
                state['car_v'] = u_t
                next_state, _ = sim._step_multi_pedestrian(state, rng, p2p=p2p)
                state = next_state
                total_steps += 1
                step_idx += 1
                states.append(copy.deepcopy(state))
                states[-1]['eta'] = 0
                states[-1]['pre_p_ped'] = np.array([[0]*num_pedestrians,[0]*num_pedestrians]).T
                continue

                

            last_u = np.full((horizon_T,), float(u_init), dtype=np.float64)
            p_veh_0 = np.array([state["car_x"], state["car_y"]], dtype=np.float32)
            u_0 = np.array([state["car_v"]], dtype=np.float32)



            t0 = time.perf_counter()
            p_ped_0_multi = np.array([[state["walker_x"][i], state["walker_y"][i]] for i in range(num_pedestrians)], dtype=np.float32)
            past_p_ped_multi = np.zeros((num_pedestrians, horizon_T, 2), dtype=np.float32)
            ppast_p_ped_multi = np.zeros((num_pedestrians, horizon_T, 2), dtype=np.float32)
            # 全用p_ped_0_multi初始化past_p_ped_multi和ppast_p_ped_multi
            for j in range(num_pedestrians):
                for t in range(horizon_T):
                    past_p_ped_multi[j, t] = p_ped_0_multi[j]
                    ppast_p_ped_multi[j, t] = p_ped_0_multi[j]
            length = max(len(states)-1, 0)
            for j in range(num_pedestrians):
                for t in range(length, max(length - horizon_T, 0), -1):
                    past_p_ped_multi[j, horizon_T - (length - t) - 1] = np.array([states[t]["walker_x"][j], states[t]["walker_y"][j]], dtype=np.float32)
            for j in range(num_pedestrians):
                for t in range(max(length-horizon_T,0), max(length - 2*horizon_T, 0), -1):
                    ppast_p_ped_multi[j, horizon_T - (max(length - horizon_T, 0) - t)-1] = np.array([states[t]["walker_x"][j], states[t]["walker_y"][j]], dtype=np.float32)
            past_pred_p_ped_multi = nn_predict_positions_multi(model, device, ppast_p_ped_multi) # [M, T, 2]
            pred_p_ped_multi = nn_predict_positions_multi(model, device, past_p_ped_multi) # [M, T, 2]
            # update eta
            gamma = 0.08
            eta_const = 0.1
            if step_idx >= horizon_T:
                mask = ((np.linalg.norm(past_p_ped_multi - past_pred_p_ped_multi, axis=2) - C_eta.reshape(1, -1)) < 0)
                for t in range(horizon_T):
                    if mask[:,t].all():
                        one_minus_eta = 1 - eta[t]
                        one_minus_eta = one_minus_eta + gamma*eta_const
                        eta[t] = 1 - one_minus_eta
                        eta[t] = min(max(0.05, eta[t]), 0.99)

                    else:
                        one_minus_eta = 1 - eta[t]
                        one_minus_eta = one_minus_eta + gamma*(eta_const - 1)
                        eta[t] = 1 - one_minus_eta
                        eta[t] = min(max(0.05, eta[t]), 0.99)

                C_eta = np.zeros(horizon_T)
                for t in range(horizon_T):
                    C_eta[t] = np.quantile(error_npy[:,t], eta[t])

            if method == "monte_carlo":
                u_opt = monte_carlo_sampling(
                    u_ref=np.array([u_ref] * horizon_T, dtype=np.float32), 
                    C_eta=C_eta,
                    u_min=u_min,
                    u_max=u_max,
                    d_safe=d_safe,
                    p_veh_0=p_veh_0,
                    p_ped=pred_p_ped_multi,
                )
            elif method == "opt":
                u_opt = optimize_u(
                    u_ref=np.array([u_ref] * horizon_T, dtype=np.float32), 
                    C_eta=C_eta, 
                    u_min=u_min, 
                    u_max=u_max, 
                    d_safe=d_safe,
                    p_veh_0=p_veh_0,
                    p_ped=pred_p_ped_multi,
                )
            if u_opt is None:
                unsolver_count += 1
                u_opt = last_u
            last_u = u_opt
            t1 = time.perf_counter()
            # metrics accumulation
            total_plan_time.append(t1 - t0)
            for i in range(control_T):
                u_t = float(u_opt[i])
                state["car_v"] = u_t
                next_state, _ = sim._step_multi_pedestrian(state, rng)
                speed_sum += float(state["car_v"])  # accumulate speed
                total_steps += 1
                state = next_state
                step_idx += 1
                states.append(copy.deepcopy(state))
                states[-1]['eta'] = C_eta[i]
                states[-1]['pre_p_ped'] = pred_p_ped_multi[:,i,:]

        if collided:
            episodes_with_collision += 1
        
        veh_speed = np.array([state["car_v"] for state in states])
        smoothness_metrics = calculate_smoothness_metrics(veh_speed)
        acceleration_dict[episode_idx] = smoothness_metrics
        # Only save the first episode as an example
        if episode_idx == 0:
            # Generate timestamp for filename (MMDDHHMM format)
            timestamp = datetime.now().strftime("%m%d%H%M")
            plot_trajectory(states, collide_state, f"trajectories/episode_0_{timestamp}.png")
    avg_speed = (speed_sum / total_steps) if total_steps > 0 else 0.0
    avg_plan_time_ms = (1000.0 * np.mean(total_plan_time) if total_plan_time else 0.0)
    acceleration_std_list = [metrics['acceleration_std'] for metrics in acceleration_dict.values()]
    acceleration_mean_abs_list = [metrics['acceleration_mean_abs'] for metrics in acceleration_dict.values()]
    jerk_std_list = [metrics['jerk_std'] for metrics in acceleration_dict.values()]
    avg_acceleration_std = np.mean(acceleration_std_list) if acceleration_std_list else 0.0
    avg_acceleration_mean_abs = np.mean(acceleration_mean_abs_list) if acceleration_mean_abs_list else 0.0
    avg_jerk_std = np.mean(jerk_std_list) if jerk_std_list else 0.0

    print(f"Total episodes: {num_episodes}")
    print(f"Total Steps: {total_steps/num_episodes:.2f}")
    print(f"Episodes with collision: {episodes_with_collision}/{num_episodes}({episodes_with_collision/num_episodes:.2%})")
    print(f"Average speed: {avg_speed:.2f} m/s")
    print(f"Average planning time: {avg_plan_time_ms:.2f} ms")
    print(f"Unsuccessfully solved episodes: {unsolver_count} / {total_steps} (ratio={unsolver_count/total_steps:.2%})")
    print(f"Average acceleration std dev: {avg_acceleration_std:.4f}")
    print(f"Average acceleration mean abs: {avg_acceleration_mean_abs:.4f}")
    print(f"Average jerk std dev: {avg_jerk_std:.4f}")

if __name__ == "__main__":
    num_episodes = 200
    max_steps_per_episode = 10000
    horizon_T = 10
    control_T = 10
    num_pedestrians = [1,5,9]
    for num_pedestrian in num_pedestrians:
        print(f"Running {num_episodes} episodes with {num_pedestrian} pedestrians")
        run_mpc(num_episodes, max_steps_per_episode, horizon_T, control_T, num_pedestrian)