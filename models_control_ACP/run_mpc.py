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

import argparse
import numpy as np
from tqdm import tqdm
import warnings
from datetime import datetime
from typing import List, Dict

from envs import simulator as sim
import time
import json
import torch
import matplotlib.pyplot as plt
import os
from typing import Tuple

def plot_trajectory(states: List[Dict], collide_state: Dict, filename: str) -> None:
    """
    Plot the trajectory of the vehicle and pedestrians.
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    ped_traj = np.array([[state["walker_x"], state["walker_y"]] for state in states])
    veh_traj = np.array([[state["car_x"], state["car_y"]] for state in states])
    # 散点图，且点的颜色随时间变化
    colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
    plt.scatter(veh_traj[:, 0], veh_traj[:, 1], label="Vehicle", color=colors)
    plt.scatter(ped_traj[:, 0], ped_traj[:, 1], label=f"Pedestrian {1}",marker="^", color=colors)
    # 若发生碰撞，将碰撞点标红
    if collide_state:
        plt.scatter(collide_state["car_x"], collide_state["car_y"], label="Collision", color="red", s=100)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory")
    plt.savefig(filename)
    plt.close()

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

def solve_ack_problem(
    u_ref: np.ndarray, 
    u0: np.ndarray, 
    C_eta: np.ndarray, 
    u_min: float, 
    u_max: float, 
    rho_ref: float, 
    d_safe: float,
    p_veh_0: np.ndarray,
    p_ped: np.ndarray,
) -> np.ndarray:
    """Solve convex subproblem:
        minimize   ||u - u_ref||_2^2
    Returns optimal u (numpy [T]).
    
    """
    T = u0.shape[0]
    u = cp.Variable(T)
    # diff matrix
    D = np.eye(T) - np.roll(np.eye(T), 1, axis=0)
    diff_u = D @ u

    # objective function
    # objective = cp.sum_squares(u - u_ref) + rho_ref * cp.sum_squares(diff_u[1:]) #+ 0.1*cp.sum_squares(u - 5)
    objective = cp.sum_squares(u - u_ref)
    constraints = []
    # Linearized constraints for all (m,t) flattened
    # For each ped, p_veh-p_ped >= C_eta[t]
    L = np.tril(np.ones((T, T)))
    p_veh_x = np.eye(T) @ np.array([p_veh_0[0]]*T) + L @ u  # [T,2]
    p_veh_y = np.ones(T) * p_veh_0[1]
    M = p_ped.shape[0]
    # Remove the pdb debug statement if not needed
    # import pdb; pdb.set_trace()
    # 将所有的约束集中在一起，避免 t 的循环
    for m in range(M):
        # 计算 p_veh_x 和 p_veh_y 与 p_ped 的差异
        dx = p_veh_x - p_ped[m, :, 0]  # p_veh_x 和 p_ped_x 之间的差异
        dy = p_veh_y - p_ped[m, :, 1]  # p_veh_y 和 p_ped_y 之间的差异

        # 使用 vstack 合并 dx 和 dy，并计算其 Frobenius 范数
        norm_diff = cp.norm(cp.vstack([dx, dy]), 'fro', axis=0)
        
        # 添加约束：确保每个时间步的差异满足 ||(p_veh - p_ped)|| >= C_eta[t]
        constraints.append(norm_diff >= C_eta)
    constraints.append(u >= u_min)
    constraints.append(u <= u_max)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    # prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, max_iter=20000, warm_start=True, verbose=False)
    prob.solve(solver=cp.GUROBI)
    if u.value is None:
        raise RuntimeError("SCP subproblem infeasible or solver failed")
    return np.asarray(u.value, dtype=np.float64)

def acp_optimize(
    model_path: str,
    error_npy_path: str,
    p_ped_0_multi: np.ndarray,
    past_p_ped_multi: np.ndarray,
    p_veh_0: np.ndarray,
    u_0: np.ndarray,
    eta: float,
    rho_ref: float,
    d_safe: float,
    T: int = 10,
    outer_iters: int = 5,
    u_init: float = 0.1,
    u_ref: float = 15.0,
    u_min: float = 0.1,
    u_max: float = 15.0,
    rho_trust: float = 1.0,
    reject_stats_path: str = None,
    trust_region_initial: float = 10.0,
    trust_region_decay: float = 0.8,
    log_file: str = None,
) -> Tuple[np.ndarray, int, List[int], np.ndarray, np.ndarray]:
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

    error_npy = np.load(error_npy_path)

    dt = float(C.dt)
    last_u = np.full((T,), float(u_init), dtype=np.float64)
    ref_u = np.full((T,), float(u_ref), dtype=np.float64)
    
    # Freeze eta based on last_u
    C_eta = [np.quantile(error_npy[:,i], eta) for i in range(error_npy.shape[1])]

    p_ped = nn_predict_positions_multi(model, device, past_p_ped_multi)  # [M,T,2]

    u_new = solve_ack_problem(
        u_ref=ref_u, 
        u0=u_0, 
        C_eta=C_eta, 
        u_min=u_min, 
        u_max=u_max, 
        rho_ref=rho_ref, 
        d_safe=d_safe,
        p_veh_0=p_veh_0,
        p_ped=p_ped,
    )
    if u_new is None:
        print("ACP problem infeasible or solver failed")
        return last_u
    return u_new



def run_episodes_acp(
    num_episodes: int,
    max_steps_per_episode: int,
    horizon_T: int,
    outer_iters: int,
    seed: int,
    model_path: str,
    eta_csv: str,
    error_npy_path: str,
    u_init: float = 0.1,
    u_ref: float = 15.0,
    u_min: float = 1.0,
    u_max: float = 15.0,
    d_safe: float = 1.0,
    trust_region_initial: float = 10.0,
    trust_region_decay: float = 0.8,
    rho_ref: float = 2.0,
    num_pedestrians: int = 1,
    log_file: str = None,
    explicit_log: str = None,
    method: str = "scp",
) -> None:
    rng = np.random.RandomState(seed)
    episodes_with_collision = 0
    total_steps = 0
    speed_sum = 0.0
    total_plan_time = []
    total_plan_iters = []
    total_plan_inner_iters = []
    # Cumulative statistics matrices (3x3 for 3 bins)
    cumulative_reject_matrix = np.zeros((3, 3), dtype=np.int64)
    cumulative_transition_matrix = np.zeros((3, 3), dtype=np.int64)

    # Setup logging and warning capture
    old_warn_handler = None
    if log_file:
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Setup explicit log (parameters and results only)
    if explicit_log:
        import os
        os.makedirs(os.path.dirname(explicit_log), exist_ok=True)
        
        # Redirect warnings to log file
        def warning_to_log(message, category, filename, lineno, file=None, line=None):
            with open(log_file, 'a') as f:
                f.write(f"[WARNING] {category.__name__}: {message}\n")
        
        old_warn_handler = warnings.showwarning
        warnings.showwarning = warning_to_log
        
        # Write experiment header (append mode to preserve history)
        with open(log_file, 'a') as f:
            f.write("\n\n" + "="*70 + "\n")
            f.write("SCP Evaluation Experiment Log\n")
            f.write("="*70 + "\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Parameters:\n")
            f.write(f"  num_episodes: {num_episodes}\n")
            f.write(f"  max_steps_per_episode: {max_steps_per_episode}\n")
            f.write(f"  horizon_T: {horizon_T}\n")
            f.write(f"  outer_iters: {outer_iters}\n")
            f.write(f"  seed: {seed}\n")
            f.write(f"  model_path: {model_path}\n")
            f.write(f"  eta_csv: {eta_csv}\n")
            f.write(f"  u_init: {u_init}\n")
            f.write(f"  u_ref: {u_ref}\n")
            f.write(f"  u_min: {u_min}\n")
            f.write(f"  u_max: {u_max}\n")
            f.write(f"  d_safe: {d_safe}\n")
            f.write(f"  trust_region_initial: {trust_region_initial}\n")
            f.write(f"  trust_region_decay: {trust_region_decay}\n")
            f.write(f"  num_pedestrians: {num_pedestrians}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("Execution Log:\n")
            f.write("="*70 + "\n\n")

    # Initial state from file
    with open(f"assets/initial_state.json", "r") as f:
        initial_states = json.load(f)

    for episode_idx in tqdm(range(num_episodes), desc="Running episodes"):
        # # Initial state from simulator
        car_speed = float(rng.uniform(1.0, 15.0))
        state = sim._initial_state_multi_pedestrian(car_speed, rng, num_pedestrians)

        # Initial state from file
        # state = initial_states[episode_idx]
        collided = False

        u_opt = None
        step_idx = 0
        states = []
        collide_state = None

        initial_eta = 0.85
        while step_idx < max_steps_per_episode:
            states.append(state.copy())
            if sim._is_collision_multi_pedestrian(state):
                collide_state = state.copy()
                collided = True
            if sim._done_multi_pedestrian(state):
                break
            # 行人轨迹不够10个，continue
            if step_idx < horizon_T:
                u_t = u_ref
                state['car_v'] = u_t
                next_state, _ = sim._step_multi_pedestrian(state, rng)
                state = next_state
                total_steps += 1
                step_idx += 1
                continue


            p_veh_0 = np.array([state["car_x"], state["car_y"]], dtype=np.float32)
            u_0 = np.array([state["car_v"]], dtype=np.float32)

            p_ped_0_multi = np.array([[state["walker_x"][i], state["walker_y"][i]] for i in range(num_pedestrians)], dtype=np.float32)
            past_p_ped_multi = np.array([[states[i]["walker_x"], states[i]["walker_y"]] for i in range(step_idx - horizon_T, step_idx)], dtype=np.float32) # [M, T, 2]
            # [T,2,M] --> [M,T,2]
            past_p_ped_multi = past_p_ped_multi.transpose(2, 0, 1)
            t0 = time.perf_counter()
            u_opt, res_pre_p_ped = acp_optimize(
                model_path=model_path,
                error_npy_path=error_npy_path,
                p_veh_0=p_veh_0,
                p_ped_0_multi=p_ped_0_multi,
                past_p_ped_multi=past_p_ped_multi,
                u_0=u_0,
                eta=initial_eta,
                T=horizon_T,
                outer_iters=outer_iters,
                u_init=u_init,
                u_ref=u_ref,
                u_min=u_min,
                u_max=u_max,
                d_safe=d_safe,
                rho_ref=rho_ref,
                trust_region_initial=trust_region_initial,
                trust_region_decay=trust_region_decay,
                log_file=log_file,
            )
            t1 = time.perf_counter()
            # metrics accumulation
            total_plan_time.append(t1 - t0)

            # Apply control corresponding to position inside current horizon
            u_t = float(u_opt[step_idx % horizon_T])
            state["car_v"] = u_t
            next_state, _ = sim._step_multi_pedestrian(state, rng)
            speed_sum += float(state["car_v"])  # accumulate speed
            total_steps += 1
            state = next_state
            step_idx += 1

            if collided:
                episodes_with_collision += 1
            
            # Only save the first episode as an example
            if episode_idx == 0:
                # Generate timestamp for filename (MMDDHHMM format)
                timestamp = datetime.now().strftime("%m%d%H%M")
                plot_trajectory(states, collide_state, f"trajectories/episode_0_{timestamp}.png")


    avg_speed = (speed_sum / total_steps) if total_steps > 0 else 0.0
    avg_iters = (np.mean(total_plan_iters) if total_plan_iters else 0.0)
    avg_inner_iters = (np.mean(total_plan_inner_iters) if total_plan_inner_iters else 0.0)
    avg_plan_time_ms = (1000.0 * np.mean(total_plan_time) if total_plan_time else 0.0)
    
    total_outer_iters = sum(total_plan_iters)
    
    # Prepare results string
    results_lines = []
    results_lines.append(f"Episodes: {num_episodes}")
    results_lines.append(f"Method: {method}")
    results_lines.append(f"Total Steps: {total_steps/num_episodes:.2f}")
    results_lines.append(f"Avg vehicle speed over all steps: {avg_speed:.4f} m/s")
    results_lines.append(f"Episodes with collision: {episodes_with_collision} / {num_episodes} (ratio={episodes_with_collision/num_episodes:.3f})")
    results_lines.append(f"Avg outer-loop iterations per plan: {avg_iters:.2f}")
    results_lines.append(f"Avg inner SCP steps per outer step: {avg_inner_iters:.2f}")
    results_lines.append(f"Avg solve time per T-step plan: {avg_plan_time_ms:.2f} ms")
    results_lines.append("")
    results_lines.append(f"Total outer iterations: {total_outer_iters}")
    results_lines.append("")
    results_lines.append(f"\nAll Transitions Count Matrix (Row: last_u bin, Col: opt_u bin):")
    results_lines.append("Values show total counts")
    results_lines.append("       Bin0      Bin1      Bin2")
    for i in range(3):
        row_str = f"Bin{i} "
        for j in range(3):
            row_str += f"{cumulative_transition_matrix[i, j]:8d}  "
        results_lines.append(row_str)
    results_lines.append(f"\nReject Count Matrix (Row: last_u bin, Col: opt_u bin):")
    results_lines.append("Values show total counts")
    results_lines.append("       Bin0      Bin1      Bin2")
    for i in range(3):
        row_str = f"Bin{i} "
        for j in range(3):
            row_str += f"{cumulative_reject_matrix[i, j]:8d}  "
        results_lines.append(row_str)
    
    # Print to console
    for line in results_lines:
        print(line)
    
    # Write results to log
    if log_file:
        # Restore original warning handler
        if old_warn_handler is not None:
            warnings.showwarning = old_warn_handler
        
        with open(log_file, 'a') as f:
            f.write("\nExperiment Results\n")
            f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for line in results_lines:
                f.write(line + "\n")
    
    # Write explicit log (append mode - only parameters and results)
    if explicit_log:
        with open(explicit_log, 'a') as f:
            # Separator between experiments (only here as requested)
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Parameters:\n")
            f.write(f"  num_episodes: {num_episodes}\n")
            f.write(f"  max_steps_per_episode: {max_steps_per_episode}\n")
            f.write(f"  horizon_T: {horizon_T}\n")
            f.write(f"  outer_iters: {outer_iters}\n")
            f.write(f"  seed: {seed}\n")
            f.write(f"  model_path: {model_path}\n")
            f.write(f"  eta_csv: {eta_csv}\n")
            f.write(f"  u_init: {u_init}\n")
            f.write(f"  u_ref: {u_ref}\n")
            f.write(f"  u_min: {u_min}\n")
            f.write(f"  u_max: {u_max}\n")
            f.write(f"  d_safe: {d_safe}\n")
            f.write(f"  trust_region_initial: {trust_region_initial}\n")
            f.write(f"  trust_region_decay: {trust_region_decay}\n")
            f.write(f"  rho_ref: {rho_ref}\n")
            f.write(f"  num_pedestrians: {num_pedestrians}\n")
            f.write(f"  method: {method}\n\n")
            
            f.write("Results:\n")
            for line in results_lines:
                f.write(line + "\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SCP-controlled episodes")
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--steps', type=int, default=10000, help='Max steps per episode')
    parser.add_argument('--T', type=int, default=10, help='SCP horizon length')
    parser.add_argument('--outer_iters', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model_path', type=str, default='assets/control_ped_model_ACP.pth')
    parser.add_argument('--eta_csv', type=str, default='assets/cp_eta_real_sim.csv')
    parser.add_argument('--u_init', type=float, default=0.1)
    parser.add_argument('--u_ref', type=float, default=15.0)
    parser.add_argument('--u_min', type=float, default=0.0)
    parser.add_argument('--u_max', type=float, default=15.0)
    parser.add_argument('--d_safe', type=float, default=2.0)
    parser.add_argument('--trust_region_initial', type=float, default=5.0, help='Initial trust region radius')
    parser.add_argument('--trust_region_decay', type=float, default=0.5, help='Trust region decay rate per inner iteration')
    parser.add_argument('--rho_ref', type=float, default=1.5*1e7, help='Weight on control smoothness term')
    parser.add_argument('--num_pedestrians', type=int, default=9, help='Number of pedestrians for constraints')
    parser.add_argument('--log_file', type=str, default='logs/scp_eval_complicated.log', help='Log file path')
    parser.add_argument('--explicit_log', type=str, default='logs/scp_eval_explicit.log', help='Explicit log file path (parameters and results only)')
    parser.add_argument('--method', type=str, default='scp', help='Method to use for control: scp or constant_speed')
    parser.add_argument('--error_npy_path', type=str, default='assets/cp_errors_ACP.npy')

    args = parser.parse_args()

    run_episodes_acp(
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        horizon_T=args.T,
        outer_iters=args.outer_iters,
        seed=args.seed,
        model_path=args.model_path,
        eta_csv=args.eta_csv,
        error_npy_path=args.error_npy_path,
        u_init=args.u_init,
        u_ref=args.u_ref,
        u_min=args.u_min,
        u_max=args.u_max,
        d_safe=args.d_safe,
        trust_region_initial=args.trust_region_initial,
        trust_region_decay=args.trust_region_decay,
        rho_ref=args.rho_ref,
        num_pedestrians=args.num_pedestrians,
        log_file=args.log_file,
        explicit_log=args.explicit_log,
        method=args.method,
    )


if __name__ == '__main__':
    main()