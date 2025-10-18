#!/usr/bin/env python3
"""
Sequential Convex Programming (SCP) for control u over horizon T with circular separation constraints.

Inputs (single pedestrian version; extendable to M pedestrians):
- p_veh_0: np.ndarray shape [2] vehicle initial position (x, y)
- v_veh_0: float initial vehicle speed (unused for constant-speed per-step model; u provides speed)
- p_ped_0: np.ndarray shape [2] pedestrian initial position (x, y)
- model: trained CausalPedestrianPredictor that maps (u[0..T-1], p_veh_0, p_ped_0) -> p_ped[1..T]
- eta table: CSV produced by models_control/cp.py with shape [T, 3] (columns per car_v bin)

Optimization:
  Outer loop freezes eta(u) by binning per-step u_t into 3 bins, then solves a convex subproblem
  using first-order linearization of g_t(u) = ||p_veh_t(u) - p_ped_t(u)|| around last u.
  Solver: OSQP via cvxpy.

Notes:
- Vehicle kinematics: x_{t+1} = x_t + u_t * dt, y fixed.
- Binning edges for u_t: [0,5), [5,10), [10,15].
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List

# Ensure project root on path BEFORE importing project modules
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import cvxpy as cp

import utils.constants as C
from models_control.model_def import CausalPedestrianPredictor


def load_eta_csv(path: str) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != "t"]
    eta = df[cols].to_numpy(dtype=np.float32)  # [T,B]
    # Try to parse edges from column names of form: bin{i}_[L,R)
    edges: List[Tuple[float, float]] = []
    for name in cols:
        try:
            bracket = name.split('_[')[1]
            nums = bracket.rstrip(')').split(',')
            L = float(nums[0])
            R = float(nums[1])
            edges.append((L, R))
        except Exception:
            edges = []
            break
    return eta, edges


def bin_index_for_speed(u_t: float, edges: List[Tuple[float, float]]) -> int:
    if edges:
        # edges are [(L0,R0),...,(L_{B-1},R_{B-1})]
        for i, (L, R) in enumerate(edges):
            if (u_t >= L) and (u_t < R or (i == len(edges)-1 and u_t <= R)):
                return i
        return max(0, len(edges) - 1)
    # Fallback to 3 default bins if edges absent
    if u_t < 5.0:
        return 0
    if u_t < 10.0:
        return 1
    return 2


def eta_of_u(eta_table: np.ndarray, u: np.ndarray, edges: List[Tuple[float, float]]) -> np.ndarray:
    """Return per-step eta by binning each u_t.
    eta_table: [T,B]; u: [T]
    Returns: [T]
    """
    T = u.shape[0]
    out = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        idx = bin_index_for_speed(float(u[t]), edges)
        out[t] = float(eta_table[t, idx])
    return out


@torch.no_grad()
def nn_predict_positions(model: CausalPedestrianPredictor, device: torch.device, u: np.ndarray, p_veh0: np.ndarray, p_ped0: np.ndarray) -> np.ndarray:
    """Single pedestrian prediction: returns [T,2]."""
    T = u.shape[0]
    u_tensor = torch.from_numpy(u.reshape(1, T, 1).astype(np.float32)).to(device)
    pveh0_tensor = torch.from_numpy(p_veh0.reshape(1, 2).astype(np.float32)).to(device)
    pped0_tensor = torch.from_numpy(p_ped0.reshape(1, 2).astype(np.float32)).to(device)
    pred = model(u_tensor, pveh0_tensor, pped0_tensor)  # [1,T,2]
    return pred.squeeze(0).cpu().numpy()


@torch.no_grad()
def nn_predict_positions_multi(model: CausalPedestrianPredictor, device: torch.device, u: np.ndarray, p_veh0: np.ndarray, p_ped0_multi: np.ndarray) -> np.ndarray:
    """Predict for M pedestrians in batch.
    u: [T], p_veh0: [2], p_ped0_multi: [M,2]
    Returns: [M,T,2]
    """
    T = u.shape[0]
    M = p_ped0_multi.shape[0]
    u_tensor = torch.from_numpy(u.reshape(1, T, 1).astype(np.float32)).repeat(M, 1, 1).to(device)
    pveh0_tensor = torch.from_numpy(p_veh0.reshape(1, 2).astype(np.float32)).repeat(M, 1).to(device)
    pped0_tensor = torch.from_numpy(p_ped0_multi.astype(np.float32)).to(device)
    pred = model(u_tensor, pveh0_tensor, pped0_tensor)  # [M,T,2]
    return pred.cpu().numpy()


def vehicle_trajectory(p_veh0: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Compute vehicle positions p_veh[1..T] for per-step speed u_t.
    p_veh0: [2], u: [T] (speed in m/s), dt seconds
    Returns: [T,2]
    """
    T = u.shape[0]
    p = np.zeros((T, 2), dtype=np.float32)
    x = float(p_veh0[0])
    y = float(p_veh0[1])
    for t in range(T):
        x = x + float(u[t]) * dt
        p[t, 0] = x
        p[t, 1] = y
    return p


def finite_diff_grad_multi(model: CausalPedestrianPredictor, device: torch.device, u0: np.ndarray, p_veh0: np.ndarray, p_ped0_multi: np.ndarray, dt: float, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Compute g0 and grad for M pedestrians.
    g_{m,t}(u) = ||p_veh_t(u) - p_ped_t^{(m)}(u)||
    Returns:
      g0_flat: [M*T]
      J: [M*T, T]
    """
    T = u0.shape[0]
    M = p_ped0_multi.shape[0]
    p_ped = nn_predict_positions_multi(model, device, u0, p_veh0, p_ped0_multi)  # [M,T,2]
    p_veh = vehicle_trajectory(p_veh0, u0, dt)  # [T,2]
    diff = p_veh.reshape(1, T, 2) - p_ped  # [M,T,2]
    g0 = np.linalg.norm(diff, axis=2)  # [M,T]

    J = np.zeros((M * T, T), dtype=np.float64)
    for k in range(T):
        u_pert = u0.copy()
        u_pert[k] += eps
        p_ped_p = nn_predict_positions_multi(model, device, u_pert, p_veh0, p_ped0_multi)  # [M,T,2]
        p_veh_p = vehicle_trajectory(p_veh0, u_pert, dt)  # [T,2]
        diff_p = p_veh_p.reshape(1, T, 2) - p_ped_p  # [M,T,2]
        g_p = np.linalg.norm(diff_p, axis=2)  # [M,T]
        grad_k = (g_p - g0) / eps  # [M,T]
        J[:, k] = grad_k.reshape(M * T)
    return g0.reshape(M * T).astype(np.float64), J


def solve_scp_subproblem(
    u_ref: np.ndarray, 
    u0: np.ndarray, 
    g0: np.ndarray, 
    J: np.ndarray, 
    C_eta: np.ndarray, 
    u_min: float, 
    u_max: float, 
    rho_ref: float, 
    d_safe: float,
    rho_trust: float = 1.0, 
    trust_region_radius: float = None,
) -> np.ndarray:
    """Solve convex subproblem:
        minimize   ||u - u_ref||_2^2
        subject to g0 + J (u - u0) >= C_eta
                   ||u - u0||_inf <= trust_region_radius  (if provided)
                   u_min <= u <= u_max
    Returns optimal u (numpy [T]).
    
    Args:
        trust_region_radius: If provided, limits ||u - u0||_inf <= trust_region_radius
                            (i.e., -R <= u_i - u0_i <= R for all i)
    """
    T = u0.shape[0]
    u = cp.Variable(T)
    # diff matrix
    D = np.eye(T) - np.roll(np.eye(T), 1, axis=0)
    diff_u = D @ u

    # objective function
    objective = cp.sum_squares(u - u_ref) + rho_ref * cp.sum_squares(diff_u[1:]) #+ 0.1*cp.sum_squares(u - 5)
    # objective = cp.sum_squares(u - u_ref)
    constraints = []
    # Linearized constraints for all (m,t) flattened
    MT = g0.shape[0]
    for r in range(MT):
        # Map row r to its time index t to select C_eta[t]
        t = r % T
        # Linearized: (||p_veh - p_ped|| - d_safe) >= C_eta[t]
        constraints.append(g0[r] - d_safe + J[r, :] @ (u - u0) >= C_eta[t])
    
    # Trust region constraint (infinity norm, linear constraints)
    # ||u - u0||_inf <= trust_region_radius  =>  -R <= u - u0 <= R
    if trust_region_radius is not None and trust_region_radius > 0:
        constraints.append(u - u0 <= trust_region_radius)
        constraints.append(u - u0 >= -trust_region_radius)
    
    constraints += [u >= u_min, u <= u_max]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    # prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, max_iter=20000, warm_start=True, verbose=False)
    prob.solve(solver=cp.GUROBI)
    if u.value is None:
        raise RuntimeError("SCP subproblem infeasible or solver failed")
    return np.asarray(u.value, dtype=np.float64)


def verify_constraints_multi(model: CausalPedestrianPredictor, device: torch.device, u: np.ndarray, p_veh0: np.ndarray, p_ped0_multi: np.ndarray, C_eta: np.ndarray, dt: float, d_safe: float = 1.0) -> Tuple[bool, int]:
    """
    Verify constraints and return (ok, violated_t).
    violated_t = -1 if all satisfied, else first time step index where constraint violated.
    """
    p_ped = nn_predict_positions_multi(model, device, u, p_veh0, p_ped0_multi)  # [M,T,2]
    p_veh = vehicle_trajectory(p_veh0, u, dt)  # [T,2]
    diff = p_veh.reshape(1, -1, 2) - p_ped
    norms = np.linalg.norm(diff, axis=2)  # [M,T]
    # Check (||p_veh - p_ped|| - d_safe) >= C_eta[t]
    violations = (norms - d_safe) < (C_eta.reshape(1, -1) - 1e-8)  # [M,T]
    if np.any(violations):
        # Find first violated time step (across all pedestrians)
        violated_indices = np.argwhere(violations)
        if len(violated_indices) > 0:
            # violated_indices is [num_violations, 2] with columns [m, t]
            first_t = int(violated_indices[:, 1].min())
            return False, first_t
    return True, -1


def scp_optimize(
    model_path: str,
    eta_csv_path: str,
    p_veh_0: np.ndarray,
    p_ped_0_multi: np.ndarray,
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
    model = CausalPedestrianPredictor(u_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load eta table
    eta_table, edges = load_eta_csv(eta_csv_path)  # [T,B] and edges
    assert eta_table.shape[0] >= T, "eta table T mismatch"

    dt = float(C.dt)
    last_u = np.full((T,), float(u_init), dtype=np.float64)
    ref_u = np.full((T,), float(u_ref), dtype=np.float64)

    iters_used = 0
    inner_scp_steps_list: List[int] = []
    
    # Statistics matrices: [last_bin, opt_bin]
    num_bins = eta_table.shape[1]
    reject_matrix = np.zeros((num_bins, num_bins), dtype=np.int64)  # rejected transitions
    transition_matrix = np.zeros((num_bins, num_bins), dtype=np.int64)  # all transitions (accepted + rejected)
    
    for it in range(outer_iters):
        # Freeze eta based on last_u
        C_eta = eta_of_u(eta_table, last_u.astype(np.float32), edges)  # [T]
        
        # Inner SCP loop (fixed C_eta): iterate linearize+QP until convergence or max steps
        # Trust region radius decays exponentially with each inner iteration
        u_curr = last_u.copy()
        max_inner_steps = 15
        tol = 1e-3
        inner_steps = 0
        for _inner in range(max_inner_steps):
            # Compute trust region radius for this inner iteration
            trust_radius = trust_region_initial * (trust_region_decay ** _inner)  *  last_u[-1] *10 * (0.1 ** _inner)
            
            g0, J = finite_diff_grad_multi(
                model,
                device,
                u_curr.astype(np.float32),
                p_veh_0.astype(np.float32),
                p_ped_0_multi.astype(np.float32),
                dt,
            )
            try:
                u_new = solve_scp_subproblem(
                    ref_u, u_curr, g0, J, C_eta, u_min, u_max, 
                    rho_ref=rho_ref, d_safe=d_safe, rho_trust=rho_trust,
                    trust_region_radius=trust_radius
                )
            except Exception as e:
                msg = f"SCP subproblem failed at outer iter {it}, inner iter {_inner}: {e}"
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(msg + '\n')
                # print(msg)
                return last_u, iters_used, inner_scp_steps_list, reject_matrix, transition_matrix
            # Log objective component magnitudes
            # if log_file:
            #     try:
            #         ref_term = float(np.sum((u_new - ref_u) ** 2))
            #         diff = u_new[1:] - u_new[:-1]
            #         smooth_term = float(np.sum(diff ** 2))
            #         with open(log_file, 'a') as f:
            #             f.write(
            #                 f"OBJ (outer={it}, inner={_inner}): ref={ref_term:.3e}, "
            #                 f"smooth={smooth_term:.3e}, rho_ref={rho_ref:g}, "
            #                 f"rho*smooth={(rho_ref*smooth_term):.3e}\n"
            #             )
            #     except Exception:
            #         pass
            inner_steps += 1
            if np.linalg.norm(u_new - u_curr, ord=2) <= tol:
                u_curr = u_new
                break
            u_curr = u_new
        opt_u = u_curr

        # Verify with its own eta
        C_eta_new = eta_of_u(eta_table, opt_u.astype(np.float32), edges)
        ok, violated_t = verify_constraints_multi(model, device, opt_u.astype(np.float32), p_veh_0.astype(np.float32), p_ped_0_multi.astype(np.float32), C_eta_new, dt, d_safe=d_safe)
        iters_used += 1
        # store average OSQP iters across inner steps for this outer iteration
        inner_scp_steps_list.append(inner_steps)
        
        # Record transition statistics using the violated time step (or mean if none)
        if violated_t >= 0 and violated_t < T:
            last_bin = bin_index_for_speed(float(last_u[violated_t]), edges)
            opt_bin = bin_index_for_speed(float(opt_u[violated_t]), edges)
        else:
            # Use mean speed for binning if violated_t invalid
            last_bin = bin_index_for_speed(float(np.mean(last_u)), edges)
            opt_bin = bin_index_for_speed(float(np.mean(opt_u)), edges)
        
        transition_matrix[last_bin, opt_bin] += 1  # Count all transitions
        
        if ok:
            last_u = opt_u
        else:
            reject_matrix[last_bin, opt_bin] += 1  # Count rejections
            msg = f"reject at outerloop iteration {it}, violated_t={violated_t}: last_u[{violated_t}]={last_u[violated_t]:.2f} (bin{last_bin}), opt_u[{violated_t}]={opt_u[violated_t]:.2f} (bin{opt_bin})"
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(msg + '\n')
            # print(msg)
            return last_u, iters_used, inner_scp_steps_list, reject_matrix, transition_matrix

    # Save reject statistics if path provided
    if reject_stats_path is not None:
        df_reject = pd.DataFrame(reject_matrix, 
                                  index=[f"last_bin{i}" for i in range(num_bins)],
                                  columns=[f"opt_bin{i}" for i in range(num_bins)])
        df_reject.to_csv(reject_stats_path)
        # print(f"Reject statistics saved to {reject_stats_path}")

    return last_u, iters_used, inner_scp_steps_list, reject_matrix, transition_matrix


