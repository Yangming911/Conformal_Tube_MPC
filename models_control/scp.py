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
from typing import Tuple, List, Optional

# Ensure project root on path BEFORE importing project modules
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import cvxpy as cp
import time

import utils.constants as C
from models_control.model_def import CausalPedestrianPredictor


class SCPSubproblemSolver:
    """更紧凑的SCP求解器，使用完全矩阵形式"""
    
    def __init__(self, T: int, M: int, u_min: float, u_max: float, rho_ref: float, d_safe: float):
        self.T = T
        self.M = M
        self.u_min = u_min
        self.u_max = u_max
        self.rho_ref = rho_ref
        self.d_safe = d_safe
        
        # 定义所有参数
        self.u_ref_param = cp.Parameter(T, nonneg=True)
        self.u0_param = cp.Parameter(T, nonneg=True)
        self.g0_param = cp.Parameter((M, T))
        self.J_param = cp.Parameter((M, T))
        self.C_eta_param = cp.Parameter(T, nonneg=True)
        self.trust_region_radius_param = cp.Parameter(nonneg=True)
        
        # 构建优化变量
        self.u = cp.Variable(T, nonneg=True)
        
        # 目标函数
        objective = cp.sum_squares(self.u - self.u_ref_param)
        
        # 约束条件
        self.constraints = []
        
        # 控制输入边界约束
        self.constraints += [self.u >= u_min, self.u <= u_max]
        
        # 信任域约束
        self.constraints += [
            self.u - self.u0_param <= self.trust_region_radius_param,
            self.u - self.u0_param >= -self.trust_region_radius_param
        ]
        
        # 线性化安全约束 - 完全矩阵形式
        # 将g0和J展平，创建一个大的向量约束
        g0_flat = cp.vec(self.g0_param)  # 形状: (M*T,)
        J_flat = cp.vec(self.J_param)    # 形状: (M*T,)
        
        # 创建重复的(u - u0)向量，形状: (M*T,)
        u_diff_repeated = cp.vstack([self.u - self.u0_param for _ in range(M)])
        u_diff_flat = cp.vec(u_diff_repeated)
        
        # 创建重复的(C_eta + d_safe)^2向量，形状: (M*T,)
        C_eta_safe_sq = cp.square(self.C_eta_param + d_safe)
        C_eta_safe_repeated = cp.vstack([C_eta_safe_sq for _ in range(M)])
        C_eta_safe_flat = cp.vec(C_eta_safe_repeated)
        
        # 单个向量约束，包含所有安全约束
        safety_constraint = g0_flat + cp.multiply(J_flat, u_diff_flat) >= C_eta_safe_flat
        self.constraints.append(safety_constraint)
        
        # 构建问题
        self.problem = cp.Problem(cp.Minimize(objective), self.constraints)
        
        # 初始编译
        self._compile_with_dummy_data()
    
    def _compile_with_dummy_data(self):
        """使用虚拟数据编译问题"""
        dummy_data = np.ones(self.T)
        dummy_matrix = np.ones((self.M, self.T))
        
        self.u_ref_param.value = dummy_data
        self.u0_param.value = dummy_data
        self.g0_param.value = dummy_matrix
        self.J_param.value = dummy_matrix
        self.C_eta_param.value = dummy_data
        self.trust_region_radius_param.value = 1e6
        
        try:
            self.problem.solve(solver=cp.GUROBI)
            print("紧凑SCP求解器编译成功")
        except Exception as e:
            print(f"初始编译失败: {e}")
    
    def solve(self, u_ref, u0, g0, J, C_eta, trust_region_radius=None):
        """求解方法"""
        # 更新参数值
        self.u_ref_param.value = u_ref
        self.u0_param.value = u0
        self.g0_param.value = g0
        self.J_param.value = J
        self.C_eta_param.value = C_eta
        
        if trust_region_radius is not None and trust_region_radius > 0:
            self.trust_region_radius_param.value = trust_region_radius
        else:
            self.trust_region_radius_param.value = 1e6
        
        self.problem.solve(solver=cp.GUROBI, warm_start=True)
        
        if self.u.value is None:
            print(f"紧凑SCP求解器求解失败: {self.problem.status}")
            raise RuntimeError("SCP子问题不可行或求解器失败")
        
        return np.asarray(self.u.value, dtype=np.float64)

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


def finite_diff_grad_multi(model: CausalPedestrianPredictor, device: torch.device, u0: np.ndarray, p_veh0: np.ndarray, p_ped0_multi: np.ndarray, dt: float,scaler=None, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dg/du and g(u0) for M pedestrians.
    """
    T = u0.shape[0]
    M = p_ped0_multi.shape[0]
    p_veh = vehicle_trajectory(p_veh0, u0, dt) # [T,2]
    p_ped = nn_predict_positions_multi(model, device, u0, p_veh0, p_ped0_multi, scaler)  # [M,T,2]
    # g = (p_veh - p_ped)^2
    g = np.linalg.norm(p_veh[None,:,:] - p_ped, axis=2)**2  # [M,T]
    L_x = np.tril(np.ones((T, T), dtype=np.float32))
    L_y = np.zeros((T, T), dtype=np.float32)
    L = np.stack([L_x, L_y], axis=2)  # [T,T,2]
    # delta_omiga = d(p_ped)/d(u) with shape [M,T,T]
    delta_omiga = np.zeros((M, T, T, 2), dtype=np.float32)
    for t in range(T):
        eps_u = u0
        eps_ = eps+np.random.normal(0, 0.1*eps)
        eps_u[t] += eps_
        p_ped_eps = nn_predict_positions_multi(model, device, eps_u, p_veh0, p_ped0_multi, scaler)  # [M,T,2]
        diff = p_ped_eps - p_ped  # [M,T,2]

        delta_omiga[:, t, :, :] = diff / eps_  # [M,T,2]

    # dg/du=2(p_veh - p_ped)(L- delta_omiga)
    J = np.zeros((M, T, 2), dtype=np.float32)
    for m in range(M):
        # x axis
        J[m, :, 0] = 2 * (p_veh[:,0] - p_ped[m, :, 0]).reshape(-1,1).transpose() @ (L_x - delta_omiga[m, :, :, 0])  # [T]
        # y axis
        J[m, :, 1] = 2 * (p_veh[:,1] - p_ped[m, :, 1]).reshape(-1,1).transpose() @ (L_y - delta_omiga[m, :, :, 1])  # [T]

    #
    J = np.sum(J, axis=2)  # [M,T]
    return g, J


def verify_constraints_multi(model: CausalPedestrianPredictor, device: torch.device, u: np.ndarray, p_veh0: np.ndarray, p_ped0_multi: np.ndarray, C_eta: np.ndarray, dt: float,scaler=None, d_safe: float = 1.0) -> Tuple[bool, int]:
    """
    Verify constraints and return (ok, violated_t).
    violated_t = -1 if all satisfied, else first time step index where constraint violated.
    """
    p_ped = nn_predict_positions_multi(model, device, u, p_veh0, p_ped0_multi,scaler)  # [M,T,2]
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
    solver: SCPSubproblemSolver,
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
    model: CausalPedestrianPredictor=None,
    device: torch.device=None,
    scaler: dict=None,
) -> Tuple[np.ndarray, int, List[int], np.ndarray, np.ndarray]:
    if model is None:
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
        scaler = None
        if 'u_scaler' in checkpoint.keys():
            scaler = {
                'u_scaler': checkpoint['u_scaler'],
                'p_veh0_scaler': checkpoint['p_veh0_scaler'],
                'p_ped0_scaler': checkpoint['p_ped0_scaler'],
                'p_seq_scaler': checkpoint['p_seq_scaler'],
            }

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
            trust_radius = trust_region_initial * (trust_region_decay ** _inner) *  last_u[-1] *10 * (0.1 ** _inner)
            
            g0, J = finite_diff_grad_multi(
                model,
                device,
                u_curr.astype(np.float32),
                p_veh_0.astype(np.float32),
                p_ped_0_multi.astype(np.float32),
                dt,
                scaler
            )

            try:
                u_new = solver.solve(ref_u,u_curr,g0,J,C_eta,trust_radius)
            except Exception as e:
                msg = f"SCP subproblem failed at outer iter {it}, inner iter {_inner}: {e}, p_veh={p_veh_0}"
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(msg + '\n')
                # print(msg)
                return last_u, iters_used, inner_scp_steps_list, reject_matrix, transition_matrix, msg
            inner_steps += 1
            if np.linalg.norm(u_new - u_curr, ord=2) <= tol:
                u_curr = u_new
                break
            u_curr = u_new
        opt_u = u_curr

        # Verify with its own eta
        C_eta_new = eta_of_u(eta_table, opt_u.astype(np.float32), edges)
        ok, violated_t = verify_constraints_multi(model, device, opt_u.astype(np.float32), p_veh_0.astype(np.float32), p_ped_0_multi.astype(np.float32), C_eta_new, dt,scaler, d_safe=d_safe)
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
            return last_u, iters_used, inner_scp_steps_list, reject_matrix, transition_matrix, msg

    # Save reject statistics if path provided
    if reject_stats_path is not None:
        df_reject = pd.DataFrame(reject_matrix, 
                                  index=[f"last_bin{i}" for i in range(num_bins)],
                                  columns=[f"opt_bin{i}" for i in range(num_bins)])
        df_reject.to_csv(reject_stats_path)
        # print(f"Reject statistics saved to {reject_stats_path}")

    return last_u, iters_used, inner_scp_steps_list, reject_matrix, transition_matrix, "success"