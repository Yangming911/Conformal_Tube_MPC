import numpy as np
import cvxpy as cp
import sys
import time
from pathlib import Path
from scipy.optimize import minimize

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.constants as C
from envs.simulator import _is_collision
from models.predictor import WalkerActionPredictor
from models.conformal_grid import get_eta
import torch

# 0.95
# DEFAULE_ETA_X = 0.1980
# DEFAULE_ETA_Y = 0.9718

# 0.75
# DEFAULE_ETA_X = 0.1158
# DEFAULE_ETA_Y = 0.5763

# 0.85
DEFAULE_ETA_X = 0.1446
DEFAULE_ETA_Y = 0.7179

# 0.9
# DEFAULE_ETA_X = 0.165629
# DEFAULE_ETA_Y = 0.822073

# 0.75
# DEFAULE_ETA_X = 0.115841
# DEFAULE_ETA_Y = 0.576356

# 0.5
# DEFAULE_ETA_X = 0.067946
# DEFAULE_ETA_Y = 0.339933

def cbf_controller(state, T=10, N=100, d_safe=2.5, model_path='assets/best_model.pth', device=None, use_eta=True, use_slsqp=True, cp_alpha=0.85, gamma=1.0):
    """
    Quadratic Program (QP)-based Control Barrier Function controller with network prediction.
    
    Args:
        state: Dict containing car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy
        T: Prediction horizon (unused in this simple version)
        N: Number of samples (unused in this simple version)
        d_safe: Safety distance
        model_path: Path to the trained walker prediction model
        device: Device to use for model inference (default: 'cuda' if available, else 'cpu')
        use_eta: Whether to use conformal calibration (eta) for uncertainty adjustment
        use_slsqp: Whether to use SLSQP solver instead of OSQP
    
    Returns:
        float: Control input (car velocity)
    """
    # Set default device to GPU if available
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    car_x = state["car_x"]
    car_y = state["car_y"]
    car_v = state["car_v"]
    walker_x = state["walker_x"]
    walker_y = state["walker_y"]
    walker_vx = state["walker_vx"]
    walker_vy = state["walker_vy"]

    # Parameters
    # gamma is now passed as function parameter
    u_max = 15.0
    u_min = 0.0
    u_des = u_max  # Want to go as fast as possible

    # If collision detected, return conservative control
    if _is_collision(state):
        return u_min

    # Use network prediction to estimate next state velocity of walker
    try:
        predictor = WalkerActionPredictor(model_path=model_path, device=device)
        next_walker_vx, next_walker_vy = predictor.predict(
            car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy
        )
        
        # Predict next state positions using direct integration
        dt = float(C.dt)
        next_car_x = car_x + car_v * dt
        next_car_y = car_y  # Car y position doesn't change (stays in lane)
        next_walker_x = walker_x + next_walker_vx * dt
        next_walker_y = walker_y + next_walker_vy * dt
        
        # Compute distance and gradient for next state
        # Distance function: h = ||[car_y - ped_y, car_x - ped_x]|| - d_safe
        dx_next = next_car_x - next_walker_x  # car_x - ped_x
        dy_next = next_car_y - next_walker_y  # car_y - ped_y
        dist_next = np.sqrt(dx_next**2 + dy_next**2)
        
        # Adjust safety distance based on use_eta flag
        if use_eta:
            # Use conformal calibration to get safe region (eta values)
            # eta_x, eta_y = get_eta(car_x, car_v, walker_x, walker_y, walker_vx, walker_vy, cp_alpha)
            eta_x, eta_y = DEFAULE_ETA_X, DEFAULE_ETA_Y
            d_safe_adjusted = d_safe + np.sqrt(eta_x**2 + eta_y**2)
        else:
            # Use original safety distance without uncertainty adjustment
            d_safe_adjusted = d_safe
        
        # Check if predicted next state would result in collision
        next_state = {
            "car_x": next_car_x,
            "car_y": next_car_y,
            "car_v": car_v,
            "walker_x": next_walker_x,
            "walker_y": next_walker_y,
            "walker_vx": next_walker_vx,
            "walker_vy": next_walker_vy,
        }
        if _is_collision(next_state):
            return u_min
            
    except Exception as e:
        # Fallback to current state if prediction fails
        print(f"Prediction failed: {e}, using current state")
        # Distance function: h = ||[car_y - ped_y, car_x - ped_x]|| - d_safe
        dx_next = car_x - walker_x  # car_x - ped_x
        dy_next = car_y - walker_y  # car_y - ped_y
        dist_next = np.sqrt(dx_next**2 + dy_next**2)
        d_safe_adjusted = d_safe

    # Define control variable
    u = cp.Variable()

    # CBF constraint for next state
    # h = ||[car_y - ped_y, car_x - ped_x]|| - d_safe
    # ∂h/∂car_x = (car_x - ped_x) / ||[car_y - ped_y, car_x - ped_x]||
    # CBF constraint: ∂h/∂car_x · u + γ · h ≥ 0
    grad_h = dx_next / dist_next  # ∂h/∂car_x = (car_x - ped_x) / ||[car_y - ped_y, car_x - ped_x]||
    h_val = dist_next - d_safe_adjusted
    cbf_constraint = grad_h * u + gamma * h_val >= 0

    # Solve CBF optimization problem
    if use_slsqp:
        # Use SLSQP solver with time limit
        return solve_cbf_with_slsqp(grad_h, h_val, u_des, u_min, u_max, gamma, time_limit=0.1)
    else:
        # Use CVXPY with OSQP solver
        objective = cp.Minimize((u - u_des)**2)
        constraints = [cbf_constraint, u >= u_min, u <= u_max]

        prob = cp.Problem(objective, constraints)
        try:
            # Use OSQP solver with strict time limit (0.1s max)
            start_time = time.time()
            prob.solve(solver=cp.OSQP, max_iter=1000, eps_abs=1e-6, eps_rel=1e-6)
            solve_time = time.time() - start_time
            
            # If solve time exceeds 0.1s, return safe control
            if solve_time > 0.1:
                print(f"Warning: CBF solve time {solve_time:.3f}s exceeded 0.1s limit, returning u_min")
                return u_min
                
            if u.value is None:
                return u_min
            return float(u.value)
        except Exception as e:
            print(f"CBF solver failed: {e}")
            return u_min  # fallback if solver fails


def cbf_controller_no_eta(state, T=10, N=100, d_safe=0.5, model_path='assets/best_model.pth', device=None, cp_alpha=0.85, gamma=1.0):

    return cbf_controller(state, T, N, d_safe, model_path, device, use_eta=False, cp_alpha=cp_alpha, gamma=gamma)


def cbf_controller_slsqp(state, T=10, N=100, d_safe=0.5, model_path='assets/best_model.pth', device=None, use_eta=True, cp_alpha=0.85, gamma=1.0):


    return cbf_controller(state, T, N, d_safe, model_path, device, use_eta=use_eta, use_slsqp=True, cp_alpha=cp_alpha, gamma=gamma)


def cbf_controller_multi_pedestrian_no_eta(state, T=10, N=100, d_safe=2.5, model_path='assets/best_model.pth', device=None, use_slsqp=True, cp_alpha=0.85, gamma=1.0):

    return cbf_controller_multi_pedestrian(state, T, N, d_safe, model_path, device, use_eta=False, use_slsqp=use_slsqp, cp_alpha=cp_alpha, gamma=gamma)


def cbf_controller_multi_pedestrian(state, T=10, N=100, d_safe=2.5, model_path='assets/best_model.pth', device=None, use_eta=True, use_slsqp=True, cp_alpha=0.85, gamma=1.0):

    # Set default device to GPU if available
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    car_x = state["car_x"]
    car_y = state["car_y"]
    car_v = state["car_v"]
    walker_x_list = state["walker_x"]
    walker_y_list = state["walker_y"]
    walker_vx_list = state["walker_vx"]
    walker_vy_list = state["walker_vy"]

    num_pedestrians = len(walker_x_list)

    # Parameters
    # gamma is now passed as function parameter
    u_max = 15.0
    u_min = 0.0
    u_des = u_max  # Want to go as fast as possible

    # Check if collision occurs with any pedestrian
    from envs.simulator import _is_collision_multi_pedestrian
    if _is_collision_multi_pedestrian(state):
        return u_min

    # Predict next state for each pedestrian
    all_constraints = []
    all_gradients = []
    all_h_values = []
    
    try:
        predictor = WalkerActionPredictor(model_path=model_path, device=device)
        
        for i in range(num_pedestrians):
            # Use neural network to predict next velocity for each pedestrian
            next_walker_vx, next_walker_vy = predictor.predict(
                car_x, car_y, car_v, walker_x_list[i], walker_y_list[i], walker_vx_list[i], walker_vy_list[i]
            )
            
            # Predict next position
            dt = float(C.dt)
            next_car_x = car_x + car_v * dt
            next_car_y = car_y  # Car y position doesn't change (stays in lane)
            next_walker_x = walker_x_list[i] + next_walker_vx * dt
            next_walker_y = walker_y_list[i] + next_walker_vy * dt
            
            # Calculate distance and gradient
            dx_next = next_car_x - next_walker_x
            dy_next = next_car_y - next_walker_y
            dist_next = np.sqrt(dx_next**2 + dy_next**2)
            
            # Adjust safety distance
            if use_eta:
                # eta_x, eta_y = get_eta(car_x, car_v, walker_x_list[i], walker_y_list[i], walker_vx_list[i], walker_vy_list[i], cp_alpha)
                eta_x, eta_y = DEFAULE_ETA_X, DEFAULE_ETA_Y
                d_safe_adjusted = d_safe + np.sqrt(eta_x**2 + eta_y**2)
            else:
                d_safe_adjusted = d_safe
            
            # Check if predicted state will cause collision
            next_state = {
                "car_x": next_car_x,
                "car_y": next_car_y,
                "car_v": car_v,
                "walker_x": [next_walker_x],
                "walker_y": [next_walker_y],
                "walker_vx": [next_walker_vx],
                "walker_vy": [next_walker_vy],
            }
            if _is_collision_multi_pedestrian(next_state):
                return u_min
            
            # Calculate CBF constraint
            grad_h = dx_next / dist_next
            h_val = dist_next - d_safe_adjusted
            
            all_gradients.append(grad_h)
            all_h_values.append(h_val)
            
    except Exception as e:
        print(f"Prediction failed: {e}, using current state")
        # If prediction fails, use current state
        # for i in range(num_pedestrians):
        #     dx_next = car_x - walker_x_list[i]
        #     dy_next = car_y - walker_y_list[i]
        #     dist_next = np.sqrt(dx_next**2 + dy_next**2)
            
        #     # 调整安全距离
        #     if use_eta:
        #         eta_x, eta_y = get_eta(car_x, car_v, walker_x_list[i], walker_y_list[i], walker_vx_list[i], walker_vy_list[i], cp_alpha)
        #         d_safe_adjusted = d_safe + np.sqrt(eta_x**2 + eta_y**2)
        #     else:
        #         d_safe_adjusted = d_safe
            
        #     grad_h = dx_next / dist_next
        #     h_val = dist_next - d_safe_adjusted
            
        #     all_gradients.append(grad_h)
        #     all_h_values.append(h_val)

    # Define control variable
    u = cp.Variable()

    # Add CBF constraints for each pedestrian
    constraints = [u >= u_min, u <= u_max]
    for i in range(num_pedestrians):
        cbf_constraint = all_gradients[i] * u + gamma * all_h_values[i] >= 0
        constraints.append(cbf_constraint)

    # Solve CBF optimization problem
    if use_slsqp:
        # Use SLSQP solver
        return solve_cbf_multi_pedestrian_with_slsqp(all_gradients, all_h_values, u_des, u_min, u_max, gamma, time_limit=0.1)
    else:
        # Use CVXPY with OSQP solver
        objective = cp.Minimize((u - u_des)**2)
        prob = cp.Problem(objective, constraints)

        try:
            start_time = time.time()
            prob.solve(solver=cp.OSQP, max_iter=1000, eps_abs=1e-6, eps_rel=1e-6)
            solve_time = time.time() - start_time
            
            if solve_time > 0.1:
                print(f"Warning: CBF solve time {solve_time:.3f}s exceeded 0.1s limit, returning u_min")
                return u_min
                
            if u.value is None:
                return u_min
            return float(u.value)
        except Exception as e:
            print(f"CBF solver failed: {e}")
            return u_min


def solve_cbf_multi_pedestrian_with_slsqp(gradients, h_values, u_des, u_min, u_max, gamma, time_limit=0.1):
    """
    Use SLSQP solver to solve multi-pedestrian CBF optimization problem with time limit
    
    Args:
        gradients: List of CBF constraint gradients for each pedestrian
        h_values: List of CBF function values for each pedestrian
        u_des: Desired control input
        u_min: Minimum control input
        u_max: Maximum control input
        gamma: CBF parameter
        time_limit: Time limit (seconds)
    
    Returns:
        float: Control input value, returns u_min if timeout
    """
    def objective(u_val):
        return (u_val - u_des)**2
    
    def constraint(u_val):
        # Return minimum value among all constraints to ensure all constraints are satisfied
        constraint_values = [grad * u_val + gamma * h_val for grad, h_val in zip(gradients, h_values)]
        return min(constraint_values)
    
    # Initial guess
    u0 = np.clip(u_des, u_min, u_max)
    
    # Constraints
    constraints = {'type': 'ineq', 'fun': constraint}
    bounds = [(u_min, u_max)]
    
    try:
        start_time = time.time()
        result = minimize(
            objective,
            u0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        solve_time = time.time() - start_time
        
        if solve_time > time_limit:
            # print(f"Warning: SLSQP solve time {solve_time:.3f}s exceeded {time_limit}s limit, returning u_min")
            return u_min
        
        if result.success:
            return float(result.x[0])
        else:
            # print(f"SLSQP solver failed: {result.message}")
            return u_min
            
    except Exception as e:
        # print(f"SLSQP solver error: {e}")
        return u_min


def solve_cbf_with_slsqp(grad_h, h_val, u_des, u_min, u_max, gamma, time_limit=0.1):
    """
    Use SLSQP solver to solve CBF optimization problem with time limit
    
    Args:
        grad_h: CBF constraint gradient
        h_val: CBF function value
        u_des: Desired control input
        u_min: Minimum control input
        u_max: Maximum control input
        gamma: CBF parameter
        time_limit: Time limit (seconds)
    
    Returns:
        float: Control input value, returns u_min if timeout
    """
    def objective(u_val):
        return (u_val - u_des)**2
    
    def constraint(u_val):
        return grad_h * u_val + gamma * h_val
    
    # Initial guess
    u0 = np.clip(u_des, u_min, u_max)
    
    # Constraints
    constraints = {'type': 'ineq', 'fun': constraint}
    bounds = [(u_min, u_max)]
    
    try:
        start_time = time.time()
        result = minimize(
            objective,
            u0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        solve_time = time.time() - start_time
        
        # If solve time exceeds limit, return safe control
        if solve_time > time_limit:
            # print(f"Warning: SLSQP solve time {solve_time:.3f}s exceeded {time_limit}s limit, returning u_min")
            return u_min
        
        if result.success:
            return float(result.x[0])
        else:
            # print(f"SLSQP solver failed: {result.message}")
            return u_min
            
    except Exception as e:
        # print(f"SLSQP solver error: {e}")
        return u_min
