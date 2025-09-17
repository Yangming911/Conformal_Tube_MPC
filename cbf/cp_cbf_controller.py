import sys
from pathlib import Path
import numpy as np
import cvxpy as cp

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mpc.car_dynamics import forward_car
from mpc.ped_dynamics import forward_ped_trace
from models.conformal_grid import get_eta
import utils.constants as C

def cp_cbf_controller(state, T=10, d_safe=10.0, cp_alpha=0.85):
    """
    考虑 conformal 预测误差的轨迹CBF控制器，输出当前控制输入 u0。
    
    Args:
        state: Dict containing car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy
        T: 预测步长
        d_safe: 安全距离
    
    Returns:
        float: Control input (car velocity)
    """
    car_x = state["car_x"]
    car_y = state["car_y"]
    car_v = state["car_v"]
    walker_x = state["walker_x"]
    walker_y = state["walker_y"]
    walker_vx = state["walker_vx"]
    walker_vy = state["walker_vy"]
    
    # 初始位置
    x0 = car_x
    y0 = [walker_x, walker_y]
    
    u_des = 15.0
    u_min, u_max = 0.0, 15.0

    # Step 1: 尝试匀速是否安全
    u_nominal = np.full(T, u_des)
    x_nominal = forward_car(x0, u_nominal)
    y_nominal = forward_ped_trace(y0, x_nominal, car_x0=x0, car_y=car_y)

    is_safe = True
    for t in range(T):
        car_pos = np.array([x_nominal[t], car_y])
        ped_pos = np.array(y_nominal[t])

        # 提取信息用于获取 eta (采用可用量的近似)
        car_speed = u_nominal[t]
        car_x_curr = x_nominal[t]
        walker_x_curr, walker_y_curr = ped_pos[0], ped_pos[1]
        # 用至今位移均速近似行人当前速度
        approx_vx = (ped_pos[0] - y0[0]) / (t + 1e-3)
        approx_vy = (ped_pos[1] - y0[1]) / (t + 1e-3)
        eta_x, eta_y = get_eta(car_x_curr, car_speed, walker_x_curr, walker_y_curr, approx_vx, approx_vy, cp_alpha)
        d_eff = np.sqrt((d_safe + eta_x)**2 + eta_y**2)

        dist = np.linalg.norm(car_pos - ped_pos)
        if dist < d_eff:
            is_safe = False
            break

    if is_safe:
        return u_des

    # Step 2: 构造优化问题（目标仍是 u0 接近 u_des）
    u_seq = cp.Variable(T)
    x_trace = forward_car(x0, u_seq)
    x_sample = forward_car(x0, np.full(T, u_des))
    y_trace = forward_ped_trace(y0, x_sample, car_x0=x0, car_y=car_y)

    constraints = [u_seq >= u_min, u_seq <= u_max]
    for t in range(T):
        dx = x_trace[t] - y_trace[t][0]
        dy = car_y - y_trace[t][1]

        # 同样获取 eta 并构造 d_eff
        car_speed = u_des  # 用 sample 轨迹的速度近似
        car_x_curr = x_sample[t]
        walker_x_curr, walker_y_curr = y_trace[t][0], y_trace[t][1]
        approx_vx = (y_trace[t][0] - y0[0]) / (t + 1e-3)
        approx_vy = (y_trace[t][1] - y0[1]) / (t + 1e-3)
        eta_x, eta_y = get_eta(car_x_curr, car_speed, walker_x_curr, walker_y_curr, approx_vx, approx_vy, cp_alpha)
        d_eff = np.sqrt((d_safe + eta_x)**2 + eta_y**2)

        dist_expr = cp.norm(cp.hstack([dx, dy]))
        constraints.append(dist_expr >= d_eff)

    prob = cp.Problem(cp.Minimize((u_seq[0] - u_des)**2), constraints)
    try:
        prob.solve(solver=cp.OSQP)
        if u_seq.value is None:
            return u_min
        return float(u_seq.value[0])
    except:
        return u_min


def cp_cbf_controller_multi_pedestrian(state, T=10, d_safe=10.0, cp_alpha=0.85):
    """
    考虑 conformal 预测误差的多行人轨迹CBF控制器，输出当前控制输入 u0。
    
    Args:
        state: Dict containing car_x, car_y, car_v, walker_x (list), walker_y (list), walker_vx (list), walker_vy (list)
        T: 预测步长
        d_safe: 安全距离
    
    Returns:
        float: Control input (car velocity)
    """
    car_x = state["car_x"]
    car_y = state["car_y"]
    car_v = state["car_v"]
    walker_x_list = state["walker_x"]
    walker_y_list = state["walker_y"]
    walker_vx_list = state["walker_vx"]
    walker_vy_list = state["walker_vy"]
    
    num_pedestrians = len(walker_x_list)
    
    # 初始位置
    x0 = car_x
    y0_list = [[walker_x_list[i], walker_y_list[i]] for i in range(num_pedestrians)]
    
    u_des = 15.0
    u_min, u_max = 0.0, 15.0

    # Step 1: 尝试匀速是否安全
    u_nominal = np.full(T, u_des)
    x_nominal = forward_car(x0, u_nominal)
    
    # 为每个行人预测轨迹
    y_nominal_list = []
    for i in range(num_pedestrians):
        y_nominal = forward_ped_trace(y0_list[i], x_nominal, car_x0=x0, car_y=car_y)
        y_nominal_list.append(y_nominal)

    is_safe = True
    for t in range(T):
        car_pos = np.array([x_nominal[t], car_y])
        
        # 检查与每个行人的距离
        for i in range(num_pedestrians):
            ped_pos = np.array(y_nominal_list[i][t])

            # 提取信息用于获取 eta
            car_speed = u_nominal[t]
            car_x_curr = x_nominal[t]
            walker_x_curr, walker_y_curr = ped_pos[0], ped_pos[1]
            # 用至今位移均速近似行人当前速度
            approx_vx = (ped_pos[0] - y0_list[i][0]) / (t + 1e-3)
            approx_vy = (ped_pos[1] - y0_list[i][1]) / (t + 1e-3)
            eta_x, eta_y = get_eta(car_x_curr, car_speed, walker_x_curr, walker_y_curr, approx_vx, approx_vy, cp_alpha)
            d_eff = np.sqrt((d_safe + eta_x)**2 + eta_y**2)

            dist = np.linalg.norm(car_pos - ped_pos)
            if dist < d_eff:
                is_safe = False
                break
        
        if not is_safe:
            break

    if is_safe:
        return u_des

    # Step 2: 构造优化问题（目标仍是 u0 接近 u_des）
    u_seq = cp.Variable(T)
    x_trace = forward_car(x0, u_seq)
    x_sample = forward_car(x0, np.full(T, u_des))
    
    # 为每个行人预测轨迹
    y_trace_list = []
    for i in range(num_pedestrians):
        y_trace = forward_ped_trace(y0_list[i], x_sample, car_x0=x0, car_y=car_y)
        y_trace_list.append(y_trace)

    constraints = [u_seq >= u_min, u_seq <= u_max]
    
    # 为每个行人和每个时间步添加约束
    for t in range(T):
        for i in range(num_pedestrians):
            dx = x_trace[t] - y_trace_list[i][t][0]
            dy = car_y - y_trace_list[i][t][1]

            # 同样获取 eta 并构造 d_eff
            car_speed = u_des  # 用 sample 轨迹的速度近似
            car_x_curr = x_sample[t]
            walker_x_curr, walker_y_curr = y_trace_list[i][t][0], y_trace_list[i][t][1]
            approx_vx = (y_trace_list[i][t][0] - y0_list[i][0]) / (t + 1e-3)
            approx_vy = (y_trace_list[i][t][1] - y0_list[i][1]) / (t + 1e-3)
            eta_x, eta_y = get_eta(car_x_curr, car_speed, walker_x_curr, walker_y_curr, approx_vx, approx_vy, cp_alpha)
            d_eff = np.sqrt((d_safe + eta_x)**2 + eta_y**2)

            dist_expr = cp.norm(cp.hstack([dx, dy]))
            constraints.append(dist_expr >= d_eff)

    prob = cp.Problem(cp.Minimize((u_seq[0] - u_des)**2), constraints)
    try:
        prob.solve(solver=cp.OSQP)
        if u_seq.value is None:
            return u_min
        return float(u_seq.value[0])
    except:
        return u_min

# SLSQP
# import sys
# from pathlib import Path
# import numpy as np
# from scipy.optimize import minimize

# # Ensure project root is on sys.path
# PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from mpc.car_dynamics import forward_car
# from mpc.ped_dynamics import forward_ped_trace
# from models.conformal_grid import get_eta
# import utils.constants as C

# def cp_cbf_controller(state, T=10, d_safe=0.5):
#     """
#     使用scipy.optimize的考虑conformal预测误差的轨迹CBF控制器。
#     真正使用训练的网络和CP进行不确定性量化。
    
#     Args:
#         state: Dict containing car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy
#         T: 预测步长
#         d_safe: 安全距离
    
#     Returns:
#         float: Control input (car velocity)
#     """
#     car_x = state["car_x"]
#     car_y = state["car_y"]
#     car_v = state["car_v"]
#     walker_x = state["walker_x"]
#     walker_y = state["walker_y"]
#     walker_vx = state["walker_vx"]
#     walker_vy = state["walker_vy"]
    
#     # 初始位置
#     x0 = car_x
#     y0 = [walker_x, walker_y]
    
#     u_des = 15.0
#     u_min, u_max = 0.0, 15.0

#     # Step 1: 尝试匀速是否安全（使用真实的网络预测）
#     u_nominal = np.full(T, u_des)
#     x_nominal = forward_car(x0, u_nominal)
#     y_nominal = forward_ped_trace(y0, x_nominal, car_x0=x0, car_y=car_y)

#     is_safe = True
#     for t in range(T):
#         car_pos = np.array([x_nominal[t], car_y])
#         ped_pos = np.array(y_nominal[t])

#         # 使用真实的网络预测和CP
#         car_speed = u_nominal[t]
#         car_x_curr = x_nominal[t]
#         walker_x_curr, walker_y_curr = ped_pos[0], ped_pos[1]
#         approx_vx = (ped_pos[0] - y0[0]) / (t + 1e-3)
#         approx_vy = (ped_pos[1] - y0[1]) / (t + 1e-3)
#         eta_x, eta_y = get_eta(car_x_curr, car_speed, walker_x_curr, walker_y_curr, approx_vx, approx_vy, cp_alpha)
#         d_eff = np.sqrt((d_safe + eta_x)**2 + eta_y**2)

#         dist = np.linalg.norm(car_pos - ped_pos)
#         if dist < d_eff:
#             is_safe = False
#             break

#     if is_safe:
#         return u_des

#     # Step 2: 使用scipy.optimize进行优化
#     def objective(u_seq):
#         """目标函数：使第一个控制输入接近期望值"""
#         return (u_seq[0] - u_des)**2
    
#     def constraint_distance(u_seq, t):
#         """距离约束：确保在时刻t保持安全距离"""
#         # 使用真实的网络预测车辆轨迹
#         x_trace = forward_car(x0, u_seq)
        
#         # 使用真实的网络预测行人轨迹
#         y_trace = forward_ped_trace(y0, x_trace, car_x0=x0, car_y=car_y)
        
#         # 计算距离
#         car_pos = np.array([x_trace[t], car_y])
#         ped_pos = np.array(y_trace[t])
#         dist = np.linalg.norm(car_pos - ped_pos)
        
#         # 使用真实的CP计算eta
#         car_speed = u_seq[t]
#         car_x_curr = x_trace[t]
#         walker_x_curr, walker_y_curr = ped_pos[0], ped_pos[1]
#         approx_vx = (ped_pos[0] - y0[0]) / (t + 1e-3)
#         approx_vy = (ped_pos[1] - y0[1]) / (t + 1e-3)
#         eta_x, eta_y = get_eta(car_x_curr, car_speed, walker_x_curr, walker_y_curr, approx_vx, approx_vy, cp_alpha)
#         d_eff = np.sqrt((d_safe + eta_x)**2 + eta_y**2)
        
#         return dist - d_eff  # 返回距离 - 有效安全距离，需要 >= 0
    
#     # 构建约束
#     constraints = []
#     for t in range(T):
#         constraints.append({
#             'type': 'ineq',
#             'fun': lambda u_seq, t=t: constraint_distance(u_seq, t)
#         })
    
#     # 边界约束
#     bounds = [(u_min, u_max)] * T
    
#     # 初始猜测
#     x0_guess = u_des * np.ones(T)
    
#     try:
#         # 使用SLSQP求解器
#         result = minimize(objective, x0_guess, method='SLSQP', 
#                          constraints=constraints, bounds=bounds,
#                          options={'maxiter': 100, 'ftol': 1e-6})
        
#         if result.success:
#             return float(result.x[0])
#         else:
#             # 如果优化失败，尝试更保守的方法
#             return max(u_min, u_des * 0.8)
            
#     except Exception as e:
#         print(f"Optimization failed: {e}")
#         return u_min