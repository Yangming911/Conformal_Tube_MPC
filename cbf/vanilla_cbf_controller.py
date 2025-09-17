import cvxpy as cp
import sys
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mpc.car_dynamics import forward_car
from mpc.ped_dynamics import forward_ped_trace
import utils.constants as C

def vanilla_cbf_controller(state, T=10, d_safe=0.5):
    """
    使用cvxpy求解轨迹CBF问题，返回当前控制输入 u0。

    参数：
        state: Dict containing car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy
        T: 预测步长
        d_safe: 安全距离

    返回：
        u0: 当前步速度（float）
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

    # Step 1: 尝试匀速控制是否安全
    u_nominal = np.full(T, u_des)
    x_nominal = forward_car(x0, u_nominal)
    y_nominal = forward_ped_trace(y0, x_nominal, car_x0=x0, car_y=car_y)

    is_safe = True
    for t in range(T):
        car_pos = np.array([x_nominal[t], car_y])
        ped_pos = np.array(y_nominal[t])
        if np.linalg.norm(car_pos - ped_pos) < d_safe:
            is_safe = False
            break
    if is_safe:
        return u_des  # 匀速控制已足够安全，直接返回

    # Step 2: 构造cvxpy优化问题，优化整个 u_seq
    u_seq = cp.Variable(T)
    x_trace = forward_car(x0, u_seq)  # 符号 car 轨迹
    x_trace_sample = forward_car(x0, np.full(T, u_des))  # 用于推理行人
    y_trace = forward_ped_trace(y0, x_trace_sample, car_x0=x0, car_y=car_y)

    constraints = [u_seq >= u_min, u_seq <= u_max]
    for t in range(T):
        dx = x_trace[t] - y_trace[t][0]
        dy = car_y - y_trace[t][1]
        dist_expr = cp.norm(cp.hstack([dx, dy]))
        constraints.append(dist_expr >= d_safe)

    objective = cp.Minimize((u_seq[0] - u_des)**2)
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.OSQP)
        if u_seq.value is None:
            return u_min
        return float(u_seq.value[0])
    except:
        return u_min
