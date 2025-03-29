import cvxpy as cp
from mpc.car_dynamics import forward_car
from mpc.ped_dynamics import forward_ped_trace
import numpy as np

def vanilla_cbf_controller(x0, y0, T=10, d_safe=10.0, car_y=320.0):
    """
    使用cvxpy求解轨迹CBF问题，返回当前控制输入 u0。

    参数：
        x0: 当前车辆横向位置
        y0: 当前行人位置 [x, y]
        forward_ped_trace: 给定 car 轨迹，返回预测的行人轨迹函数
        T: 预测步长
        d_safe: 安全距离
        car_y: 车道纵向固定坐标

    返回：
        u0: 当前步速度（float）
    """
    u_des = 15.0
    u_min, u_max = 0.0, 15.0

    # Step 1: 尝试匀速控制是否安全
    u_nominal = np.full(T, u_des)
    x_nominal = forward_car(x0, u_nominal)
    y_nominal = forward_ped_trace(y0, x_nominal)

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
    y_trace = forward_ped_trace(y0, x_trace_sample)

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
