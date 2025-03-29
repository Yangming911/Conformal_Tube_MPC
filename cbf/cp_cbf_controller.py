from mpc.car_dynamics import forward_car
from mpc.ped_dynamics import forward_ped_trace
from models.conformal_grid import get_eta
import numpy as np
import cvxpy as cp

def cp_cbf_controller(x0, y0, T=10, d_safe=10.0, car_y=320.0):
    """
    考虑 conformal 预测误差的轨迹CBF控制器，输出当前控制输入 u0。
    """
    u_des = 15.0
    u_min, u_max = 0.0, 15.0

    # Step 1: 尝试匀速是否安全
    u_nominal = np.full(T, u_des)
    x_nominal = forward_car(x0, u_nominal)
    y_nominal = forward_ped_trace(y0, x_nominal)

    is_safe = True
    for t in range(T):
        car_pos = np.array([x_nominal[t], car_y])
        ped_pos = np.array(y_nominal[t])

        # 提取速度信息用于获取 eta
        car_speed = u_nominal[t]
        v_x = 0.0  # conformal grid里 vx 不分bin
        v_y = np.linalg.norm([ped_pos[0] - y0[0], ped_pos[1] - y0[1]]) / (t + 1e-3)
        walker_y = ped_pos[1]
        eta_x, eta_y = get_eta(car_speed, v_x, v_y, walker_y)
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
    y_trace = forward_ped_trace(y0, x_sample)

    constraints = [u_seq >= u_min, u_seq <= u_max]
    for t in range(T):
        dx = x_trace[t] - y_trace[t][0]
        dy = car_y - y_trace[t][1]

        # 同样获取 eta 并构造 d_eff
        car_speed = u_des  # 用 sample 轨迹的速度近似
        v_x = 0.0
        v_y = np.linalg.norm([y_trace[t][0] - y0[0], y_trace[t][1] - y0[1]]) / (t + 1e-3)
        walker_y = y_trace[t][1]
        eta_x, eta_y = get_eta(car_speed, v_x, v_y, walker_y)
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
