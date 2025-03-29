import numpy as np
import cvxpy as cp

def cbf_controller(car_x, walker_xy, T=10, N=100, d_safe=10.0):
    """
    Quadratic Program (QP)-based Control Barrier Function controller.
    """
    walker_x, walker_y = walker_xy
    car_y = 320.0  # 假设车道固定

    # Parameters
    alpha = 1.0
    u_max = 15.0
    u_min = 0.0
    u_des = u_max  # 想尽可能快

    # Compute distance and gradient
    dx = car_x - walker_x
    dy = car_y - walker_y
    dist = np.sqrt(dx**2 + dy**2)

    # If too close (dist≈0), return conservative
    if dist < 1e-2:
        return u_min

    # Define control variable
    u = cp.Variable()

    # CBF constraint: -dx/dist * u + alpha * (dist - d_safe) >= 0
    grad_h = -dx / dist  # since ∂h/∂x_car = -(x_car - x_walker) / |x - y|
    h_val = dist - d_safe
    cbf_constraint = grad_h * u + alpha * h_val >= 0

    # Define and solve QP
    objective = cp.Minimize((u - u_des)**2)
    constraints = [cbf_constraint, u >= u_min, u <= u_max]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP)
        if u.value is None:
            return u_min
        return float(u.value)
    except:
        return u_min  # fallback if solver fails
