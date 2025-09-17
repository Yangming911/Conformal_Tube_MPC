import numpy as np

def walker_logic(car_v, walker_y_position):
    walker_noise_y_sigma = 0.5
    walker_noise_x_sigma = 0.1

    if walker_y_position < 300:
        if car_v > 10:
            walker_v_y = 1
            walker_v_x = 0.8
        elif 5 < car_v <= 10:
            walker_v_y = 2
            walker_v_x = 0.5
        else:
            walker_v_y = 3
            walker_v_x = 0
    elif walker_y_position < 450:
        if car_v > 10:
            walker_v_y = 1
            walker_v_x = -0.8
        elif 5 < car_v <= 10:
            walker_v_y = 2
            walker_v_x = -0.5
        else:
            walker_v_y = 3
            walker_v_x = 0

    walker_v_y += np.random.normal(0, walker_noise_y_sigma)
    walker_v_x += np.random.normal(0, walker_noise_x_sigma)
    return walker_v_x, walker_v_y

def update_car_position(car_x_position, car_v, dt=0.1):
    car_x_position += car_v * dt  # 正确使用 x = x + v*dt
    if car_x_position > 800:
        car_x_position = -40
        car_v = np.random.uniform(low=1, high=15)
    return car_x_position, car_v

def update_walker_position(car_v, walker_x_position, walker_y_position, dt=0.1):
    if 200 <= walker_y_position < 450:
        walker_v_x, walker_v_y = walker_logic(car_v, walker_y_position)
        walker_y_position += walker_v_y * dt  # 正确使用 x = x + v*dt
        walker_x_position += walker_v_x * dt  # 正确使用 x = x + v*dt
    else:
        walker_y_position = 200
        walker_x_position = 400
    return walker_x_position, walker_y_position
