import numpy as np
import pandas as pd
from envs.dynamics import walker_logic

def data_collecting(num_samples=10000):
    data = []
    for _ in range(num_samples):
        car_speed = np.random.uniform(1, 15)
        walker_y = np.random.uniform(200, 450)  # 可选：覆盖前半段
        walker_speed_x, walker_speed_y = walker_logic(car_speed, walker_y)
        data.append([car_speed, walker_y, walker_speed_x, walker_speed_y])

    df = pd.DataFrame(data, columns=['car_speed', 'walker_y', 'walker_speed_x', 'walker_speed_y'])
    print(df.head())
    return data
