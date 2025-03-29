import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from models.model_def import WalkerSpeedPredictor

# 加载模型
model = WalkerSpeedPredictor()
model.load_state_dict(torch.load("assets/walker_speed_predictor.pth"))
model.eval()

# 加载 CP Grid
with open("assets/conformal_grid.pkl", "rb") as f:
    cp_grid = pickle.load(f)

# 准备数据：15 个 car_speed bin（[0, 1), [1, 2), ..., [14, 15)）
car_bins = np.linspace(0, 15, 16)
car_centers = (car_bins[:-1] + car_bins[1:]) / 2

# 每个 bin 采样若干 walker_y（这里选 275, 375 分别对应 zone 0, 1）
walker_y = 275  # 固定一个 walker_y（你可以尝试不同值）
v_par_means, v_perp_means = [], []
v_par_etas, v_perp_etas = [], []

for i in range(15):  # car speed bin
    car_speed = car_centers[i]
    input_tensor = torch.tensor([[car_speed, walker_y]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(input_tensor).numpy()[0]
    
    v_par_means.append(pred[0])
    v_perp_means.append(pred[1])

    j = min(int(pred[1]), 2)  # v_perp bin
    k = 0                     # v_par 只有一个 bin
    m = 0 if walker_y < 300 else 1

    eta_par, eta_perp = cp_grid.get((i, j, k, m), (0.5, 0.5))
    v_par_etas.append(eta_par)
    v_perp_etas.append(eta_perp)

# 绘图
plt.figure(figsize=(10, 6))

# Δa_perp
plt.plot(car_centers, v_perp_means, color='blue', label=r'$\hat{a}_{\perp}$ Mean')
plt.fill_between(car_centers,
                 np.array(v_perp_means) - np.array(v_perp_etas),
                 np.array(v_perp_means) + np.array(v_perp_etas),
                 color='blue', alpha=0.2, label=r'$\eta_{\perp}$ Region')

# Δa_par
plt.plot(car_centers, v_par_means, color='red', label=r'$\hat{a}_{\parallel}$ Mean')
plt.fill_between(car_centers,
                 np.array(v_par_means) - np.array(v_par_etas),
                 np.array(v_par_means) + np.array(v_par_etas),
                 color='red', alpha=0.2, label=r'$\eta_{\parallel}$ Region')

ground_truth_vx = []
ground_truth_vy = []

for car_v in car_centers:
    if car_v > 10:
        walker_v_y = 1
        walker_v_x = 0.8
    elif 5 < car_v <= 10:
        walker_v_y = 2
        walker_v_x = 0.5
    else:
        walker_v_y = 3
        walker_v_x = 0
    ground_truth_vx.append(walker_v_x)
    ground_truth_vy.append(walker_v_y)

# 在图中添加虚线
plt.plot(car_centers, ground_truth_vy, 'b--', label=r'Ground Truth $a_{\perp}$')
plt.plot(car_centers, ground_truth_vx, 'r--', label=r'Ground Truth $a_{\parallel}$')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Car Speed (dm/s)', fontsize=12)
plt.ylabel(r'$a_t$ and Conformal Region (dm/s)', fontsize=12)
plt.title('Predicted $\hat{a}_t$ with Conformal Region vs Car Speed', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
