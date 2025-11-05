import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from models_control.model_def import ACPCausalPedestrianPredictor, CausalPedestrianPredictor
from models_control_ACP.run_mpc_ACP import nn_predict_positions_multi
from models_control.scp import nn_predict_positions
from tools.collect_control_sequences import simulate_one_sequence
# load ACP model
ACP_model_path = os.path.join("assets_ACP/control_ped_model.pth")
error_npy_path = os.path.join("assets_ACP/cp_errors.npy")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(ACP_model_path, map_location=device)
hidden_dim = int(checkpoint.get('config', {}).get('hidden_dim', 128))
num_layers = int(checkpoint.get('config', {}).get('num_layers', 2))
dropout = float(checkpoint.get('config', {}).get('dropout', 0.1))
model_acp = ACPCausalPedestrianPredictor(p_dim=2, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
model_acp.load_state_dict(checkpoint['model_state_dict'])
model_acp.to(device)
model_acp.eval()

# load our model
model_path = os.path.join("assets/control_ped_model.pth")
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

# 定义不同的u值序列
different_u_sequences = [
    np.array([[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]], dtype=np.float32).T,  # 
    np.array([[15, 15, 15, 15, 15, 15, 15, 15, 15, 15]], dtype=np.float32).T,  # 
    np.array([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5]], dtype=np.float32).T,  # 
    np.array([[5, 7, 9, 11, 13, 15, 13, 11, 9, 7]], dtype=np.float32).T,    # 先加速后减速
    np.array([[15, 15, 10, 10, 5, 5, 10, 10, 15, 15]], dtype=np.float32).T,  # 波动速度
]
# 对应的标签
u_labels = [
    "Medium Speed (u=10)",
    "High Speed (u=15)",
    "Low Speed (u=5)",
    "Accelerate then Decelerate",
    "Fluctuating Speed"
]

past_ped = np.array([[[30.       ,  9.999069 ],
        [30.       , 10.099069 ],
        [30.000029 , 10.225465 ],
        [30.001083 , 10.3602495],
        [30.023096 , 10.497641 ],
        [30.03543  , 10.635512 ],
        [30.048586 , 10.773263 ],
        [30.06915  , 10.910527 ],
        [30.103958 , 11.04691  ],
        [30.162447 , 11.181846 ]]])


# 其他固定参数
# p_veh0 = np.array([[1.5, 12.0]], dtype=np.float32)  # [1,2]
# p_ped0 = np.array([[30.0, 10.769]], dtype=np.float32)  # [1,2]
p_ped0 = np.array([[30.162447 , 11.181846]], dtype=np.float32)  # [1,2]
p_veh0 = np.array([[10, 12.0]], dtype=np.float32)  # [1,2]

# 为不同的预测准备不同的颜色
colors = plt.cm.viridis(np.linspace(0, 1, len(different_u_sequences)))

pred_ped_ACP = nn_predict_positions_multi(model_acp, device, past_ped)[0]
# 计算所有u序列的预测结果
predictions = []
gt_peds = []
for u_seq in different_u_sequences:
    gt_ped = simulate_one_sequence(u_seq, p_veh0[0], p_ped0[0])
    pred = nn_predict_positions(model, device, u_seq, p_veh0, p_ped0)
    predictions.append(pred)
    gt_peds.append(gt_ped)
    
# 绘制图表
plt.figure(figsize=(12, 8))

# 绘制每个u序列的预测结果
for i, (pred_ped, gt_ped, color, label) in enumerate(zip(predictions, gt_peds, colors, u_labels)):
    plt.scatter(pred_ped[:, 0], pred_ped[:, 1], color=color, marker='^', s=50, label=label)
    plt.plot(pred_ped[:, 0], pred_ped[:, 1], color=color, alpha=0.7)
    plt.scatter(gt_ped[:, 0], gt_ped[:, 1], color=color, marker='o', s=100, label=f"Ground Truth {label}")
    plt.plot(gt_ped[:, 0], gt_ped[:, 1], color=color, linestyle='--', alpha=0.5)

# 绘制ACP的预测结果
plt.scatter(pred_ped_ACP[:, 0], pred_ped_ACP[:, 1], color='red', marker='^', s=50, label="ACP Predictions")
plt.plot(pred_ped_ACP[:, 0], pred_ped_ACP[:, 1], color='red', alpha=0.7)



plt.legend(fontsize=10, loc='best')
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.title("Pedestrian Trajectory Predictions with Different Control Inputs", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("multiple_u_predictions.png", dpi=300)
plt.close()
