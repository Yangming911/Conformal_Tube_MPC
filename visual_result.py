# # result plot
# n_ped = [1,3,5,7,9]
# collision_rate_ours = [1/200, 1/200, 1/200, 1/200, 0/200]
# collision_rate_ACP  = [6/200,11/200,15/200,15/200,13/200]
# acc_mean_ours = [0.77, 1.30, 1.34, 1.32, 1.28]
# acc_mean_ACP = [0.89, 2.37, 3.03, 3.55, 4.53]
# avg_speed_ours = [12.52, 9.69, 8.39, 7.75, 7.43]
# avg_speed_ACP = [9.64, 7.83, 7.04, 6.59, 6.40]
# avg_step_ours = [40.63, 52.38, 60.48, 65.49, 68.29]
# avg_step_ACP = [46.20, 56.00, 61.90, 66.00, 68.10]

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 4))
# plt.subplot(2, 2, 1)
# plt.plot(n_ped, collision_rate_ours, label="Ours")
# plt.plot(n_ped, collision_rate_ACP, label="ACP")
# plt.legend()
# plt.xlabel("Number of Pedestrians")
# plt.ylabel("Collision Rate")
# plt.title("Collision Rate")

# plt.subplot(2, 2, 2)
# plt.plot(n_ped, acc_mean_ours, label="Ours")
# plt.plot(n_ped, acc_mean_ACP, label="ACP")
# plt.legend()
# plt.xlabel("Number of Pedestrians")
# plt.ylabel("Acceleration")
# plt.title("Acceleration")

# plt.subplot(2, 2, 3)
# plt.plot(n_ped, avg_speed_ours, label="Ours")
# plt.plot(n_ped, avg_speed_ACP, label="ACP")
# plt.legend()
# plt.xlabel("Number of Pedestrians")
# plt.ylabel("Average Speed")
# plt.title("Average Speed")

# plt.subplot(2, 2, 4)
# plt.plot(n_ped, avg_step_ours, label="Ours")
# plt.plot(n_ped, avg_step_ACP, label="ACP")
# plt.legend()
# plt.xlabel("Number of Pedestrians")
# plt.ylabel("Average Step")
# plt.title("Average Step")

# plt.tight_layout()
# plt.savefig("result.png")
# plt.close()

import json
from typing import List, Dict
import numpy as np

def plot_trajectory(states1: List[Dict], states2: List[Dict], collide_state: Dict, filename: str) -> None:
    """
    Plot the trajectory of the vehicle and pedestrians.
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ped_traj1 = np.array([[state["walker_x"], state["walker_y"]] for state in states1])
    veh_traj1 = np.array([[state["car_x"], state["car_y"]] for state in states1])
    veh_speed1 = np.array([state["car_v"] for state in states1])
    ped_traj2 = np.array([[state["walker_x"], state["walker_y"]] for state in states2])
    veh_traj2 = np.array([[state["car_x"], state["car_y"]] for state in states2])
    veh_speed2 = np.array([state["car_v"] for state in states2])
    pre_ped_traj_ori1 = np.array([state["pre_p_ped"] for state in states1])
    eta0_ori1 = np.array([state["eta"] for state in states1])
    pre_ped_traj_ori2 = np.array([state["pre_p_ped"] for state in states2])
    eta0_ori2 = np.array([state["eta"] for state in states2])

    colors1 = plt.cm.viridis(np.linspace(0, 1, len(states1)))
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(states2)))

    plt.figure(figsize=(6, 4))
    ped_idx1 = 1
    ped_idx2 = 0
    plt.subplot(1, 2, 1)
    plt.scatter(ped_traj1[:, 0, ped_idx1], ped_traj1[:, 1, ped_idx1], label=f"Real Position", s=10, color=colors1)
    plt.scatter(pre_ped_traj_ori1[:, ped_idx1, 0], pre_ped_traj_ori1[:, ped_idx1, 1], label=f"Predicted Position",marker="x", s=10, color=colors1)
    # 以当前点为中心，绘制以eta为半径的圆
    for t in range(eta0_ori1.shape[0]):
        circle = plt.Circle((pre_ped_traj_ori1[t, ped_idx1, 0], pre_ped_traj_ori1[t, ped_idx1, 1]), eta0_ori1[t], color=colors1[t], alpha=0.2)
        plt.gca().add_patch(circle)
    plt.subplot(1, 2, 2)
    plt.scatter(ped_traj2[:, 0, ped_idx2], ped_traj2[:, 1, ped_idx2], label=f"Real Position", s=10, color=colors2)
    plt.scatter(pre_ped_traj_ori2[:, ped_idx2, 0], pre_ped_traj_ori2[:, ped_idx2, 1], label=f"Predicted Position",marker="x", s=10, color=colors2)
    # 以当前点为中心，绘制以eta为半径的圆
    for t in range(eta0_ori2.shape[0]):
        circle = plt.Circle((pre_ped_traj_ori2[t, ped_idx2, 0], pre_ped_traj_ori2[t, ped_idx2, 1]), eta0_ori2[t], color=colors2[t], alpha=0.2)
        plt.gca().add_patch(circle)
    # plt.xlabel("X")
    # plt.ylabel("Y")
    plt.legend()
    plt.savefig(filename.replace(".png", "_trajectory.png"))
    plt.close()
    print("Trajectory with eta saved to:", filename.replace(".png", "_trajectory.png"))

    # 画error图，并用eta bound住
    plt.figure(figsize=(10,4))
    error = np.linalg.norm(pre_ped_traj_ori1[:, ped_idx1, :2] - ped_traj1[:, :2, ped_idx1], axis=1)
    plt.plot(error[:30], label=f"Position error Ours", color=colors1[0])
    plt.fill_between(range(30), 0, eta0_ori1[:30], color=colors1[0], alpha=0.2, label=f"Conformal Region Ours")

    error = np.linalg.norm(pre_ped_traj_ori2[:, ped_idx2, :2] - ped_traj2[:, :2, ped_idx2], axis=1)
    plt.plot(error[:30], label=f"Position error ACP", color=colors2[-1])
    plt.fill_between(range(30), 0, eta0_ori2[:30], color=colors2[-1], alpha=0.2, label=f"Conformal Region ACP")

    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig(filename.replace(".png", "_error.png"))
    plt.close()
    print("Error with eta saved to:", filename.replace(".png", "_error.png"))


with open('9_10_10_1_our.json', 'r') as f:
    states_ours = json.load(f)
with open('9_10_10_1_ACP.json', 'r') as f:
    states_ACP = json.load(f)

plot_trajectory(states_ours, states_ACP, None, "./result.png")