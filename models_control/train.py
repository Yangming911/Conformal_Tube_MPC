#!/usr/bin/env python3
"""
Train causal pedestrian position predictor

Inputs:
- u sequence (scalar control): shape [B, T, 1]
- p_veh0 (vehicle initial position): shape [B, 2]
- p_ped0 (pedestrian initial position): shape [B, 2]

Targets:
- p_ped[1..T] (positions): shape [B, T, 2]

Data format expected (CSV by default):
- Columns: u0,...,u{T-1}, p_veh0_x, p_veh0_y, p_ped0_x, p_ped0_y,
           p_ped1_x, p_ped1_y, ..., p_ped{T}_x, p_ped{T}_y

You can also point to a .npz file with arrays: 'u' [N,T,1], 'p_veh0' [N,2],
'p_ped0' [N,2], 'p_ped_seq' [N,T,2].
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict

# Project root on path BEFORE importing project modules
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utils.constants as C
from envs.dynamics_social_force import walker_logic_SF

from models_control.model_def import CausalPedestrianPredictor, compute_sequence_loss
from tools.collect_control_sequences import collect_dataset as collect_constant_speed_dataset
from tools.collect_control_sequences import save_csv as save_constant_speed_csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SequenceDataset(Dataset):
    def __init__(self, u: np.ndarray, p_veh0: np.ndarray, p_ped0: np.ndarray, p_ped_seq: np.ndarray):
        assert u.ndim == 3 and u.shape[-1] == 1
        assert p_veh0.shape[-1] == 2 and p_ped0.shape[-1] == 2
        assert p_ped_seq.ndim == 3 and p_ped_seq.shape[-1] == 2
        self.u = torch.from_numpy(u).float()
        self.p_veh0 = torch.from_numpy(p_veh0).float()
        self.p_ped0 = torch.from_numpy(p_ped0).float()
        self.p_ped_seq = torch.from_numpy(p_ped_seq).float()

    def __len__(self) -> int:
        return self.u.shape[0]

    def __getitem__(self, idx: int):
        return self.u[idx], self.p_veh0[idx], self.p_ped0[idx], self.p_ped_seq[idx]


def load_from_csv(csv_path: str, T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    u_cols = [f"u{i}" for i in range(T)]
    veh_cols = ["p_veh0_x", "p_veh0_y"]
    ped0_cols = ["p_ped0_x", "p_ped0_y"]
    ped_seq_cols = []
    for t in range(1, T + 1):
        ped_seq_cols += [f"p_ped{t}_x", f"p_ped{t}_y"]

    u = df[u_cols].to_numpy(dtype=np.float32).reshape(-1, T, 1)
    p_veh0 = df[veh_cols].to_numpy(dtype=np.float32)
    p_ped0 = df[ped0_cols].to_numpy(dtype=np.float32)
    ped_seq = df[ped_seq_cols].to_numpy(dtype=np.float32).reshape(-1, T, 2)
    return u, p_veh0, p_ped0, ped_seq


def simulate_one_sequence(u_seq: np.ndarray, p_veh0: np.ndarray, p_ped0: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Roll out T steps using Social Force model with control sequence u as car speed.

    Args:
        u_seq: [T, 1]
        p_veh0: [2] (x, y)
        p_ped0: [2] (x, y)
    Returns:
        p_ped_seq: [T, 2] positions for steps 1..T
    """
    T = u_seq.shape[0]
    car_x, car_y = float(p_veh0[0]), float(p_veh0[1])
    walker_x, walker_y = float(p_ped0[0]), float(p_ped0[1])
    walker_vx, walker_vy = float(C.WALKER_START_V_X), float(C.WALKER_START_V_Y)
    dt = float(C.dt)

    out = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        car_v = float(u_seq[t, 0])
        next_vx, next_vy = walker_logic_SF(
            car_v,
            car_x,
            car_y,
            walker_x,
            walker_y,
            walker_vx,
            walker_vy,
            v_max=C.v_max,
            a_max=C.a_max,
            destination_y=C.WALKER_DESTINATION_Y,
            rng=rng,
        )
        # integrate positions
        walker_x = walker_x + next_vx * dt
        walker_y = walker_y + next_vy * dt
        car_x = car_x + car_v * dt
        # update velocities
        walker_vx, walker_vy = next_vx, next_vy
        out[t, 0] = walker_x
        out[t, 1] = walker_y
    return out


def collect_sequence_dataset(num_episodes: int, T: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    u_all = np.zeros((num_episodes, T, 1), dtype=np.float32)
    pveh0_all = np.zeros((num_episodes, 2), dtype=np.float32)
    pped0_all = np.zeros((num_episodes, 2), dtype=np.float32)
    pseq_all = np.zeros((num_episodes, T, 2), dtype=np.float32)

    for i in range(num_episodes):
        # Random control speeds within realistic range [1, 15] m/s
        u_seq = rng.uniform(low=1.0, high=15.0, size=(T, 1)).astype(np.float32)

        # Initial vehicle and pedestrian positions
        p_veh0 = np.array([float(C.CAR_START_X), float(C.CAR_LANE_Y)], dtype=np.float32)
        p_ped0 = np.array([
            float(C.WALKER_START_X),
            float(rng.uniform(C.WALKER_START_Y, C.WALKER_DESTINATION_Y)),
        ], dtype=np.float32)

        p_seq = simulate_one_sequence(u_seq, p_veh0, p_ped0, rng)

        u_all[i] = u_seq
        pveh0_all[i] = p_veh0
        pped0_all[i] = p_ped0
        pseq_all[i] = p_seq

    return u_all, pveh0_all, pped0_all, pseq_all


def split_train_val_test(N: int, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    n_test = int(N * test_ratio)
    n_val = int((N - n_test) * val_ratio)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    return train_idx, val_idx, test_idx


def visualize_predictions(model: nn.Module, loader: DataLoader, device: torch.device, 
                         save_dir: str = "assets/visualizations", num_samples: int = 10):
    """
    可视化模型预测结果，对比预测值和真实值
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集样本数据
    all_preds = []
    all_targets = []
    all_u = []
    all_p_veh0 = []
    all_p_ped0 = []
    
    with torch.no_grad():
        for u, p_veh0, p_ped0, p_seq in loader:
            u = u.to(device)
            p_veh0 = p_veh0.to(device)
            p_ped0 = p_ped0.to(device)
            p_seq = p_seq.to(device)
            
            pred = model(u, p_veh0, p_ped0)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(p_seq.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_p_veh0.append(p_veh0.cpu().numpy())
            all_p_ped0.append(p_ped0.cpu().numpy())
            
            if len(all_preds) * loader.batch_size >= num_samples:
                break
    
    # 合并数据
    preds = np.concatenate(all_preds, axis=0)[:num_samples]
    targets = np.concatenate(all_targets, axis=0)[:num_samples]
    u_vals = np.concatenate(all_u, axis=0)[:num_samples]
    p_veh0_vals = np.concatenate(all_p_veh0, axis=0)[:num_samples]
    p_ped0_vals = np.concatenate(all_p_ped0, axis=0)[:num_samples]
    
    # 创建可视化
    create_comprehensive_visualizations(preds, targets, u_vals, p_veh0_vals, p_ped0_vals, save_dir)

def create_comprehensive_visualizations(preds, targets, u_vals, p_veh0_vals, p_ped0_vals, save_dir):
    """
    创建全面的可视化图表
    """
    num_samples = len(preds)
    T = preds.shape[1]
    
    # 1. 单个样本轨迹对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    samples = np.random.choice(num_samples, size=min(6, num_samples), replace=False)
    
    for i, sample_idx in enumerate(samples):
        ax = axes[i]
        
        # 提取数据
        pred_traj = preds[sample_idx]  # [T, 2]
        true_traj = targets[sample_idx]  # [T, 2]
        u_seq = u_vals[sample_idx, :, 0]  # [T]
        
        # 绘制轨迹
        ax.plot(true_traj[:, 0], true_traj[:, 1], 'b-', linewidth=2, label='gt', alpha=0.8)
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, label='pred', alpha=0.8)
        
        # 标记起点和终点
        ax.scatter(true_traj[0, 0], true_traj[0, 1], c='green', s=100, label='start', marker='o')
        ax.scatter(true_traj[-1, 0], true_traj[-1, 1], c='red', s=100, label='goal', marker='s')
        ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], c='orange', s=100, label='pred goal', marker='^')
        
        # 添加时间步标记
        for t in range(0, T, max(1, T//5)):
            ax.annotate(f't={t}', (true_traj[t, 0], true_traj[t, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax.annotate(f't={t}', (pred_traj[t, 0], pred_traj[t, 1]), 
                       xytext=(5, -15), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'sample {i+1} (u_seq: {u_seq[0]:.1f} m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 误差分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 计算误差
    errors = np.linalg.norm(preds - targets, axis=2)  # [num_samples, T]
    
    # 误差随时间变化
    ax = axes[0, 0]
    mean_errors = errors.mean(axis=0)
    std_errors = errors.std(axis=0)
    ax.plot(range(T), mean_errors, 'b-', linewidth=2, label='ave error')
    ax.fill_between(range(T), mean_errors - std_errors, mean_errors + std_errors, 
                   alpha=0.3, label='±1std')
    ax.set_xlabel('time step')
    ax.set_ylabel('L2 error (meters)')
    ax.set_title('error over time steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 误差分布直方图
    ax = axes[0, 1]
    ax.hist(errors.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('L2 error (meters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution Histogram')
    ax.grid(True, alpha=0.3)
    
    # 最终位置误差 vs 控制速度
    ax = axes[1, 0]
    final_errors = errors[:, -1]
    mean_u = u_vals.mean(axis=1)  # 平均控制速度
    ax.scatter(mean_u, final_errors, alpha=0.6)
    ax.set_xlabel('average control speed (m/s)')
    ax.set_ylabel('final L2 error (meters)')
    ax.set_title('Final Position Error vs Control Speed')
    ax.grid(True, alpha=0.3)
    
    # 累积误差热力图
    ax = axes[1, 1]
    error_matrix = errors[:min(20, num_samples)]  # 限制样本数量
    im = ax.imshow(error_matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel('time step')
    ax.set_ylabel('sample index')
    ax.set_title('Cumulative Error Heatmap')
    plt.colorbar(im, ax=ax, label='L2 error (meters)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 详细统计报告
    create_detailed_report(preds, targets, errors, save_dir)

def create_detailed_report(preds, targets, errors, save_dir):
    """
    创建详细的统计报告
    """
    report = {
        '总体统计': {
            '平均误差': f"{errors.mean():.4f} ± {errors.std():.4f} 米",
            '最大误差': f"{errors.max():.4f} 米",
            '最小误差': f"{errors.min():.4f} 米",
            '中位数误差': f"{np.median(errors):.4f} 米"
        },
        '时间步统计': {
            f'第{t}步平均误差': f"{errors[:, t].mean():.4f} 米" 
            for t in range(errors.shape[1])
        }
    }
    
    # 保存报告
    with open(f'{save_dir}/prediction_report.txt', 'w') as f:
        f.write("模型预测性能报告\n")
        f.write("="*50 + "\n\n")
        
        for section, stats in report.items():
            f.write(f"{section}:\n")
            f.write("-"*30 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"可视化结果已保存到: {save_dir}")

def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, 
                                    scaler: Dict[str, StandardScaler] = None, visualize: bool = False) -> Dict[str, float]:
    """
    增强的评估函数，包含可视化选项
    """
    model.eval()
    mse_loss = nn.MSELoss(reduction="mean")
    total = 0
    sum_mse = 0.0
    
    # 收集数据用于可视化
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for u, p_veh0, p_ped0, p_seq in loader:
            u = u.to(device)
            p_veh0 = p_veh0.to(device)
            p_ped0 = p_ped0.to(device)
            p_seq = p_seq.to(device)
            pred = model(u, p_veh0, p_ped0)
            loss = mse_loss(pred, p_seq)
            
            # 收集数据
            if visualize:
                all_preds.append(pred.cpu().numpy())
                all_targets.append(p_seq.cpu().numpy())
            
            bs = u.size(0)
            sum_mse += loss.item() * bs
            total += bs
    
    avg_mse = sum_mse / max(total, 1)

    # 反归一化
    if scaler:
        p_seq_reshape = p_seq.reshape(-1, 2).detach().cpu().numpy()
        pred_reshape = pred.reshape(-1, 2).detach().cpu().numpy()
        p_seq_reshape = scaler['p_seq_scaler'].inverse_transform(p_seq_reshape)
        pred_reshape = scaler['p_seq_scaler'].inverse_transform(pred_reshape)
        p_seq = p_seq_reshape.reshape(p_seq.shape)
        pred = pred_reshape.reshape(pred.shape)

    # 计算每步MSE
    per_step_sums = None
    per_step_counts = 0
    with torch.no_grad():
        for u, p_veh0, p_ped0, p_seq in loader:
            u = u.to(device)
            p_veh0 = p_veh0.to(device)
            p_ped0 = p_ped0.to(device)
            p_seq = p_seq.to(device)
            pred = model(u, p_veh0, p_ped0)
            err2 = (pred - p_seq) ** 2
            step_mse = err2.mean(dim=2).mean(dim=0)
            if per_step_sums is None:
                per_step_sums = step_mse.detach().cpu()
            else:
                per_step_sums += step_mse.detach().cpu()
            per_step_counts += 1

    per_step_mse = (per_step_sums / max(per_step_counts, 1)).numpy().tolist()
    
    # 如果启用可视化，生成图表
    if visualize and all_preds:
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        create_comprehensive_visualizations(preds[:10], targets[:10], 
                                          np.zeros((10, 10, 1)),  # 简化处理
                                          np.zeros((10, 2)), 
                                          np.zeros((10, 2)), 
                                          "results")

    return {"mse": float(avg_mse), "rmse": float(np.sqrt(avg_mse)), "per_step_mse": per_step_mse}


def main():
    parser = argparse.ArgumentParser(description="Train causal pedestrian position predictor")
    parser.add_argument('--data', type=str, default='assets/control_sequences.csv', help='Path to CSV dataset')
    parser.add_argument('--citr_data', type=str, default='assets/citr_data/citr_train.csv')
    parser.add_argument('--T', type=int, default=10, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dim')
    parser.add_argument('--layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--save_path', type=str, default='assets/control_ped_model.pth', help='Checkpoint path')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--from_sim', action='store_true', help='Collect training data from social force simulator (per-step random speed)')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of simulated episodes/sequences when generating data')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (set to 0 to disable)')
    args = parser.parse_args()

    # Resolve dataset: prefer existing file, otherwise auto-generate under assets
    data_path = args.data
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Auto-generating using Social Force simulator (constant per-episode speed)...")
        u, p_veh0, p_ped0, p_seq = collect_constant_speed_dataset(args.episodes, args.T, seed=42)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        save_constant_speed_csv(data_path, u, p_veh0, p_ped0, p_seq)
        print(f"Generated dataset saved to {data_path}")

    # Load dataset into memory
    if args.from_sim and os.path.exists(data_path) is False:
        # Fallback: per-step random speed generator (legacy option)
        print(f"Collecting simulated dataset with per-step random speed: episodes={args.episodes}, T={args.T}")
        u, p_veh0, p_ped0, p_seq = collect_sequence_dataset(args.episodes, args.T, seed=42)
        print(f"Sim data shapes: u={u.shape}, p_veh0={p_veh0.shape}, p_ped0={p_ped0.shape}, p_seq={p_seq.shape}")
    else:
        u, p_veh0, p_ped0, p_seq = load_from_csv(data_path, args.T)
        # u_real, p_veh0_real, p_ped0_real, p_seq_real = load_from_csv(args.citr_data, args.T)
        # u = np.concatenate([u, u_real], axis=0)
        # p_veh0 = np.concatenate([p_veh0, p_veh0_real], axis=0)
        # p_ped0 = np.concatenate([p_ped0, p_ped0_real], axis=0)
        # p_seq = np.concatenate([p_seq, p_seq_real], axis=0)

    N = u.shape[0]
    train_idx, val_idx, test_idx = split_train_val_test(N)
    # # scale dataset

    # u_scaler = StandardScaler()
    # p_veh0_scaler = StandardScaler()
    # p_ped0_scaler = StandardScaler()
    # p_seq_scaler = StandardScaler()

    # # 修复：将3维数据重塑为2维进行标准化
    # # u的形状是 [N, T, 1]，需要重塑为 [N*T, 1]
    # u_reshaped = u.reshape(-1, 1)
    # u_scaler.fit(u_reshaped)
    # u = u_scaler.transform(u_reshaped).reshape(u.shape)
    
    # # p_veh0的形状是 [N, 2]，可以直接标准化
    # p_veh0 = p_veh0_scaler.fit_transform(p_veh0)
    
    # # p_ped0的形状是 [N, 2]，可以直接标准化
    # p_ped0 = p_ped0_scaler.fit_transform(p_ped0)
    
    # # p_seq的形状是 [N, T, 2]，需要重塑为 [N*T, 2]
    # p_seq_reshaped = p_seq.reshape(-1, 2)
    # p_seq_scaler.fit(p_seq_reshaped)
    # p_seq = p_seq_scaler.transform(p_seq_reshaped).reshape(p_seq.shape)

    train_ds = SequenceDataset(u[train_idx], p_veh0[train_idx], p_ped0[train_idx], p_seq[train_idx])
    val_ds = SequenceDataset(u[val_idx], p_veh0[val_idx], p_ped0[val_idx], p_seq[val_idx])
    test_ds = SequenceDataset(u[test_idx], p_veh0[test_idx], p_ped0[test_idx], p_seq[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CausalPedestrianPredictor(u_dim=1, hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout)
    # model = LSTMPedestrianPredictor(u_dim=1, hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for u_b, pveh_b, pped0_b, pseq_b in train_loader:
            u_b = u_b.to(device)
            pveh_b = pveh_b.to(device)
            pped0_b = pped0_b.to(device)
            pseq_b = pseq_b.to(device)

            pred = model(u_b, pveh_b, pped0_b)
            loss, _ = compute_sequence_loss(pred, pseq_b)

            optimizer.zero_grad()
            loss.backward()
            
            # 添加梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()

            epoch_losses.append(loss.item())

        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics["mse"])

        if (epoch + 1) % 10 == 0:
            step_str = ", ".join([f"t{t+1}:{m:.4f}" for t, m in enumerate(val_metrics.get('per_step_mse', []))])
            print(f"Epoch {epoch+1}/{args.epochs} | TrainLoss: {np.mean(epoch_losses):.6f} | ValMSE: {val_metrics['mse']:.6f} | ValRMSE: {val_metrics['rmse']:.6f} | ValStepMSE: [{step_str}]")

        # Early stopping
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            patience_counter = 0
            # Save checkpoint
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'model_name': 'CausalPedestrianPredictor',
                    'T': args.T,
                    'u_dim': 1,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.layers,
                    'dropout': args.dropout,
                },
                # 'u_scaler': u_scaler,
                # 'p_veh0_scaler': p_veh0_scaler,
                # 'p_ped0_scaler': p_ped0_scaler,
                # 'p_seq_scaler': p_seq_scaler,
            }, args.save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                # print(f"Early stopping at epoch {epoch+1}")
                # break
                print("update learning rate")
                lr = args.lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                patience_counter = 0


    # 在测试评估后添加可视化
    # Load best and evaluate on test
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # scaler = {
    #     'u_scaler': checkpoint['u_scaler'],
    #     'p_veh0_scaler': checkpoint['p_veh0_scaler'],
    #     'p_ped0_scaler': checkpoint['p_ped0_scaler'],
    #     'p_seq_scaler': checkpoint['p_seq_scaler'],
    # }
    
    
    test_metrics = evaluate_model(model, test_loader, device)
    step_str = ", ".join([f"t{t+1}:{m:.4f}" for t, m in enumerate(test_metrics.get('per_step_mse', []))])
    print(f"Test MSE: {test_metrics['mse']:.6f} | Test RMSE: {test_metrics['rmse']:.6f} | TestStepMSE: [{step_str}]")
    
    # 新增：测试时可视化
    print("开始生成预测可视化...")
    visualize_predictions(model, test_loader, device, num_samples=20)
    print("可视化完成！")


if __name__ == '__main__':
    main()