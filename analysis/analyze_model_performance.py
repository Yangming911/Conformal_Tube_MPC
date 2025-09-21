#!/usr/bin/env python3
"""
Detailed model performance analysis
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch

# Add project root directory to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.predictor import WalkerActionPredictor
from envs.dynamics_social_force import walker_logic_SF
import utils.constants as C


def analyze_single_simulation(car_speed, walker_y_init, dt=0.1, max_steps=200):
    """Analyze single simulation"""
    predictor = WalkerActionPredictor('../assets/walker_speed_predictor_new.pth')
    
    # Initial state
    car_x = C.CAR_START_X
    car_y = C.CAR_LANE_Y
    walker_x = C.WALKER_START_X
    walker_y = walker_y_init
    walker_vx = C.WALKER_START_V_X
    walker_vy = C.WALKER_START_V_Y
    
    # 存储数据
    data = []
    
    for step in range(max_steps):
        # 实际下一步速度（来自social force模型）
        actual_vx, actual_vy = walker_logic_SF(
            car_speed, car_x, car_y, walker_x, walker_y, walker_vx, walker_vy,
            v_max=C.v_max, a_max=C.a_max, destination_y=C.WALKER_DESTINATION_Y
        )
        
        # 模型预测的下一步速度
        predicted_vx, predicted_vy = predictor.predict(
            car_x, car_y, car_speed, walker_x, walker_y, walker_vx, walker_vy
        )
        
        # 记录数据
        data.append({
            'step': step,
            'time': step * dt,
            'car_x': car_x,
            'car_y': car_y,
            'walker_x': walker_x,
            'walker_y': walker_y,
            'walker_vx': walker_vx,
            'walker_vy': walker_vy,
            'actual_vx': actual_vx,
            'actual_vy': actual_vy,
            'predicted_vx': predicted_vx,
            'predicted_vy': predicted_vy,
            'vx_error': actual_vx - predicted_vx,
            'vy_error': actual_vy - predicted_vy,
            'vx_error_abs': abs(actual_vx - predicted_vx),
            'vy_error_abs': abs(actual_vy - predicted_vy)
        })
        
        # 更新位置（使用实际速度）
        walker_x += actual_vx * dt
        walker_y += actual_vy * dt
        car_x += car_speed * dt
        
        # 更新速度（使用实际速度）
        walker_vx = actual_vx
        walker_vy = actual_vy
        
        # 检查终止条件
        if car_x > C.CAR_RIGHT_LIMIT or walker_y >= C.WALKER_DESTINATION_Y:
            break
    
    return pd.DataFrame(data)


def plot_detailed_analysis(df, car_speed, walker_y_init):
    """绘制详细分析图"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. 轨迹图
    ax1 = axes[0, 0]
    ax1.plot(df['car_x'], df['car_y'], 'b-', linewidth=3, label='Car', alpha=0.8)
    ax1.plot(df['walker_x'], df['walker_y'], 'r-', linewidth=2, label='Walker', alpha=0.8)
    ax1.scatter(df['car_x'].iloc[0], df['car_y'].iloc[0], c='blue', s=100, marker='o', label='Start')
    ax1.scatter(df['walker_x'].iloc[0], df['walker_y'].iloc[0], c='red', s=100, marker='o')
    ax1.scatter(df['walker_x'].iloc[-1], df['walker_y'].iloc[-1], c='red', s=100, marker='s', label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title(f'Trajectory\nCar Speed: {car_speed:.1f} m/s, Walker Y Init: {walker_y_init:.1f} m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 速度对比 - X方向
    ax2 = axes[0, 1]
    ax2.plot(df['time'], df['actual_vx'], 'r-', linewidth=2, label='Actual vx', alpha=0.8)
    ax2.plot(df['time'], df['predicted_vx'], 'g--', linewidth=2, label='Predicted vx', alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity X (m/s)')
    ax2.set_title('Velocity X Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 速度对比 - Y方向
    ax3 = axes[1, 0]
    ax3.plot(df['time'], df['actual_vy'], 'r-', linewidth=2, label='Actual vy', alpha=0.8)
    ax3.plot(df['time'], df['predicted_vy'], 'g--', linewidth=2, label='Predicted vy', alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity Y (m/s)')
    ax3.set_title('Velocity Y Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差分析
    ax4 = axes[1, 1]
    ax4.plot(df['time'], df['vx_error_abs'], 'b-', linewidth=2, label='|vx error|', alpha=0.8)
    ax4.plot(df['time'], df['vy_error_abs'], 'orange', linewidth=2, label='|vy error|', alpha=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Absolute Error (m/s)')
    ax4.set_title('Prediction Error Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 误差分布
    ax5 = axes[2, 0]
    ax5.hist(df['vx_error'], bins=20, alpha=0.7, label='vx error', color='blue')
    ax5.hist(df['vy_error'], bins=20, alpha=0.7, label='vy error', color='orange')
    ax5.set_xlabel('Error (m/s)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 累积误差
    ax6 = axes[2, 1]
    cumulative_vx_error = np.cumsum(df['vx_error_abs'])
    cumulative_vy_error = np.cumsum(df['vy_error_abs'])
    ax6.plot(df['time'], cumulative_vx_error, 'b-', linewidth=2, label='Cumulative |vx error|', alpha=0.8)
    ax6.plot(df['time'], cumulative_vy_error, 'orange', linewidth=2, label='Cumulative |vy error|', alpha=0.8)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Cumulative Absolute Error (m/s)')
    ax6.set_title('Cumulative Error')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_detailed_metrics(df):
    """打印详细指标"""
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"Simulation Duration: {df['time'].iloc[-1]:.2f} seconds")
    print(f"Total Steps: {len(df)}")
    print(f"Final Walker Position: ({df['walker_x'].iloc[-1]:.2f}, {df['walker_y'].iloc[-1]:.2f})")
    print(f"Final Car Position: ({df['car_x'].iloc[-1]:.2f}, {df['car_y'].iloc[-1]:.2f})")
    
    print("\nVelocity Prediction Errors:")
    print(f"  vx MAE: {df['vx_error_abs'].mean():.4f} m/s")
    print(f"  vx RMSE: {np.sqrt((df['vx_error']**2).mean()):.4f} m/s")
    print(f"  vx Max Error: {df['vx_error_abs'].max():.4f} m/s")
    print(f"  vy MAE: {df['vy_error_abs'].mean():.4f} m/s")
    print(f"  vy RMSE: {np.sqrt((df['vy_error']**2).mean()):.4f} m/s")
    print(f"  vy Max Error: {df['vy_error_abs'].max():.4f} m/s")
    
    print("\nError Statistics:")
    print(f"  vx Error Mean: {df['vx_error'].mean():.4f} m/s")
    print(f"  vx Error Std: {df['vx_error'].std():.4f} m/s")
    print(f"  vy Error Mean: {df['vy_error'].mean():.4f} m/s")
    print(f"  vy Error Std: {df['vy_error'].std():.4f} m/s")
    
    print("\nVelocity Ranges:")
    print(f"  Actual vx: [{df['actual_vx'].min():.4f}, {df['actual_vx'].max():.4f}] m/s")
    print(f"  Predicted vx: [{df['predicted_vx'].min():.4f}, {df['predicted_vx'].max():.4f}] m/s")
    print(f"  Actual vy: [{df['actual_vy'].min():.4f}, {df['actual_vy'].max():.4f}] m/s")
    print(f"  Predicted vy: [{df['predicted_vy'].min():.4f}, {df['predicted_vy'].max():.4f}] m/s")


def main():
    """主函数"""
    print("=" * 60)
    print("DETAILED MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # 使用与可视化脚本相同的参数
    np.random.seed(42)
    car_speed = 7.62  # 从之前的运行结果
    walker_y_init = 7.70  # 从之前的运行结果
    
    print(f"Analysis Parameters:")
    print(f"  Car Speed: {car_speed:.2f} m/s")
    print(f"  Walker Initial Y: {walker_y_init:.2f} m")
    print(f"  Time Step: 0.1 s")
    
    # 分析仿真
    print("\nAnalyzing simulation...")
    df = analyze_single_simulation(car_speed, walker_y_init)
    
    # 打印详细指标
    print_detailed_metrics(df)
    
    # 绘制详细分析图
    print("\nGenerating detailed analysis plot...")
    fig = plot_detailed_analysis(df, car_speed, walker_y_init)
    fig.savefig('../logs/detailed_model_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: ../logs/detailed_model_analysis.png")
    
    # 保存数据到CSV
    df.to_csv('../logs/simulation_analysis_data.csv', index=False)
    print("Saved: ../logs/simulation_analysis_data.csv")
    
    print("\n" + "="*60)
    print("Analysis completed!")
    print("="*60)


if __name__ == '__main__':
    main()
