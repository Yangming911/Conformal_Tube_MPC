#!/usr/bin/env python3
"""
可视化模型性能：比较实际轨迹和模型预测轨迹
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch

# 添加项目根目录到路径
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.predictor import WalkerActionPredictor
from envs.dynamics_social_force import walker_logic_SF
import utils.constants as C


class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self, model_path='../assets/walker_speed_predictor_new.pth'):
        self.predictor = WalkerActionPredictor(model_path)
        self.dt = 0.1
        
    def simulate_trajectory(self, car_speed, walker_y_init, max_steps=200):
        """模拟轨迹"""
        # 初始状态
        car_x = C.CAR_START_X
        car_y = C.CAR_LANE_Y
        walker_x = C.WALKER_START_X
        walker_y = walker_y_init
        walker_vx = C.WALKER_START_V_X
        walker_vy = C.WALKER_START_V_Y
        
        # 存储轨迹数据
        car_trajectory = []
        walker_actual_trajectory = []
        walker_predicted_trajectory = []
        walker_actual_velocities = []
        walker_predicted_velocities = []
        
        for step in range(max_steps):
            # 记录当前位置
            car_trajectory.append([car_x, car_y])
            walker_actual_trajectory.append([walker_x, walker_y])
            walker_predicted_trajectory.append([walker_x, walker_y])  # 初始位置相同
            
            # 实际下一步速度（来自social force模型）
            actual_vx, actual_vy = walker_logic_SF(
                car_speed, car_x, car_y, walker_x, walker_y, walker_vx, walker_vy,
                v_max=C.v_max, a_max=C.a_max, destination_y=C.WALKER_DESTINATION_Y
            )
            
            # 模型预测的下一步速度
            predicted_vx, predicted_vy = self.predictor.predict(
                car_x, car_y, car_speed, walker_x, walker_y, walker_vx, walker_vy
            )
            
            # 记录速度
            walker_actual_velocities.append([actual_vx, actual_vy])
            walker_predicted_velocities.append([predicted_vx, predicted_vy])
            
            # 更新位置（使用实际速度）
            walker_x += actual_vx * self.dt
            walker_y += actual_vy * self.dt
            car_x += car_speed * self.dt
            
            # 更新速度（使用实际速度）
            walker_vx = actual_vx
            walker_vy = actual_vy
            
            # 检查终止条件
            if car_x > C.CAR_RIGHT_LIMIT or walker_y >= C.WALKER_DESTINATION_Y:
                break
        
        return {
            'car_trajectory': np.array(car_trajectory),
            'walker_actual_trajectory': np.array(walker_actual_trajectory),
            'walker_predicted_trajectory': np.array(walker_predicted_trajectory),
            'walker_actual_velocities': np.array(walker_actual_velocities),
            'walker_predicted_velocities': np.array(walker_predicted_velocities),
            'steps': len(car_trajectory)
        }
    
    def simulate_predicted_trajectory(self, car_speed, walker_y_init, max_steps=200):
        """使用模型预测的速度模拟完整轨迹"""
        # 初始状态
        car_x = C.CAR_START_X
        car_y = C.CAR_LANE_Y
        walker_x = C.WALKER_START_X
        walker_y = walker_y_init
        walker_vx = C.WALKER_START_V_X
        walker_vy = C.WALKER_START_V_Y
        
        # 存储轨迹数据
        car_trajectory = []
        walker_predicted_trajectory = []
        walker_predicted_velocities = []
        
        for step in range(max_steps):
            # 记录当前位置
            car_trajectory.append([car_x, car_y])
            walker_predicted_trajectory.append([walker_x, walker_y])
            
            # 模型预测的下一步速度
            predicted_vx, predicted_vy = self.predictor.predict(
                car_x, car_y, car_speed, walker_x, walker_y, walker_vx, walker_vy
            )
            
            # 记录速度
            walker_predicted_velocities.append([predicted_vx, predicted_vy])
            
            # 更新位置（使用预测速度）
            walker_x += predicted_vx * self.dt
            walker_y += predicted_vy * self.dt
            car_x += car_speed * self.dt
            
            # 更新速度（使用预测速度）
            walker_vx = predicted_vx
            walker_vy = predicted_vy
            
            # 检查终止条件
            if car_x > C.CAR_RIGHT_LIMIT or walker_y >= C.WALKER_DESTINATION_Y:
                break
        
        return {
            'car_trajectory': np.array(car_trajectory),
            'walker_predicted_trajectory': np.array(walker_predicted_trajectory),
            'walker_predicted_velocities': np.array(walker_predicted_velocities),
            'steps': len(car_trajectory)
        }
    
    def plot_trajectories(self, actual_data, predicted_data, car_speed, walker_y_init):
        """绘制轨迹对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 轨迹对比图
        ax1 = axes[0, 0]
        ax1.plot(actual_data['car_trajectory'][:, 0], actual_data['car_trajectory'][:, 1], 
                'b-', linewidth=3, label='Car Trajectory', alpha=0.8)
        ax1.plot(actual_data['walker_actual_trajectory'][:, 0], actual_data['walker_actual_trajectory'][:, 1], 
                'r-', linewidth=2, label='Walker Actual', alpha=0.8)
        ax1.plot(predicted_data['walker_predicted_trajectory'][:, 0], predicted_data['walker_predicted_trajectory'][:, 1], 
                'g--', linewidth=2, label='Walker Predicted', alpha=0.8)
        
        # 标记起点和终点
        ax1.scatter(actual_data['car_trajectory'][0, 0], actual_data['car_trajectory'][0, 1], 
                   c='blue', s=100, marker='o', label='Car Start')
        ax1.scatter(actual_data['walker_actual_trajectory'][0, 0], actual_data['walker_actual_trajectory'][0, 1], 
                   c='red', s=100, marker='o', label='Walker Start')
        ax1.scatter(predicted_data['walker_predicted_trajectory'][-1, 0], predicted_data['walker_predicted_trajectory'][-1, 1], 
                   c='green', s=100, marker='s', label='Predicted End')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'Trajectory Comparison\nCar Speed: {car_speed:.1f} m/s, Walker Y Init: {walker_y_init:.1f} m')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. 速度对比图 - X方向
        ax2 = axes[0, 1]
        time_steps = np.arange(len(actual_data['walker_actual_velocities'])) * self.dt
        ax2.plot(time_steps, actual_data['walker_actual_velocities'][:, 0], 
                'r-', linewidth=2, label='Actual vx', alpha=0.8)
        ax2.plot(time_steps, actual_data['walker_predicted_velocities'][:, 0], 
                'g--', linewidth=2, label='Predicted vx', alpha=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity X (m/s)')
        ax2.set_title('Velocity X Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 速度对比图 - Y方向
        ax3 = axes[1, 0]
        ax3.plot(time_steps, actual_data['walker_actual_velocities'][:, 1], 
                'r-', linewidth=2, label='Actual vy', alpha=0.8)
        ax3.plot(time_steps, actual_data['walker_predicted_velocities'][:, 1], 
                'g--', linewidth=2, label='Predicted vy', alpha=0.8)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity Y (m/s)')
        ax3.set_title('Velocity Y Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 误差分析
        ax4 = axes[1, 1]
        vx_error = np.abs(actual_data['walker_actual_velocities'][:, 0] - actual_data['walker_predicted_velocities'][:, 0])
        vy_error = np.abs(actual_data['walker_actual_velocities'][:, 1] - actual_data['walker_predicted_velocities'][:, 1])
        
        ax4.plot(time_steps, vx_error, 'b-', linewidth=2, label='|vx error|', alpha=0.8)
        ax4.plot(time_steps, vy_error, 'orange', linewidth=2, label='|vy error|', alpha=0.8)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Absolute Error (m/s)')
        ax4.set_title('Prediction Error')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_velocity_vectors(self, actual_data, predicted_data, car_speed, walker_y_init, step_interval=5):
        """绘制速度向量图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 实际轨迹和速度向量
        ax1.plot(actual_data['walker_actual_trajectory'][:, 0], actual_data['walker_actual_trajectory'][:, 1], 
                'r-', linewidth=2, alpha=0.6, label='Actual Trajectory')
        
        for i in range(0, len(actual_data['walker_actual_trajectory']), step_interval):
            x = actual_data['walker_actual_trajectory'][i, 0]
            y = actual_data['walker_actual_trajectory'][i, 1]
            vx = actual_data['walker_actual_velocities'][i, 0]
            vy = actual_data['walker_actual_velocities'][i, 1]
            ax1.arrow(x, y, vx*0.5, vy*0.5, head_width=0.3, head_length=0.2, 
                     fc='red', ec='red', alpha=0.7)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Actual Trajectory with Velocity Vectors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 预测轨迹和速度向量
        ax2.plot(predicted_data['walker_predicted_trajectory'][:, 0], predicted_data['walker_predicted_trajectory'][:, 1], 
                'g-', linewidth=2, alpha=0.6, label='Predicted Trajectory')
        
        for i in range(0, len(predicted_data['walker_predicted_trajectory']), step_interval):
            x = predicted_data['walker_predicted_trajectory'][i, 0]
            y = predicted_data['walker_predicted_trajectory'][i, 1]
            vx = predicted_data['walker_predicted_velocities'][i, 0]
            vy = predicted_data['walker_predicted_velocities'][i, 1]
            ax2.arrow(x, y, vx*0.5, vy*0.5, head_width=0.3, head_length=0.2, 
                     fc='green', ec='green', alpha=0.7)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('Predicted Trajectory with Velocity Vectors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        return fig
    
    def calculate_metrics(self, actual_data, predicted_data):
        """计算性能指标"""
        # 速度误差
        vx_error = actual_data['walker_actual_velocities'][:, 0] - actual_data['walker_predicted_velocities'][:, 0]
        vy_error = actual_data['walker_actual_velocities'][:, 1] - actual_data['walker_predicted_velocities'][:, 1]
        
        # 轨迹误差（位置误差）
        min_len = min(len(actual_data['walker_actual_trajectory']), len(predicted_data['walker_predicted_trajectory']))
        pos_error_x = actual_data['walker_actual_trajectory'][:min_len, 0] - predicted_data['walker_predicted_trajectory'][:min_len, 0]
        pos_error_y = actual_data['walker_actual_trajectory'][:min_len, 1] - predicted_data['walker_predicted_trajectory'][:min_len, 1]
        
        metrics = {
            'vx_mae': np.mean(np.abs(vx_error)),
            'vy_mae': np.mean(np.abs(vy_error)),
            'vx_rmse': np.sqrt(np.mean(vx_error**2)),
            'vy_rmse': np.sqrt(np.mean(vy_error**2)),
            'pos_x_mae': np.mean(np.abs(pos_error_x)),
            'pos_y_mae': np.mean(np.abs(pos_error_y)),
            'pos_x_rmse': np.sqrt(np.mean(pos_error_x**2)),
            'pos_y_rmse': np.sqrt(np.mean(pos_error_y**2)),
            'total_velocity_mae': np.mean(np.sqrt(vx_error**2 + vy_error**2)),
            'total_position_mae': np.mean(np.sqrt(pos_error_x**2 + pos_error_y**2))
        }
        
        return metrics


def main():
    """主函数"""
    print("=" * 60)
    print("Walker Speed Predictor Model Performance Visualization")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = TrajectoryVisualizer()
    
    # 随机选择参数
    np.random.seed(42)  # 固定随机种子以便复现
    car_speed = np.random.uniform(5.0, 12.0)
    walker_y_init = np.random.uniform(2.0, 8.0)
    
    print(f"Simulation Parameters:")
    print(f"  Car Speed: {car_speed:.2f} m/s")
    print(f"  Walker Initial Y: {walker_y_init:.2f} m")
    print(f"  Time Step: {visualizer.dt} s")
    
    # 模拟实际轨迹（使用social force模型）
    print("\nSimulating actual trajectory...")
    actual_data = visualizer.simulate_trajectory(car_speed, walker_y_init)
    
    # 模拟预测轨迹（使用神经网络模型）
    print("Simulating predicted trajectory...")
    predicted_data = visualizer.simulate_predicted_trajectory(car_speed, walker_y_init)
    
    # 计算性能指标
    print("Calculating performance metrics...")
    metrics = visualizer.calculate_metrics(actual_data, predicted_data)
    
    # 打印性能指标
    print("\nPerformance Metrics:")
    print(f"  Velocity X - MAE: {metrics['vx_mae']:.4f} m/s, RMSE: {metrics['vx_rmse']:.4f} m/s")
    print(f"  Velocity Y - MAE: {metrics['vy_mae']:.4f} m/s, RMSE: {metrics['vy_rmse']:.4f} m/s")
    print(f"  Position X - MAE: {metrics['pos_x_mae']:.4f} m, RMSE: {metrics['pos_x_rmse']:.4f} m")
    print(f"  Position Y - MAE: {metrics['pos_y_mae']:.4f} m, RMSE: {metrics['pos_y_rmse']:.4f} m")
    print(f"  Total Velocity MAE: {metrics['total_velocity_mae']:.4f} m/s")
    print(f"  Total Position MAE: {metrics['total_position_mae']:.4f} m")
    
    # 绘制轨迹对比图
    print("\nGenerating trajectory comparison plot...")
    fig1 = visualizer.plot_trajectories(actual_data, predicted_data, car_speed, walker_y_init)
    fig1.savefig('../logs/model_performance_trajectories.png', dpi=300, bbox_inches='tight')
    print("Saved: ../logs/model_performance_trajectories.png")
    
    # 绘制速度向量图
    print("Generating velocity vectors plot...")
    fig2 = visualizer.plot_velocity_vectors(actual_data, predicted_data, car_speed, walker_y_init)
    fig2.savefig('../logs/model_performance_vectors.png', dpi=300, bbox_inches='tight')
    print("Saved: ../logs/model_performance_vectors.png")
    
    # 显示图表
    plt.show()
    
    print("\n" + "=" * 60)
    print("Visualization completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
