#!/usr/bin/env python3
"""
Compare performance of base version and V2 version models
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


class ModelComparator:
    """Model comparator"""
    
    def __init__(self):
        self.dt = 0.1
        self.predictor_v1 = WalkerActionPredictor('../assets/walker_speed_predictor_new.pth')
        self.predictor_v2 = WalkerActionPredictor('../assets/walker_speed_predictor_v2_fixed.pth')
    
    def simulate_with_model(self, predictor, car_speed, walker_y_init, max_steps=200):
        """Simulate trajectory using specified model"""
        # 初始状态
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
                'time': step * self.dt,
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
            walker_x += actual_vx * self.dt
            walker_y += actual_vy * self.dt
            car_x += car_speed * self.dt
            
            # 更新速度（使用实际速度）
            walker_vx = actual_vx
            walker_vy = actual_vy
            
            # 检查终止条件
            if car_x > C.CAR_RIGHT_LIMIT or walker_y >= C.WALKER_DESTINATION_Y:
                break
        
        return pd.DataFrame(data)
    
    def compare_models(self, car_speed, walker_y_init):
        """比较两个模型"""
        print(f"Comparing models with Car Speed: {car_speed:.2f} m/s, Walker Y Init: {walker_y_init:.2f} m")
        
        # 使用V1模型模拟
        print("Simulating with V1 model...")
        df_v1 = self.simulate_with_model(self.predictor_v1, car_speed, walker_y_init)
        
        # 使用V2模型模拟
        print("Simulating with V2 model...")
        df_v2 = self.simulate_with_model(self.predictor_v2, car_speed, walker_y_init)
        
        return df_v1, df_v2
    
    def calculate_metrics(self, df):
        """计算性能指标"""
        metrics = {
            'vx_mae': df['vx_error_abs'].mean(),
            'vy_mae': df['vy_error_abs'].mean(),
            'vx_rmse': np.sqrt((df['vx_error']**2).mean()),
            'vy_rmse': np.sqrt((df['vy_error']**2).mean()),
            'vx_max_error': df['vx_error_abs'].max(),
            'vy_max_error': df['vy_error_abs'].max(),
            'total_velocity_mae': np.mean(np.sqrt(df['vx_error']**2 + df['vy_error']**2)),
            'total_velocity_rmse': np.sqrt(np.mean(df['vx_error']**2 + df['vy_error']**2)),
            'simulation_duration': df['time'].iloc[-1],
            'total_steps': len(df)
        }
        return metrics
    
    def plot_comparison(self, df_v1, df_v2, car_speed, walker_y_init):
        """绘制对比图"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # 1. 轨迹对比
        ax1 = axes[0, 0]
        ax1.plot(df_v1['car_x'], df_v1['car_y'], 'b-', linewidth=3, label='Car', alpha=0.8)
        ax1.plot(df_v1['walker_x'], df_v1['walker_y'], 'r-', linewidth=2, label='Walker (V1)', alpha=0.8)
        ax1.plot(df_v2['walker_x'], df_v2['walker_y'], 'g--', linewidth=2, label='Walker (V2)', alpha=0.8)
        ax1.scatter(df_v1['car_x'].iloc[0], df_v1['car_y'].iloc[0], c='blue', s=100, marker='o', label='Start')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'Trajectory Comparison\nCar Speed: {car_speed:.1f} m/s, Walker Y Init: {walker_y_init:.1f} m')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. 速度对比 - X方向
        ax2 = axes[0, 1]
        ax2.plot(df_v1['time'], df_v1['actual_vx'], 'k-', linewidth=2, label='Actual vx', alpha=0.8)
        ax2.plot(df_v1['time'], df_v1['predicted_vx'], 'r-', linewidth=2, label='V1 Predicted vx', alpha=0.8)
        ax2.plot(df_v2['time'], df_v2['predicted_vx'], 'g--', linewidth=2, label='V2 Predicted vx', alpha=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity X (m/s)')
        ax2.set_title('Velocity X Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 速度对比 - Y方向
        ax3 = axes[1, 0]
        ax3.plot(df_v1['time'], df_v1['actual_vy'], 'k-', linewidth=2, label='Actual vy', alpha=0.8)
        ax3.plot(df_v1['time'], df_v1['predicted_vy'], 'r-', linewidth=2, label='V1 Predicted vy', alpha=0.8)
        ax3.plot(df_v2['time'], df_v2['predicted_vy'], 'g--', linewidth=2, label='V2 Predicted vy', alpha=0.8)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity Y (m/s)')
        ax3.set_title('Velocity Y Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 误差对比 - X方向
        ax4 = axes[1, 1]
        ax4.plot(df_v1['time'], df_v1['vx_error_abs'], 'r-', linewidth=2, label='V1 |vx error|', alpha=0.8)
        ax4.plot(df_v2['time'], df_v2['vx_error_abs'], 'g--', linewidth=2, label='V2 |vx error|', alpha=0.8)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Absolute Error (m/s)')
        ax4.set_title('Velocity X Error Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 误差对比 - Y方向
        ax5 = axes[2, 0]
        ax5.plot(df_v1['time'], df_v1['vy_error_abs'], 'r-', linewidth=2, label='V1 |vy error|', alpha=0.8)
        ax5.plot(df_v2['time'], df_v2['vy_error_abs'], 'g--', linewidth=2, label='V2 |vy error|', alpha=0.8)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Absolute Error (m/s)')
        ax5.set_title('Velocity Y Error Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 累积误差对比
        ax6 = axes[2, 1]
        cumulative_v1 = np.cumsum(np.sqrt(df_v1['vx_error']**2 + df_v1['vy_error']**2))
        cumulative_v2 = np.cumsum(np.sqrt(df_v2['vx_error']**2 + df_v2['vy_error']**2))
        ax6.plot(df_v1['time'], cumulative_v1, 'r-', linewidth=2, label='V1 Cumulative Error', alpha=0.8)
        ax6.plot(df_v2['time'], cumulative_v2, 'g--', linewidth=2, label='V2 Cumulative Error', alpha=0.8)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Cumulative Error (m/s)')
        ax6.set_title('Cumulative Error Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_comparison_table(self, metrics_v1, metrics_v2):
        """打印对比表格"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        print(f"{'Metric':<25} {'V1 (Basic)':<15} {'V2 (ResNet+Attn)':<20} {'Improvement':<15}")
        print("-"*80)
        
        # 速度误差对比
        v1_vx_mae, v2_vx_mae = metrics_v1['vx_mae'], metrics_v2['vx_mae']
        v1_vy_mae, v2_vy_mae = metrics_v1['vy_mae'], metrics_v2['vy_mae']
        v1_total_mae, v2_total_mae = metrics_v1['total_velocity_mae'], metrics_v2['total_velocity_mae']
        
        print(f"{'vx MAE (m/s)':<25} {v1_vx_mae:<15.4f} {v2_vx_mae:<20.4f} {((v1_vx_mae-v2_vx_mae)/v1_vx_mae*100):<15.2f}%")
        print(f"{'vy MAE (m/s)':<25} {v1_vy_mae:<15.4f} {v2_vy_mae:<20.4f} {((v1_vy_mae-v2_vy_mae)/v1_vy_mae*100):<15.2f}%")
        print(f"{'Total Velocity MAE':<25} {v1_total_mae:<15.4f} {v2_total_mae:<20.4f} {((v1_total_mae-v2_total_mae)/v1_total_mae*100):<15.2f}%")
        
        # RMSE对比
        v1_vx_rmse, v2_vx_rmse = metrics_v1['vx_rmse'], metrics_v2['vx_rmse']
        v1_vy_rmse, v2_vy_rmse = metrics_v1['vy_rmse'], metrics_v2['vy_rmse']
        v1_total_rmse, v2_total_rmse = metrics_v1['total_velocity_rmse'], metrics_v2['total_velocity_rmse']
        
        print(f"{'vx RMSE (m/s)':<25} {v1_vx_rmse:<15.4f} {v2_vx_rmse:<20.4f} {((v1_vx_rmse-v2_vx_rmse)/v1_vx_rmse*100):<15.2f}%")
        print(f"{'vy RMSE (m/s)':<25} {v1_vy_rmse:<15.4f} {v2_vy_rmse:<20.4f} {((v1_vy_rmse-v2_vy_rmse)/v1_vy_rmse*100):<15.2f}%")
        print(f"{'Total Velocity RMSE':<25} {v1_total_rmse:<15.4f} {v2_total_rmse:<20.4f} {((v1_total_rmse-v2_total_rmse)/v1_total_rmse*100):<15.2f}%")
        
        # 最大误差对比
        v1_vx_max, v2_vx_max = metrics_v1['vx_max_error'], metrics_v2['vx_max_error']
        v1_vy_max, v2_vy_max = metrics_v1['vy_max_error'], metrics_v2['vy_max_error']
        
        print(f"{'vx Max Error (m/s)':<25} {v1_vx_max:<15.4f} {v2_vx_max:<20.4f} {((v1_vx_max-v2_vx_max)/v1_vx_max*100):<15.2f}%")
        print(f"{'vy Max Error (m/s)':<25} {v1_vy_max:<15.4f} {v2_vy_max:<20.4f} {((v1_vy_max-v2_vy_max)/v1_vy_max*100):<15.2f}%")
        
        print("-"*80)
        
        # 总结
        better_vx = "V2" if v2_vx_mae < v1_vx_mae else "V1"
        better_vy = "V2" if v2_vy_mae < v1_vy_mae else "V1"
        better_total = "V2" if v2_total_mae < v1_total_mae else "V1"
        
        print(f"Summary:")
        print(f"  X-direction: {better_vx} performs better")
        print(f"  Y-direction: {better_vy} performs better")
        print(f"  Overall: {better_total} performs better")
        
        print("="*80)


def main():
    """主函数"""
    print("=" * 60)
    print("MODEL COMPARISON: V1 vs V2")
    print("=" * 60)
    
    # 创建比较器
    comparator = ModelComparator()
    
    # 使用相同的参数进行比较
    np.random.seed(42)
    car_speed = 7.62
    walker_y_init = 7.70
    
    print(f"Comparison Parameters:")
    print(f"  Car Speed: {car_speed:.2f} m/s")
    print(f"  Walker Initial Y: {walker_y_init:.2f} m")
    print(f"  Time Step: 0.1 s")
    
    # 比较模型
    df_v1, df_v2 = comparator.compare_models(car_speed, walker_y_init)
    
    # 计算指标
    print("Calculating performance metrics...")
    metrics_v1 = comparator.calculate_metrics(df_v1)
    metrics_v2 = comparator.calculate_metrics(df_v2)
    
    # 打印对比表格
    comparator.print_comparison_table(metrics_v1, metrics_v2)
    
    # 绘制对比图
    print("\nGenerating comparison plot...")
    fig = comparator.plot_comparison(df_v1, df_v2, car_speed, walker_y_init)
    fig.savefig('../logs/model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: ../logs/model_comparison.png")
    
    # 保存数据
    df_v1.to_csv('../logs/model_v1_simulation_data.csv', index=False)
    df_v2.to_csv('../logs/model_v2_simulation_data.csv', index=False)
    print("Saved: ../logs/model_v1_simulation_data.csv, ../logs/model_v2_simulation_data.csv")
    
    print("\n" + "="*60)
    print("Comparison completed!")
    print("="*60)


if __name__ == '__main__':
    main()
