#!/usr/bin/env python3
"""
测试脚本：对比 0.75 和 0.85 在不同情景下的 eta 大小
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_eta_comparison_75_85():
    """对比 0.75 和 0.85 在不同情景下的 eta 大小"""
    print("对比 0.75 和 0.85 在不同情景下的 eta 大小...")
    
    try:
        from models.conformal_grid import get_eta
        
        # 定义不同的测试情景
        scenarios = [
            {
                'name': '低速接近行人',
                'params': {'car_x': 0.0, 'car_v': 3.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': 0.0, 'walker_vy': 0.0}
            },
            {
                'name': '中速接近行人',
                'params': {'car_x': 0.0, 'car_v': 8.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': 0.0, 'walker_vy': 0.0}
            },
            {
                'name': '高速接近行人',
                'params': {'car_x': 0.0, 'car_v': 12.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': 0.0, 'walker_vy': 0.0}
            },
            {
                'name': '行人静止，车辆在左侧',
                'params': {'car_x': -5.0, 'car_v': 5.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': 0.0, 'walker_vy': 0.0}
            },
            {
                'name': '行人静止，车辆在右侧',
                'params': {'car_x': 5.0, 'car_v': 5.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': 0.0, 'walker_vy': 0.0}
            },
            {
                'name': '行人向前移动',
                'params': {'car_x': 0.0, 'car_v': 5.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': 1.0, 'walker_vy': 0.0}
            },
            {
                'name': '行人向后移动',
                'params': {'car_x': 0.0, 'car_v': 5.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': -1.0, 'walker_vy': 0.0}
            },
            {
                'name': '行人向左移动',
                'params': {'car_x': 0.0, 'car_v': 5.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': 0.0, 'walker_vy': 1.0}
            },
            {
                'name': '行人向右移动',
                'params': {'car_x': 0.0, 'car_v': 5.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': 0.0, 'walker_vy': -1.0}
            },
            {
                'name': '行人斜向移动',
                'params': {'car_x': 0.0, 'car_v': 5.0, 'walker_x': 0.0, 'walker_y': 0.0, 'walker_vx': 0.5, 'walker_vy': 0.5}
            },
            {
                'name': '行人位置较远',
                'params': {'car_x': 0.0, 'car_v': 5.0, 'walker_x': 0.0, 'walker_y': 5.0, 'walker_vx': 0.0, 'walker_vy': 0.0}
            },
            {
                'name': '行人位置很远',
                'params': {'car_x': 0.0, 'car_v': 5.0, 'walker_x': 0.0, 'walker_y': 10.0, 'walker_vx': 0.0, 'walker_vy': 0.0}
            }
        ]
        
        alpha_values = [0.75, 0.85]
        results = {}
        
        print("\n" + "="*100)
        print("ETA 值对比结果")
        print("="*100)
        print(f"{'情景':<20} {'Alpha':<8} {'Eta_X':<12} {'Eta_Y':<12} {'Eta_Total':<12} {'安全距离调整':<15}")
        print("-"*100)
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            params = scenario['params']
            results[scenario_name] = {}
            
            for alpha in alpha_values:
                try:
                    eta_x, eta_y = get_eta(**params, cp_alpha=alpha)
                    eta_total = np.sqrt(eta_x**2 + eta_y**2)
                    d_safe_adjusted = 2.5 + eta_total  # 假设基础安全距离为2.5
                    
                    results[scenario_name][alpha] = {
                        'eta_x': eta_x,
                        'eta_y': eta_y,
                        'eta_total': eta_total,
                        'd_safe_adjusted': d_safe_adjusted
                    }
                    
                    print(f"{scenario_name:<20} {alpha:<8} {eta_x:<12.6f} {eta_y:<12.6f} {eta_total:<12.6f} {d_safe_adjusted:<15.6f}")
                    
                except Exception as e:
                    print(f"{scenario_name:<20} {alpha:<8} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<15}")
                    results[scenario_name][alpha] = None
        
        print("-"*100)
        
        # 分析差异
        print("\n" + "="*100)
        print("差异分析")
        print("="*100)
        print(f"{'情景':<20} {'Eta_X差异':<15} {'Eta_Y差异':<15} {'Eta_Total差异':<15} {'安全距离差异':<15}")
        print("-"*100)
        
        for scenario_name, scenario_results in results.items():
            if scenario_results[0.75] and scenario_results[0.85]:
                eta_75 = scenario_results[0.75]
                eta_85 = scenario_results[0.85]
                
                diff_x = eta_85['eta_x'] - eta_75['eta_x']
                diff_y = eta_85['eta_y'] - eta_75['eta_y']
                diff_total = eta_85['eta_total'] - eta_75['eta_total']
                diff_safe = eta_85['d_safe_adjusted'] - eta_75['d_safe_adjusted']
                
                print(f"{scenario_name:<20} {diff_x:<15.6f} {diff_y:<15.6f} {diff_total:<15.6f} {diff_safe:<15.6f}")
            else:
                print(f"{scenario_name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        
        print("-"*100)
        
        # 统计总结
        print("\n" + "="*100)
        print("统计总结")
        print("="*100)
        
        valid_scenarios = [name for name, results in results.items() 
                          if results[0.75] and results[0.85]]
        
        if valid_scenarios:
            # 计算平均差异
            avg_diff_x = np.mean([results[name][0.85]['eta_x'] - results[name][0.75]['eta_x'] 
                                 for name in valid_scenarios])
            avg_diff_y = np.mean([results[name][0.85]['eta_y'] - results[name][0.75]['eta_y'] 
                                 for name in valid_scenarios])
            avg_diff_total = np.mean([results[name][0.85]['eta_total'] - results[name][0.75]['eta_total'] 
                                     for name in valid_scenarios])
            avg_diff_safe = np.mean([results[name][0.85]['d_safe_adjusted'] - results[name][0.75]['d_safe_adjusted'] 
                                    for name in valid_scenarios])
            
            print(f"有效情景数量: {len(valid_scenarios)}")
            print(f"平均 Eta_X 差异: {avg_diff_x:.6f}")
            print(f"平均 Eta_Y 差异: {avg_diff_y:.6f}")
            print(f"平均 Eta_Total 差异: {avg_diff_total:.6f}")
            print(f"平均安全距离差异: {avg_diff_safe:.6f}")
            
            # 计算差异百分比
            avg_eta_75 = np.mean([results[name][0.75]['eta_total'] for name in valid_scenarios])
            avg_eta_85 = np.mean([results[name][0.85]['eta_total'] for name in valid_scenarios])
            percent_diff = ((avg_eta_85 - avg_eta_75) / avg_eta_75) * 100
            
            print(f"平均 Eta_Total 相对差异: {percent_diff:.2f}%")
            
            print("\n结论:")
            if avg_diff_total > 0:
                print("✅ Alpha=0.85 的 eta 值普遍大于 Alpha=0.75，符合预期")
                print("   更高的置信度需要更大的预测区间")
            else:
                print("❌ Alpha=0.85 的 eta 值小于 Alpha=0.75，可能有问题")
        
        else:
            print("❌ 没有有效的情景数据进行分析")
        
    except ImportError as e:
        print(f"导入错误: {e}")

if __name__ == "__main__":
    test_eta_comparison_75_85()
