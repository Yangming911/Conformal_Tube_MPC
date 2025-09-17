#!/usr/bin/env python3
"""
MPC控制器批量测试评估脚本
使用新的模拟器进行测试
"""

import numpy as np
import sys
from pathlib import Path
import time
import argparse
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.simulator import _initial_state, _step, _is_collision, _done
from mpc.tubempc_controller import mpc_control
from mpc.vanillampc_controller import mpc_control_vanilla
import utils.constants as C

# 设置随机种子
np.random.seed(42)

# 测试参数
DEFAULT_SAMPLE_NUM = 1000
DEFAULT_MAX_STEPS = 200
DEFAULT_D_SAFE = 0.5
DEFAULT_T = 10

def simulate_single_episode(state, controller_func, controller_name, T=DEFAULT_T, d_safe=DEFAULT_D_SAFE, max_steps=DEFAULT_MAX_STEPS):
    """
    模拟单个episode
    
    Args:
        state: 初始状态字典
        controller_func: 控制器函数
        controller_name: 控制器名称
        T: 预测步长
        d_safe: 安全距离
        max_steps: 最大步数
    
    Returns:
        tuple: (collision_occurred, steps_taken, average_speed, average_calc_time)
    """
    current_state = state.copy()
    total_speed = 0.0
    total_calc_time = 0.0
    steps_taken = 0
    
    for step in range(max_steps):
        if _done(current_state):
            break
            
        # 检查碰撞
        if _is_collision(current_state):
            return True, steps_taken, total_speed / max(1, steps_taken), total_calc_time / max(1, steps_taken)
        
        # 计算控制输入
        start_time = time.perf_counter()
        
        if controller_name == "constant_speed":
            # 匀速通过，固定15m/s
            control_input = 15.0
        elif controller_name == "tubempc":
            # 使用Tube MPC控制器
            control_input = mpc_control(
                current_state["car_x"], 
                np.array([current_state["walker_x"], current_state["walker_y"]]), 
                T, 
                d_safe
            )
        elif controller_name == "vanillampc":
            # 使用Vanilla MPC控制器
            control_input = mpc_control_vanilla(
                current_state["car_x"], 
                np.array([current_state["walker_x"], current_state["walker_y"]]), 
                T, 
                d_safe
            )
        else:
            # 使用其他控制器
            control_input = controller_func(current_state, T=T, d_safe=d_safe)
        
        end_time = time.perf_counter()
        calc_time = end_time - start_time
        
        # 更新状态
        current_state["car_v"] = control_input
        next_state, _ = _step(current_state)
        current_state = next_state
        
        # 记录统计信息
        total_speed += control_input
        total_calc_time += calc_time
        steps_taken += 1
    
    # 最终检查碰撞
    collision_occurred = _is_collision(current_state)
    
    return collision_occurred, steps_taken, total_speed / max(1, steps_taken), total_calc_time / max(1, steps_taken)

def batch_evaluate_controller(controller_func, controller_name, sample_num=DEFAULT_SAMPLE_NUM, 
                            T=DEFAULT_T, d_safe=DEFAULT_D_SAFE, max_steps=DEFAULT_MAX_STEPS):
    """
    批量评估单个控制器
    
    Args:
        controller_func: 控制器函数
        controller_name: 控制器名称
        sample_num: 样本数量
        T: 预测步长
        d_safe: 安全距离
        max_steps: 最大步数
    
    Returns:
        dict: 评估结果
    """
    collision_count = 0
    total_steps = 0
    total_speed = 0.0
    total_calc_time = 0.0
    
    print(f"Evaluating {controller_name}...")
    
    for i in tqdm(range(sample_num), desc=f"{controller_name}"):
        # 生成随机初始状态，使用固定的种子确保可重复性
        rng = np.random.RandomState(42 + i)  # 为每个样本使用不同的但固定的种子
        car_speed = float(rng.uniform(1.0, 15.0))
        initial_state = _initial_state(car_speed, rng)
        
        # 模拟单个episode
        collision_occurred, steps_taken, avg_speed, avg_calc_time = simulate_single_episode(
            initial_state, controller_func, controller_name, T, d_safe, max_steps
        )

        if collision_occurred:
            collision_count += 1

        total_steps += steps_taken
        total_speed += avg_speed
        total_calc_time += avg_calc_time
    
    # 计算统计结果
    collision_probability = collision_count / sample_num
    average_steps = total_steps / sample_num
    average_speed = total_speed / sample_num
    average_calc_time = total_calc_time / sample_num
    
    return {
        "controller": controller_name,
        "collision_probability": collision_probability,
        "collision_count": collision_count,
        "sample_num": sample_num,
        "average_steps": average_steps,
        "average_speed": average_speed,
        "average_calc_time": average_calc_time,
        "T": T,
        "d_safe": d_safe
    }

def compare_controllers(sample_num=DEFAULT_SAMPLE_NUM, T=DEFAULT_T, d_safe=DEFAULT_D_SAFE, max_steps=DEFAULT_MAX_STEPS):
    """
    对比所有控制器的性能
    
    Args:
        sample_num: 样本数量
        T: 预测步长
        d_safe: 安全距离
        max_steps: 最大步数
    
    Returns:
        list: 所有控制器的评估结果
    """
    print("=" * 80)
    print("MPC Controllers Comparison")
    print("=" * 80)
    print(f"Sample number: {sample_num}")
    print(f"Prediction horizon T: {T}")
    print(f"Safety distance: {d_safe}")
    print(f"Max steps per episode: {max_steps}")
    print("=" * 80)
    
    # 定义控制器
    controllers = [
        (lambda state, **kwargs: 15.0, "constant_speed"),
        (None, "tubempc"),  # 特殊处理
        (None, "vanillampc"),  # 特殊处理
    ]
    
    results = []
    
    for controller_func, controller_name in controllers:
        result = batch_evaluate_controller(
            controller_func, controller_name, sample_num, T, d_safe, max_steps
        )
        results.append(result)
        
        # 打印单个控制器结果
        print(f"\n{controller_name.upper()}:")
        print(f"  Collision Probability: {result['collision_probability']*100:.2f}%")
        print(f"  Collision Count: {result['collision_count']}/{result['sample_num']}")
        print(f"  Average Steps: {result['average_steps']:.1f}")
        print(f"  Average Speed: {result['average_speed']:.2f} m/s")
        print(f"  Average Calc Time: {result['average_calc_time']*1000:.3f} ms")
    
    return results

def save_results_to_file(results, filename="logs/mpc_comparison_results.log"):
    """
    保存结果到文件
    
    Args:
        results: 评估结果列表
        filename: 输出文件名
    """
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("MPC Controllers Comparison Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Number: {results[0]['sample_num']}\n")
        f.write(f"Prediction Horizon T: {results[0]['T']}\n")
        f.write(f"Safety Distance: {results[0]['d_safe']}\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"{result['controller'].upper()}:\n")
            f.write(f"  Collision Probability: {result['collision_probability']*100:.2f}%\n")
            f.write(f"  Collision Count: {result['collision_count']}/{result['sample_num']}\n")
            f.write(f"  Average Steps: {result['average_steps']:.1f}\n")
            f.write(f"  Average Speed: {result['average_speed']:.2f} m/s\n")
            f.write(f"  Average Calc Time: {result['average_calc_time']*1000:.3f} ms\n")
            f.write("-" * 40 + "\n")
        
        # 添加对比分析
        f.write("\nCOMPARISON ANALYSIS:\n")
        f.write("=" * 40 + "\n")
        
        # 找出最佳控制器
        best_safety = min(results, key=lambda x: x['collision_probability'])
        best_speed = max(results, key=lambda x: x['average_speed'])
        best_efficiency = min(results, key=lambda x: x['average_calc_time'])
        
        f.write(f"Best Safety (Lowest Collision Rate): {best_safety['controller']} ({best_safety['collision_probability']*100:.2f}%)\n")
        f.write(f"Best Speed (Highest Average Speed): {best_speed['controller']} ({best_speed['average_speed']:.2f} m/s)\n")
        f.write(f"Best Efficiency (Fastest Calculation): {best_efficiency['controller']} ({best_efficiency['average_calc_time']*1000:.3f} ms)\n")
    
    print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="MPC Controllers Batch Evaluation")
    parser.add_argument('--sample_num', type=int, default=DEFAULT_SAMPLE_NUM, 
                       help=f'Number of samples for evaluation (default: {DEFAULT_SAMPLE_NUM})')
    parser.add_argument('--T', type=int, default=DEFAULT_T, 
                       help=f'Prediction horizon T (default: {DEFAULT_T})')
    parser.add_argument('--d_safe', type=float, default=DEFAULT_D_SAFE, 
                       help=f'Safety distance (default: {DEFAULT_D_SAFE})')
    parser.add_argument('--max_steps', type=int, default=DEFAULT_MAX_STEPS, 
                       help=f'Maximum steps per episode (default: {DEFAULT_MAX_STEPS})')
    parser.add_argument('--output', type=str, default="logs/mpc_comparison_results.log", 
                       help='Output log file path')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test with reduced sample number')
    
    args = parser.parse_args()

    # 快速测试模式
    if args.quick:
        args.sample_num = 100
        print("Quick test mode: using 100 samples")
    
    # 执行对比测试
    results = compare_controllers(
        sample_num=args.sample_num,
        T=args.T,
        d_safe=args.d_safe,
        max_steps=args.max_steps
    )
    
    # 保存结果
    save_results_to_file(results, args.output)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for result in results:
        print(f"{result['controller']:15} | "
              f"Collision: {result['collision_probability']*100:6.2f}% | "
              f"Speed: {result['average_speed']:6.2f} m/s | "
              f"Time: {result['average_calc_time']*1000:6.3f} ms")
    
    print("=" * 80)

if __name__ == "__main__":
    main()