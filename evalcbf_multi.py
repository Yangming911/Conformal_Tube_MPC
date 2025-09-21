#!/usr/bin/env python3
"""
Multi-pedestrian CBF controller batch testing and evaluation script
Compare CBF controller performance in multi-pedestrian scenarios
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

from envs.simulator import (
    _initial_state_multi_pedestrian, 
    _step_multi_pedestrian, 
    _is_collision_multi_pedestrian, 
    _done_multi_pedestrian
)
from cbf.current_cbf_controller import cbf_controller_multi_pedestrian, cbf_controller_multi_pedestrian_no_eta
import utils.constants as C

# Set random seed
np.random.seed(42)

# Test parameters
DEFAULT_SAMPLE_NUM = 1000
DEFAULT_MAX_STEPS = 10000
DEFAULT_D_SAFE = 2.5
DEFAULT_T = 1
DEFAULT_NUM_PEDESTRIANS = 1

def simulate_single_episode_multi_pedestrian(state, controller_func, controller_name, 
                                           T=DEFAULT_T, d_safe=DEFAULT_D_SAFE, 
                                           max_steps=DEFAULT_MAX_STEPS, rng=None, cp_alpha=0.85, gamma=1.0):
    """
    Simulate a single multi-pedestrian episode
    
    Args:
        state: Initial state dictionary
        controller_func: Controller function
        controller_name: Controller name
        T: Prediction horizon
        d_safe: Safety distance
        max_steps: Maximum steps
    
    Returns:
        tuple: (collision_occurred, steps_taken, average_speed, average_calc_time)
    """
    current_state = state.copy()
    total_speed = 0.0
    total_calc_time = 0.0
    steps_taken = 0
    
    for step in range(max_steps):
        if _done_multi_pedestrian(current_state):
            break
            
        # Check collision
        if _is_collision_multi_pedestrian(current_state):
            return True, steps_taken, total_speed / max(1, steps_taken), total_calc_time / max(1, steps_taken)
        
        # Calculate control input
        start_time = time.perf_counter()
        
        if controller_name == "constant_speed":
            # Constant speed through, fixed at 15m/s
            control_input = 15.0
        else:
            # Use CBF controller
            control_input = controller_func(current_state, T=T, d_safe=d_safe, cp_alpha=cp_alpha, gamma=gamma)
        
        end_time = time.perf_counter()
        calc_time = end_time - start_time
        
        # Update state
        current_state["car_v"] = control_input
        next_state, _ = _step_multi_pedestrian(current_state, rng)
        current_state = next_state
        
        # Record statistics
        total_speed += control_input
        total_calc_time += calc_time
        steps_taken += 1
    
    # Final collision check
    collision_occurred = _is_collision_multi_pedestrian(current_state)
    
    return collision_occurred, steps_taken, total_speed / max(1, steps_taken), total_calc_time / max(1, steps_taken)

def batch_evaluate_controller_multi_pedestrian(controller_func, controller_name, 
                                             sample_num=DEFAULT_SAMPLE_NUM, 
                                             T=DEFAULT_T, d_safe=DEFAULT_D_SAFE, 
                                             max_steps=DEFAULT_MAX_STEPS,
                                             num_pedestrians=DEFAULT_NUM_PEDESTRIANS,
                                             cp_alpha=0.85, gamma=1.0):
    """
    Batch evaluate a single multi-pedestrian controller
    
    Args:
        controller_func: Controller function
        controller_name: Controller name
        sample_num: Number of samples
        T: Prediction horizon
        d_safe: Safety distance
        max_steps: Maximum steps
        num_pedestrians: Number of pedestrians
    
    Returns:
        dict: Evaluation results
    """
    collision_count = 0
    total_steps = 0
    total_speed = 0.0
    total_calc_time = 0.0
    
    print(f"Evaluating {controller_name} with {num_pedestrians} pedestrians...")
    
    for i in tqdm(range(sample_num), desc=f"{controller_name}"):
        # Generate random initial state, use fixed seed to ensure reproducibility
        rng = np.random.RandomState(42 + i)  # 为每个样本使用不同的但固定的种子
        car_speed = float(rng.uniform(1.0, 15.0))
        initial_state = _initial_state_multi_pedestrian(car_speed, rng, num_pedestrians)
        
        # Simulate single episode
        collision_occurred, steps_taken, avg_speed, avg_calc_time = simulate_single_episode_multi_pedestrian(
            initial_state, controller_func, controller_name, T, d_safe, max_steps, rng, cp_alpha, gamma
        )
        
        if collision_occurred:
            collision_count += 1
        
        total_steps += steps_taken
        total_speed += avg_speed
        total_calc_time += avg_calc_time
    
    # Calculate statistical results
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
        "d_safe": d_safe,
        "num_pedestrians": num_pedestrians,
        "cp_alpha": cp_alpha,
        "gamma": gamma
    }

def compare_controllers_multi_pedestrian(sample_num=DEFAULT_SAMPLE_NUM, 
                                       T=DEFAULT_T, d_safe=DEFAULT_D_SAFE, 
                                       max_steps=DEFAULT_MAX_STEPS,
                                       num_pedestrians=DEFAULT_NUM_PEDESTRIANS,
                                       cp_alpha=0.85, gamma=1.0):
    """
    Compare performance of all multi-pedestrian controllers
    
    Args:
        sample_num: Number of samples
        T: Prediction horizon
        d_safe: Safety distance
        max_steps: Maximum steps
        num_pedestrians: Number of pedestrians
    
    Returns:
        list: Evaluation results for all controllers
    """
    print("=" * 80)
    print("Multi-Pedestrian CBF Controllers Comparison")
    print("=" * 80)
    print(f"Sample number: {sample_num}")
    print(f"Conformal prediction alpha: {cp_alpha}")
    print(f"Number of pedestrians: {num_pedestrians}")
    print(f"Prediction horizon T: {T}")
    print(f"Safety distance: {d_safe}")
    print(f"Max steps per episode: {max_steps}")
    print("=" * 80)
    
    # Define controllers
    controllers = [
        (lambda state, **kwargs: 15.0, "constant_speed"),
        (cbf_controller_multi_pedestrian, "current_cbf_multi"),
        # (cbf_controller_multi_pedestrian_no_eta, "current_cbf_multi_no_eta"),
    ]
    
    results = []
    
    for controller_func, controller_name in controllers:
        result = batch_evaluate_controller_multi_pedestrian(
            controller_func, controller_name, sample_num, T, d_safe, max_steps, num_pedestrians, cp_alpha, gamma
        )
        results.append(result)
        
        # Print single controller results
        print(f"\n{controller_name.upper()}:")
        print(f"  Collision Probability: {result['collision_probability']*100:.2f}%")
        print(f"  Collision Count: {result['collision_count']}/{result['sample_num']}")
        print(f"  Average Steps: {result['average_steps']:.1f}")
        print(f"  Average Speed: {result['average_speed']:.2f} m/s")
        print(f"  Average Calc Time: {result['average_calc_time']*1000:.3f} ms")
    
    return results

def save_results_to_file_multi_pedestrian(results, filename="logs/cbf_multi_pedestrian_results.log"):
    """
    Save multi-pedestrian results to file (append mode)
    
    Args:
        results: List of evaluation results
        filename: Output filename (if file exists, will append to end of file)
    """
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Check if file exists to determine if we need a separator
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', encoding='utf-8') as f:
        # Add separator if file already exists
        if file_exists:
            f.write("\n" + "=" * 100 + "\n")
            f.write("NEW MULTI-PEDESTRIAN TEST RUN\n")
            f.write("=" * 100 + "\n\n")
        
        f.write("Multi-Pedestrian CBF Controllers Comparison Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Number: {results[0]['sample_num']}\n")
        f.write(f"Number of Pedestrians: {results[0]['num_pedestrians']}\n")
        f.write(f"Conformal Prediction Alpha: {results[0]['cp_alpha']}\n")
        f.write(f"Prediction Horizon T: {results[0]['T']}\n")
        f.write(f"Safety Distance: {results[0]['d_safe']}\n")
        f.write(f"CBF Gamma: {results[0]['gamma']}\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"{result['controller'].upper()}:\n")
            f.write(f"  Collision Probability: {result['collision_probability']*100:.2f}%\n")
            f.write(f"  Collision Count: {result['collision_count']}/{result['sample_num']}\n")
            f.write(f"  Average Steps: {result['average_steps']:.1f}\n")
            f.write(f"  Average Speed: {result['average_speed']:.2f} m/s\n")
            f.write(f"  Average Calc Time: {result['average_calc_time']*1000:.3f} ms\n")
            f.write("-" * 40 + "\n")
        
        # Add comparison analysis
        f.write("\nCOMPARISON ANALYSIS:\n")
        f.write("=" * 40 + "\n")
        
        # Find best controllers
        best_safety = min(results, key=lambda x: x['collision_probability'])
        best_speed = max(results, key=lambda x: x['average_speed'])
        best_efficiency = min(results, key=lambda x: x['average_calc_time'])
        
        f.write(f"Best Safety (Lowest Collision Rate): {best_safety['controller']} ({best_safety['collision_probability']*100:.2f}%)\n")
        f.write(f"Best Speed (Highest Average Speed): {best_speed['controller']} ({best_speed['average_speed']:.2f} m/s)\n")
        f.write(f"Best Efficiency (Fastest Calculation): {best_efficiency['controller']} ({best_efficiency['average_calc_time']*1000:.3f} ms)\n")
    
    if file_exists:
        print(f"\nResults appended to: {filename}")
    else:
        print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Pedestrian CBF Controllers Batch Evaluation")
    parser.add_argument('--sample_num', type=int, default=DEFAULT_SAMPLE_NUM, 
                       help=f'Number of samples for evaluation (default: {DEFAULT_SAMPLE_NUM})')
    parser.add_argument('--T', type=int, default=DEFAULT_T, 
                       help=f'Prediction horizon T (default: {DEFAULT_T})')
    parser.add_argument('--d_safe', type=float, default=DEFAULT_D_SAFE, 
                       help=f'Safety distance (default: {DEFAULT_D_SAFE})')
    parser.add_argument('--max_steps', type=int, default=DEFAULT_MAX_STEPS, 
                       help=f'Maximum steps per episode (default: {DEFAULT_MAX_STEPS})')
    parser.add_argument('--num_pedestrians', type=int, default=DEFAULT_NUM_PEDESTRIANS, 
                       help=f'Number of pedestrians (default: {DEFAULT_NUM_PEDESTRIANS})')
    parser.add_argument('--output', type=str, default="logs/cbf_multi_pedestrian_results.log", 
                       help='Output log file path')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test with reduced sample number')
    parser.add_argument('--cp_alpha', type=float, default=0.85, 
                       help='Conformal prediction alpha parameter (default: 0.85)')
    parser.add_argument('--gamma', type=float, default=0.5, 
                       help='CBF gamma parameter (default: 1.0)')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.sample_num = 100
        print("Quick test mode: using 100 samples")
    
    # Execute comparison test
    results = compare_controllers_multi_pedestrian(
        sample_num=args.sample_num,
        T=args.T,
        d_safe=args.d_safe,
        max_steps=args.max_steps,
        num_pedestrians=args.num_pedestrians,
        cp_alpha=args.cp_alpha,
        gamma=args.gamma
    )
    
    # Save results
    save_results_to_file_multi_pedestrian(results, args.output)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for result in results:
        print(f"{result['controller']:20} | "
              f"Collision: {result['collision_probability']*100:6.2f}% | "
              f"Speed: {result['average_speed']:6.2f} m/s | "
              f"Time: {result['average_calc_time']*1000:6.3f} ms")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
