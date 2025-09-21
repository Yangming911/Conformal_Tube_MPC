#!/usr/bin/env python3
"""
Multi-pedestrian APF (Artificial Potential Field) controller batch testing and evaluation script
Treat pedestrians as static circular obstacles at current moment, only use current position for repulsion.
Output metrics consistent with evalcbf_multi.py: collision rate, average steps, average speed, average calc time.
"""

import numpy as np
import sys
from pathlib import Path
import time
import argparse
from typing import Dict, Any
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.simulator import (
    _initial_state_multi_pedestrian,
    _step_multi_pedestrian,
    _is_collision_multi_pedestrian,
    _done_multi_pedestrian,
)
import utils.constants as C


# Default parameters (aligned with evalcbf_multi.py)
DEFAULT_SAMPLE_NUM = 1000
DEFAULT_MAX_STEPS = 10000
DEFAULT_T = 1
DEFAULT_NUM_PEDESTRIANS = 1


def apf_controller_multi_pedestrian(state: Dict[str, Any], *,
                                    goal_x: float = C.CAR_RIGHT_LIMIT,
                                    goal_y: float = C.CAR_LANE_Y,
                                    k_att: float = 0.12,
                                    k_rep: float = 27.0,
                                    d_influence: float = 12.0,
                                    v_min: float = 0.0,
                                    v_max: float = 9.0,
                                    lateral_bias_gain: float = 0.85) -> float:
    """
    Minimal APF velocity controller:
    - Attraction term: Advance towards target x (constant desired velocity, linearly scaled by distance to target x)
    - Repulsion term: Add velocity reduction term for each pedestrian based on current Euclidean distance (only forward velocity dimension)
    - Optional lateral bias (no steering control used, only speed adjustment here; bias only for directional weighting of speed reduction)
    返回：当前步的车辆速度（m/s）。
    """
    car_x = float(state["car_x"])  # type: ignore
    car_y = float(state["car_y"])  # type: ignore
    walker_x_list = state["walker_x"]  # list[float]
    walker_y_list = state["walker_y"]  # list[float]

    # 吸引速度：与目标x距离成比例，饱和到 [v_min, v_max]
    dist_to_goal_x = max(0.0, float(goal_x) - car_x)
    v_att = k_att * dist_to_goal_x

    # 排斥：对近距离行人抑制前向速度
    v_rep = 0.0
    for px, py in zip(walker_x_list, walker_y_list):
        dx = float(px) - car_x
        dy = float(py) - car_y
        d = float(np.hypot(dx, dy)) + 1e-6
        if d < d_influence:
            # 经典形式：k_rep * (1/d - 1/R) / d^2，且仅在障碍前方时更强地减速
            base = k_rep * (1.0 / d - 1.0 / d_influence) / (d * d)
            forward_weight = 1.0 if dx >= 0.0 else 0.3  # 车前方影响更大
            lateral_bias = 1.0 + lateral_bias_gain * (abs(dy) / (d + 1e-6))
            v_rep += base * forward_weight * lateral_bias

    v_cmd = v_att - v_rep
    return float(np.clip(v_cmd, v_min, v_max))


def simulate_single_episode_multi_pedestrian(state: Dict[str, Any], controller_func, controller_name: str,
                                            T: int = DEFAULT_T,  # 占位以对齐接口
                                            d_safe: float = 2.5,  # 占位以对齐接口
                                            max_steps: int = DEFAULT_MAX_STEPS,
                                            rng: np.random.RandomState = None,
                                            **controller_kwargs):
    current_state = state.copy()
    total_speed = 0.0
    total_calc_time = 0.0
    steps_taken = 0

    for _ in range(max_steps):
        if _done_multi_pedestrian(current_state):
            break

        if _is_collision_multi_pedestrian(current_state):
            return True, steps_taken, total_speed / max(1, steps_taken), total_calc_time / max(1, steps_taken)

        start_time = time.perf_counter()
        if controller_name == "constant_speed":
            control_input = 15.0
        else:
            control_input = controller_func(current_state, **controller_kwargs)
        end_time = time.perf_counter()
        calc_time = end_time - start_time

        current_state["car_v"] = float(control_input)
        next_state, _ = _step_multi_pedestrian(current_state, rng)
        current_state = next_state

        total_speed += float(control_input)
        total_calc_time += calc_time
        steps_taken += 1

    collision_occurred = _is_collision_multi_pedestrian(current_state)
    return collision_occurred, steps_taken, total_speed / max(1, steps_taken), total_calc_time / max(1, steps_taken)


def batch_evaluate_controller_multi_pedestrian(controller_func,
                                               controller_name: str,
                                               sample_num: int = DEFAULT_SAMPLE_NUM,
                                               T: int = DEFAULT_T,
                                               d_safe: float = 2.5,
                                               max_steps: int = DEFAULT_MAX_STEPS,
                                               num_pedestrians: int = DEFAULT_NUM_PEDESTRIANS,
                                               **controller_kwargs):
    collision_count = 0
    total_steps = 0
    total_speed = 0.0
    total_calc_time = 0.0

    print(f"Evaluating {controller_name} with {num_pedestrians} pedestrians...")

    for i in tqdm(range(sample_num), desc=f"{controller_name}"):
        rng = np.random.RandomState(42 + i)
        car_speed = float(rng.uniform(1.0, 15.0))
        initial_state = _initial_state_multi_pedestrian(car_speed, rng, num_pedestrians)

        collision_occurred, steps_taken, avg_speed, avg_calc_time = simulate_single_episode_multi_pedestrian(
            initial_state, controller_func, controller_name, T, d_safe, max_steps, rng, **controller_kwargs
        )

        if collision_occurred:
            collision_count += 1
        total_steps += steps_taken
        total_speed += avg_speed
        total_calc_time += avg_calc_time

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
        **controller_kwargs,
    }


def compare_controllers_multi_pedestrian(sample_num: int = DEFAULT_SAMPLE_NUM,
                                         T: int = DEFAULT_T,
                                         d_safe: float = 2.5,
                                         max_steps: int = DEFAULT_MAX_STEPS,
                                         num_pedestrians: int = DEFAULT_NUM_PEDESTRIANS,
                                         **controller_kwargs):
    print("=" * 80)
    print("Multi-Pedestrian APF Controllers Comparison")
    print("=" * 80)
    print(f"Sample number: {sample_num}")
    print(f"Number of pedestrians: {num_pedestrians}")
    print(f"Prediction horizon T: {T}")
    print(f"Safety distance: {d_safe}")
    print(f"Max steps per episode: {max_steps}")
    print("=" * 80)

    controllers = [
        (lambda state, **kwargs: 15.0, "constant_speed"),
        (apf_controller_multi_pedestrian, "apf_multi"),
    ]

    results = []
    for controller_func, controller_name in controllers:
        result = batch_evaluate_controller_multi_pedestrian(
            controller_func, controller_name, sample_num, T, d_safe, max_steps, num_pedestrians, **controller_kwargs
        )
        results.append(result)

        print(f"\n{controller_name.upper()}:")
        print(f"  Collision Probability: {result['collision_probability']*100:.2f}%")
        print(f"  Collision Count: {result['collision_count']}/{result['sample_num']}")
        print(f"  Average Steps: {result['average_steps']:.1f}")
        print(f"  Average Speed: {result['average_speed']:.2f} m/s")
        print(f"  Average Calc Time: {result['average_calc_time']*1000:.3f} ms")

    return results


def save_results_to_file_multi_pedestrian(results, filename="logs/apf_multi_pedestrian_results.log"):
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    file_exists = os.path.exists(filename)
    with open(filename, 'a', encoding='utf-8') as f:
        if file_exists:
            f.write("\n" + "=" * 100 + "\n")
            f.write("NEW MULTI-PEDESTRIAN APF TEST RUN\n")
            f.write("=" * 100 + "\n\n")

        f.write("Multi-Pedestrian APF Controllers Comparison Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Number: {results[0]['sample_num']}\n")
        f.write(f"Number of Pedestrians: {results[0]['num_pedestrians']}\n")
        f.write(f"Prediction Horizon T: {results[0]['T']}\n")
        f.write(f"Safety Distance: {results[0]['d_safe']}\n")
        # 可选写出部分APF超参数（若存在）
        for key in ["k_att", "k_rep", "d_influence", "v_min", "v_max", "lateral_bias_gain"]:
            if key in results[0]:
                f.write(f"{key}: {results[0][key]}\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"{result['controller'].upper()}:\n")
            f.write(f"  Collision Probability: {result['collision_probability']*100:.2f}%\n")
            f.write(f"  Collision Count: {result['collision_count']}/{result['sample_num']}\n")
            f.write(f"  Average Steps: {result['average_steps']:.1f}\n")
            f.write(f"  Average Speed: {result['average_speed']:.2f} m/s\n")
            f.write(f"  Average Calc Time: {result['average_calc_time']*1000:.3f} ms\n")
            f.write("-" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Pedestrian APF Controllers Batch Evaluation")
    parser.add_argument('--sample_num', type=int, default=DEFAULT_SAMPLE_NUM,
                        help=f'Number of samples for evaluation (default: {DEFAULT_SAMPLE_NUM})')
    parser.add_argument('--T', type=int, default=DEFAULT_T,
                        help=f'Prediction horizon T (default: {DEFAULT_T})')
    parser.add_argument('--d_safe', type=float, default=2.5,
                        help='Safety distance placeholder (unused, for interface parity)')
    parser.add_argument('--max_steps', type=int, default=DEFAULT_MAX_STEPS,
                        help=f'Maximum steps per episode (default: {DEFAULT_MAX_STEPS})')
    parser.add_argument('--num_pedestrians', type=int, default=DEFAULT_NUM_PEDESTRIANS,
                        help=f'Number of pedestrians (default: {DEFAULT_NUM_PEDESTRIANS})')
    parser.add_argument('--output', type=str, default="logs/apf_multi_pedestrian_results.log",
                        help='Output log file path')
    parser.add_argument('--quick', action='store_true', help='Quick test with reduced sample number')
    # APF 超参数
    parser.add_argument('--k_att', type=float, default=0.12, help='Attractive gain to goal x')
    parser.add_argument('--k_rep', type=float, default=27.0, help='Repulsive gain from pedestrians')
    parser.add_argument('--d_influence', type=float, default=12.0, help='Repulsion influence distance (m)')
    parser.add_argument('--v_min', type=float, default=0.0, help='Minimum speed (m/s)')
    parser.add_argument('--v_max', type=float, default=9.0, help='Maximum speed (m/s)')
    parser.add_argument('--lateral_bias_gain', type=float, default=0.85, help='Lateral bias gain in repulsion')

    args = parser.parse_args()

    if args.quick:
        args.sample_num = 100
        print("Quick test mode: using 100 samples")

    results = compare_controllers_multi_pedestrian(
        sample_num=args.sample_num,
        T=args.T,
        d_safe=args.d_safe,
        max_steps=args.max_steps,
        num_pedestrians=args.num_pedestrians,
        k_att=args.k_att,
        k_rep=args.k_rep,
        d_influence=args.d_influence,
        v_min=args.v_min,
        v_max=args.v_max,
        lateral_bias_gain=args.lateral_bias_gain,
    )

    save_results_to_file_multi_pedestrian(results, args.output)

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


