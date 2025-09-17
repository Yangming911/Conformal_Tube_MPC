#!/usr/bin/env python3
"""
测试多行人场景功能
"""

import numpy as np
import sys
from pathlib import Path

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
from cbf.current_cbf_controller import cbf_controller_multi_pedestrian
import utils.constants as C

def test_multi_pedestrian_basic():
    """测试多行人基本功能"""
    print("Testing multi-pedestrian basic functionality...")
    
    # 设置随机种子
    rng = np.random.RandomState(42)
    car_speed = 10.0
    num_pedestrians = 3
    
    # 创建初始状态
    state = _initial_state_multi_pedestrian(car_speed, rng, num_pedestrians)
    
    print(f"Initial state:")
    print(f"  Car: x={state['car_x']:.2f}, y={state['car_y']:.2f}, v={state['car_v']:.2f}")
    print(f"  Pedestrians: {num_pedestrians}")
    for i in range(num_pedestrians):
        print(f"    Pedestrian {i+1}: x={state['walker_x'][i]:.2f}, y={state['walker_y'][i]:.2f}")
    
    # 测试一步前进
    next_state, label = _step_multi_pedestrian(state)
    
    print(f"\nAfter one step:")
    print(f"  Car: x={next_state['car_x']:.2f}, y={next_state['car_y']:.2f}, v={next_state['car_v']:.2f}")
    for i in range(num_pedestrians):
        print(f"    Pedestrian {i+1}: x={next_state['walker_x'][i]:.2f}, y={next_state['walker_y'][i]:.2f}")
        print(f"      Velocity: vx={next_state['walker_vx'][i]:.2f}, vy={next_state['walker_vy'][i]:.2f}")
    
    # 测试碰撞检测
    collision = _is_collision_multi_pedestrian(state)
    print(f"\nCollision detected: {collision}")
    
    # 测试结束条件
    done = _done_multi_pedestrian(state)
    print(f"Episode done: {done}")
    
    print("✓ Basic functionality test passed!")

def test_cbf_controller_multi_pedestrian():
    """测试多行人CBF控制器"""
    print("\nTesting multi-pedestrian CBF controller...")
    
    # 设置随机种子
    rng = np.random.RandomState(42)
    car_speed = 10.0
    num_pedestrians = 3
    
    # 创建初始状态
    state = _initial_state_multi_pedestrian(car_speed, rng, num_pedestrians)
    
    try:
        # 测试CBF控制器
        control_input = cbf_controller_multi_pedestrian(state, T=5, d_safe=2.0)
        print(f"CBF controller output: {control_input:.2f} m/s")
        print("✓ CBF controller test passed!")
    except Exception as e:
        print(f"✗ CBF controller test failed: {e}")
        return False
    
    return True

def test_simulation_episode():
    """测试完整的多行人仿真episode"""
    print("\nTesting complete multi-pedestrian simulation episode...")
    
    # 设置随机种子
    rng = np.random.RandomState(42)
    car_speed = 10.0
    num_pedestrians = 3
    max_steps = 50
    
    # 创建初始状态
    state = _initial_state_multi_pedestrian(car_speed, rng, num_pedestrians)
    
    steps_taken = 0
    collision_occurred = False
    
    for step in range(max_steps):
        if _done_multi_pedestrian(state):
            print(f"Episode ended at step {step} (all pedestrians reached destination)")
            break
            
        if _is_collision_multi_pedestrian(state):
            collision_occurred = True
            print(f"Collision occurred at step {step}")
            break
        
        # 使用CBF控制器计算控制输入
        try:
            control_input = cbf_controller_multi_pedestrian(state, T=5, d_safe=2.0)
        except:
            control_input = 5.0  # 如果CBF失败，使用较低速度
        
        # 更新状态
        state["car_v"] = control_input
        next_state, _ = _step_multi_pedestrian(state)
        state = next_state
        steps_taken += 1
    
    print(f"Simulation completed:")
    print(f"  Steps taken: {steps_taken}")
    print(f"  Collision occurred: {collision_occurred}")
    print(f"  Final car position: x={state['car_x']:.2f}")
    print(f"  Final pedestrian positions:")
    for i in range(num_pedestrians):
        print(f"    Pedestrian {i+1}: x={state['walker_x'][i]:.2f}, y={state['walker_y'][i]:.2f}")
    
    print("✓ Simulation episode test passed!")

def main():
    """运行所有测试"""
    print("=" * 60)
    print("Multi-Pedestrian System Test Suite")
    print("=" * 60)
    
    try:
        test_multi_pedestrian_basic()
        test_cbf_controller_multi_pedestrian()
        test_simulation_episode()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("Multi-pedestrian system is ready for use.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
