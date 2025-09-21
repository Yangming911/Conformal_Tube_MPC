import sys
from pathlib import Path

# Ensure project root is on sys.path so that `utils` can be imported when running this file directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional

from envs.dynamics_social_force import walker_logic_SF
from envs.dynamics_social_force import l_r, l_f, l_w

import utils.constants as C


def _initial_state(car_v: float, rng: np.random.RandomState) -> Dict[str, float]:
    """Create initial state; sample walker_y uniformly between start and destination to reduce bias."""
    return {
        "car_x": float(C.CAR_START_X),
        "car_y": float(C.CAR_LANE_Y),
        "car_v": float(car_v),
        "walker_x": float(C.WALKER_START_X),
        "walker_y": float(rng.uniform(C.WALKER_START_Y, C.WALKER_DESTINATION_Y)),
        "walker_vx": float(C.WALKER_START_V_X),
        "walker_vy": float(C.WALKER_START_V_Y),
    }


def _initial_state_multi_pedestrian(car_v: float, rng: np.random.RandomState, num_pedestrians: int = None) -> Dict[str, any]:
    """Create initial state for multiple pedestrians; sample walker positions uniformly."""
    if num_pedestrians is None:
        num_pedestrians = C.num_pedestrians
    
    # Generate initial position for each pedestrian
    walker_x_list = []
    walker_y_list = []
    walker_vx_list = []
    walker_vy_list = []
    
    for i in range(num_pedestrians):
        # Slightly disperse pedestrians in x direction to avoid overlap
        # x_offset = rng.uniform(-2.0, 2.0)  # Within ±2 meters around starting position
        walker_x_list.append(float(C.WALKER_START_X))
        
        # y position uniformly distributed between start and end points
        walker_y_list.append(float(rng.uniform(C.WALKER_START_Y, C.WALKER_DESTINATION_Y)))
        
        # Initial velocity is 0
        walker_vx_list.append(float(C.WALKER_START_V_X))
        walker_vy_list.append(float(C.WALKER_START_V_Y))
    
    return {
        "car_x": float(C.CAR_START_X),
        "car_y": float(C.CAR_LANE_Y),
        "car_v": float(car_v),
        "walker_x": walker_x_list,
        "walker_y": walker_y_list,
        "walker_vx": walker_vx_list,
        "walker_vy": walker_vy_list,
    }


def _step(state: Dict[str, float], rng: np.random.RandomState = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Advance simulation by one time step C.dt without any wrapping/teleportation.

    Returns:
    - next_state: full next state after integrating for dt
    - label: next-step pedestrian velocity (the learning target)
    """
    car_x = state["car_x"]
    car_y = state["car_y"]
    car_v = state["car_v"]
    walker_x = state["walker_x"]
    walker_y = state["walker_y"]
    walker_vx = state["walker_vx"]
    walker_vy = state["walker_vy"]

    # Target: next-step pedestrian velocity from social-force dynamics
    next_walker_vx, next_walker_vy = walker_logic_SF(
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

    # Integrate positions with dt (no bounds clamping)
    dt = float(C.dt)
    next_walker_x = walker_x + next_walker_vx * dt
    next_walker_y = walker_y + next_walker_vy * dt

    next_car_x = car_x + car_v * dt

    next_state = {
        "car_x": float(next_car_x),
        "car_y": float(car_y),
        "car_v": float(car_v),
        "walker_x": float(next_walker_x),
        "walker_y": float(next_walker_y),
        "walker_vx": float(next_walker_vx),
        "walker_vy": float(next_walker_vy),
    }

    label = {
        "next_walker_vx": float(next_walker_vx),
        "next_walker_vy": float(next_walker_vy),
    }

    return next_state, label


def _step_multi_pedestrian(state: Dict[str, any], rng: np.random.RandomState = None) -> Tuple[Dict[str, any], Dict[str, any]]:
    """
    Advance simulation by one time step C.dt for multiple pedestrians.

    Returns:
    - next_state: full next state after integrating for dt
    - label: next-step pedestrian velocities (the learning target)
    """
    car_x = state["car_x"]
    car_y = state["car_y"]
    car_v = state["car_v"]
    walker_x_list = state["walker_x"]
    walker_y_list = state["walker_y"]
    walker_vx_list = state["walker_vx"]
    walker_vy_list = state["walker_vy"]

    num_pedestrians = len(walker_x_list)
    
    # Calculate next state for each pedestrian
    next_walker_x_list = []
    next_walker_y_list = []
    next_walker_vx_list = []
    next_walker_vy_list = []

    for i in range(num_pedestrians):
        # Temporarily use simplified pedestrian dynamics model to avoid neural network loading issues
        # Use social force model to predict next velocity for each pedestrian
        from envs.dynamics_social_force import walker_logic_SF
        
        next_walker_vx, next_walker_vy = walker_logic_SF(
            car_v,
            car_x,
            car_y,
            walker_x_list[i],
            walker_y_list[i],
            walker_vx_list[i],
            walker_vy_list[i],
            v_max=C.v_max,
            a_max=C.a_max,
            destination_y=C.WALKER_DESTINATION_Y,
            rng=rng,
        )
        
        # Integrate position
        dt = float(C.dt)
        next_walker_x = walker_x_list[i] + next_walker_vx * dt
        next_walker_y = walker_y_list[i] + next_walker_vy * dt

        next_walker_x_list.append(float(next_walker_x))
        next_walker_y_list.append(float(next_walker_y))
        next_walker_vx_list.append(float(next_walker_vx))
        next_walker_vy_list.append(float(next_walker_vy))

    next_car_x = car_x + car_v * dt

    next_state = {
        "car_x": float(next_car_x),
        "car_y": float(car_y),
        "car_v": float(car_v),
        "walker_x": next_walker_x_list,
        "walker_y": next_walker_y_list,
        "walker_vx": next_walker_vx_list,
        "walker_vy": next_walker_vy_list,
    }

    label = {
        "next_walker_vx": next_walker_vx_list,
        "next_walker_vy": next_walker_vy_list,
    }

    return next_state, label


def _done(state: Dict[str, float]) -> bool:
    """Episode termination without any resetting when limits are exceeded."""
    if state["car_x"] > C.CAR_RIGHT_LIMIT:
        return True
    if state["walker_y"] >= C.WALKER_DESTINATION_Y:
        return True
    return False


def _done_multi_pedestrian(state: Dict[str, any]) -> bool:
    """Episode termination for multiple pedestrians when limits are exceeded."""
    if state["car_x"] > C.CAR_RIGHT_LIMIT:
        return True
    
    # 检查是否所有行人都到达了目的地
    walker_y_list = state["walker_y"]
    all_pedestrians_finished = all(walker_y >= C.WALKER_DESTINATION_Y for walker_y in walker_y_list)
    if all_pedestrians_finished:
        return True
    
    return False


def _is_collision(state: Dict[str, float]) -> bool:
    """Check if the pedestrian center lies within the car's rectangle expanded by 0.5 m on all sides."""
    car_x = state["car_x"]
    car_y = state["car_y"]
    walker_x = state["walker_x"]
    walker_y = state["walker_y"]

    car_left = car_x - float(l_r) - 1
    car_right = car_x + float(l_f) + 1
    half_width = float(l_w) / 2.0
    car_bottom = (car_y - half_width) - 1
    car_top = (car_y + half_width) + 1

    if (car_left <= walker_x <= car_right) and (car_bottom <= walker_y <= car_top):
        return True
    return False


def _is_collision_multi_pedestrian(state: Dict[str, any]) -> bool:
    """Check if any pedestrian center lies within the car's rectangle expanded by 0.5 m on all sides."""
    car_x = state["car_x"]
    car_y = state["car_y"]
    walker_x_list = state["walker_x"]
    walker_y_list = state["walker_y"]

    car_left = car_x - float(l_r) - 1
    car_right = car_x + float(l_f) + 1
    half_width = float(l_w) / 2.0
    car_bottom = (car_y - half_width) - 1
    car_top = (car_y + half_width) + 1

    # 检查每个行人是否与车辆碰撞
    for i in range(len(walker_x_list)):
        walker_x = walker_x_list[i]
        walker_y = walker_y_list[i]
        
        if (car_left <= walker_x <= car_right) and (car_bottom <= walker_y <= car_top):
            return True
    
    return False


def simulate_trajectory(
    max_steps: int,
    car_speed: Optional[float] = None,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict[str, float]]:
    """
    Roll out a single trajectory:
    - Inputs: car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy
    - Targets: next_walker_vx, next_walker_vy

    No state wrapping/teleporting; episode ends on limits via _done().
    """
    if rng is None:
        rng = np.random.RandomState()

    if car_speed is None:
        car_speed = float(rng.uniform(1.0, 15.0))

    state = _initial_state(car_speed, rng)
    samples: List[Dict[str, float]] = []

    for _ in range(max_steps):
        if _done(state):
            break

        features = {
            "car_x": state["car_x"],
            "car_y": state["car_y"],
            "car_v": state["car_v"],
            "walker_x": state["walker_x"],
            "walker_y": state["walker_y"],
            "walker_vx": state["walker_vx"],
            "walker_vy": state["walker_vy"],
        }

        next_state, label = _step(state)
        samples.append({**features, **label})
        state = next_state

    return samples


def simulate_trajectory_multi_pedestrian(
    max_steps: int,
    car_speed: Optional[float] = None,
    rng: Optional[np.random.RandomState] = None,
    num_pedestrians: Optional[int] = None,
) -> List[Dict[str, any]]:
    """
    Roll out a single trajectory for multiple pedestrians:
    - Inputs: car_x, car_y, car_v, walker_x (list), walker_y (list), walker_vx (list), walker_vy (list)
    - Targets: next_walker_vx (list), next_walker_vy (list)

    No state wrapping/teleporting; episode ends on limits via _done_multi_pedestrian().
    """
    if rng is None:
        rng = np.random.RandomState()

    if car_speed is None:
        car_speed = float(rng.uniform(1.0, 15.0))

    state = _initial_state_multi_pedestrian(car_speed, rng, num_pedestrians)
    samples: List[Dict[str, any]] = []

    for _ in range(max_steps):
        if _done_multi_pedestrian(state):
            break

        features = {
            "car_x": state["car_x"],
            "car_y": state["car_y"],
            "car_v": state["car_v"],
            "walker_x": state["walker_x"].copy(),
            "walker_y": state["walker_y"].copy(),
            "walker_vx": state["walker_vx"].copy(),
            "walker_vy": state["walker_vy"].copy(),
        }

        next_state, label = _step_multi_pedestrian(state)
        samples.append({**features, **label})
        state = next_state

    return samples


def compute_collision_ratio(
    num_episodes: int = 200,
    max_steps_per_episode: int = 200,
    seed: int = 123,
) -> float:
    """Simulate multiple episodes and return the fraction that had at least one collision."""
    rng = np.random.RandomState(seed)
    episodes_with_collision = 0

    for _ in range(num_episodes):
        car_speed = float(15.0)
        state = _initial_state(car_speed, rng)
        collided = False

        for _step_idx in range(max_steps_per_episode):
            if _is_collision(state):
                collided = True
                break
            if _done(state):
                break
            next_state, _ = _step(state)
            state = next_state

        if collided:
            episodes_with_collision += 1

    return episodes_with_collision / float(num_episodes)


def compute_collision_ratio_multi_pedestrian(
    num_episodes: int = 200,
    max_steps_per_episode: int = 200,
    seed: int = 123,
    num_pedestrians: Optional[int] = None,
) -> float:
    """Simulate multiple episodes with multiple pedestrians and return the fraction that had at least one collision."""
    rng = np.random.RandomState(seed)
    episodes_with_collision = 0

    for _ in range(num_episodes):
        car_speed = float(15.0)
        state = _initial_state_multi_pedestrian(car_speed, rng, num_pedestrians)
        collided = False

        for _step_idx in range(max_steps_per_episode):
            if _is_collision_multi_pedestrian(state):
                collided = True
                break
            if _done_multi_pedestrian(state):
                break
            next_state, _ = _step_multi_pedestrian(state)
            state = next_state

        if collided:
            episodes_with_collision += 1

    return episodes_with_collision / float(num_episodes)


def collect_dataset(
    num_episodes: int = 500,
    max_steps_per_episode: int = 200,
    seed: int = 42,
    save_path: Optional[str] = "assets/sf_dataset.csv",
) -> pd.DataFrame:
    """Generate a dataset of trajectories and optionally save to CSV."""
    rng = np.random.RandomState(seed)
    all_samples: List[Dict[str, float]] = []

    for _ in range(num_episodes):
        car_speed = float(rng.uniform(1.0, 15.0))
        traj = simulate_trajectory(max_steps=max_steps_per_episode, car_speed=car_speed, rng=rng)
        all_samples.extend(traj)

    df = pd.DataFrame(all_samples, columns=[
        "car_x", "car_y", "car_v", "walker_x", "walker_y", "walker_vx", "walker_vy",
        "next_walker_vx", "next_walker_vy",
    ])

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


if __name__ == "__main__":
    df = collect_dataset(num_episodes=5000, max_steps_per_episode=10000, seed=123)
    print("Dataset saved with shape:", df.shape)
    print(df.head())
    ratio = compute_collision_ratio(num_episodes=200, max_steps_per_episode=10000, seed=123)
    print("Collision episode ratio:", ratio)

"""
output
Dataset saved with shape: (298248, 9)
      car_x  car_y      car_v   walker_x  walker_y  walker_vx  walker_vy  next_walker_vx  next_walker_vy
0  0.000000   12.0  10.750569  30.000000  5.722787   0.000000   0.000000       -0.054540       -0.076643
1  1.075057   12.0  10.750569  29.994546  5.715122  -0.054540  -0.076643       -0.012627       -0.024716
2  2.150114   12.0  10.750569  29.993283  5.712651  -0.012627  -0.024716       -0.030233        0.925495
3  3.225171   12.0  10.750569  29.990260  5.805200  -0.030233   0.925495       -0.154183        1.500526
4  4.300227   12.0  10.750569  29.974842  5.955253  -0.154183   1.500526       -0.111897        1.585189
Collision episode ratio: 0.225 (15m/s)
"""