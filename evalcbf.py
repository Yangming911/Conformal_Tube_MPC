import numpy as np
from tqdm import tqdm
from envs.dynamics import update_car_position, update_walker_position
from cbf.vanilla_cbf_controller import vanilla_cbf_controller
from cbf.cp_cbf_controller import cp_cbf_controller

import pygame
import argparse
import sys
import time
np.random.seed(0)

# 初始化pygame
pygame.init()

# 设置屏幕尺寸和标题
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Traffic Simulation")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (169, 169, 169)
RED = (255, 0, 0)

# 参数
d_safe = 10
COLLISION_THRESHOLD = 10

def calculate_collision_probability(T=1, sample_num=1000, mode='random'):
    collision_count = 0
    average_speed = 0
    average_calcu_time = 0
    for _ in tqdm(range(sample_num), desc=f"Simulating {mode} T={T}"):
        car_x = -40
        car_y = 320
        car_speed = np.random.uniform(low=1, high=15)

        walker_x = 400
        walker_y = 200
        total_time_steps = 500
        average_speed_little_loop = 0
        average_calcu_time_little_loop = 0
        step_cnt = total_time_steps
        for step in range(total_time_steps):
            collision_occurred = False
            car_x, __ = update_car_position(car_x, car_speed)
            walker_x, walker_y = update_walker_position(car_speed, walker_x, walker_y)
            
            start_time = time.perf_counter()
            if mode == 'random':
                car_speed = 15
            elif mode == 'vanillacbf':
                car_speed = vanilla_cbf_controller(car_x, np.array([walker_x, walker_y]), T, d_safe, car_y)
            elif mode == 'cpcbf':
                car_speed = cp_cbf_controller(car_x, np.array([walker_x, walker_y]), T, d_safe, car_y)
            end_time = time.perf_counter()
            average_calcu_time_little_loop += (end_time - start_time)
            average_speed_little_loop += car_speed
            
            if abs(walker_x - car_x) < COLLISION_THRESHOLD and abs(walker_y - car_y) < COLLISION_THRESHOLD:
                collision_occurred = True
                step_cnt = step
                break
        average_speed_little_loop /= step_cnt
        average_calcu_time_little_loop /= step_cnt
        average_speed += average_speed_little_loop
        average_calcu_time += average_calcu_time_little_loop

        if collision_occurred:
            collision_count += 1
    average_speed /= sample_num
    average_calcu_time /= sample_num
    return collision_count / sample_num, average_speed, average_calcu_time*1000 # ms


def main(args):
    if args.display:
        running = True
        clock = pygame.time.Clock()

        car_x = 0
        car_y = 320
        car_speed = 8

        walker_x = 400
        walker_y = 200

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            car_x, car_speed = update_car_position(car_x, car_speed)
            walker_x, walker_y = update_walker_position(car_speed, walker_x, walker_y)
            car_x, __ = update_car_position(car_x, car_speed)
            walker_x, walker_y = update_walker_position(car_speed, walker_x, walker_y)

            if abs(walker_x - car_x) < COLLISION_THRESHOLD and abs(walker_y - car_y) < COLLISION_THRESHOLD:
                break

            screen.fill(WHITE)
            pygame.draw.rect(screen, GRAY, (0, 250, 800, 200))
            for i in range(5):
                pygame.draw.rect(screen, WHITE, (350, 250 + i * 40, 100, 20))

            pygame.draw.rect(screen, BLACK, (car_x, car_y, 40, 20))
            pygame.draw.circle(screen, BLACK, (int(walker_x), int(walker_y)), 10)
            pygame.draw.circle(screen, RED, (int(car_x) + 20, int(car_y) + 10), 20, 2)
            pygame.draw.circle(screen, RED, (int(walker_x), int(walker_y)), 20, 2)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
        sys.exit()

    else:
        if args.mode == 'random':
            collision_prob, average_speed, average_calcu_time = calculate_collision_probability(mode='random')
            print(f"Collision probability: {collision_prob * 100:.2f}%")
            print(f"Average speed: {average_speed:.2f}")
            print(f"Average calculation time: {average_calcu_time:.6f} s")

        elif args.mode in ['vanillacbf', 'cpcbf']:
            collision_prob, average_speed, average_calcu_time = calculate_collision_probability(T=args.T, mode=args.mode)
            print(f"Collision probability: {collision_prob * 100:.2f}%")
            print(f"Average speed: {average_speed:.2f}")
            print(f"Average calculation time: {average_calcu_time:.6f} s")

        elif args.mode == 'batch':
            controllers = ['random','vanillacbf', 'cpcbf']
            T_values = [1, 10, 20]
            log_file = "batch_results_cbf_parallel.log"
            with open(log_file, 'w') as f:
                f.write("Batch Evaluation Log\n")
                f.write("=" * 40 + "\n")
            for ctrl in controllers:
                for T_val in T_values:
                    collision_prob, avg_speed, avg_time = calculate_collision_probability(T=T_val, mode=ctrl)
                    with open(log_file, 'a') as f:
                        f.write(f"Controller: {ctrl}, T={T_val}\n")
                        f.write(f"  Collision Rate: {collision_prob * 100:.2f}%\n")
                        f.write(f"  Average Speed: {avg_speed:.2f}\n")
                        f.write(f"  Average Calc Time: {avg_time * 1000:.3f} ms\n")
                        f.write("-" * 40 + "\n")
                    print(f"[{ctrl} T={T_val}] Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='batch', help='Mode of the simulation',
                        choices=['random', 'tubempc', 'vanillampc', 'cbf', 'vanillacbf', 'cpcbf', 'batch'])
    parser.add_argument('--display', type=bool, default=False, help='Display the animation')
    parser.add_argument('--T', type=int, default=1, help='Prediction horizon T')
    args = parser.parse_args()

    main(args)
