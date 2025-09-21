import pygame
import numpy as np
import torch
from mpc.ped_dynamics import forward_ped
from mpc.car_dynamics import forward_car
from mpc.tube_utils import is_tube_safe

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
# pygame.display.set_caption("Reachable Tube Visualization")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 22)

# Set parameters
y0 = np.array([400.0, 200.0])   # walker initial position
x0 = 370.0                      # car starting point
T = 10                          # prediction steps

# Control sequence
u_seq = np.clip(np.random.normal(loc=5.0, scale=1.5, size=T), 0.0, 15.0)
x_seq = forward_car(x0, u_seq)
tube = forward_ped(y0, u_seq, cp_alpha=0.85)
tube_centers = [(lo + hi) / 2 for lo, hi in tube]
ped_positions = [y0] + tube_centers
car_positions = [(x, 12.0) for x in x_seq]

# Main loop
step = 0
running = True
while running:
    clock.tick(1)  # 1 frame per second, slow display
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))  # White background

    # ✅ 背景地图
    pygame.draw.rect(screen, (169, 169, 169), (0, 250, 800, 200))  # 灰色道路
    for i in range(5):
        pygame.draw.rect(screen, (255, 255, 255), (350, 250 + i * 40, 100, 20))  # 斑马线

    # 🚶 画 walker tube（step 前的）
    for t in range(step):
        lo, hi = tube[t]
        width = hi[0] - lo[0]
        height = hi[1] - lo[1]
        rect = pygame.Rect(lo[0], lo[1], width, height)
        pygame.draw.rect(screen, (0, 0, 255), rect, width=0)       # 蓝色填充
        pygame.draw.rect(screen, (0, 0, 0), rect, width=1)         # 黑边框

    # 🚶 tube 中心轨迹
    for i in range(1, step + 1):
        prev = ped_positions[i - 1]
        curr = ped_positions[i]
        pygame.draw.line(screen, (0, 0, 200), prev, curr, 3)
        pygame.draw.circle(screen, (0, 0, 200), curr.astype(int), 5)

    # 🚗 car 轨迹
    for i in range(1, step + 1):
        prev = car_positions[i - 1]
        curr = car_positions[i]
        pygame.draw.line(screen, (200, 0, 0), prev, curr, 3)
        pygame.draw.rect(screen, (200, 0, 0), pygame.Rect(curr[0] - 4, curr[1] - 4, 8, 8))

    # 📊 当前 car speed 控制率
    if step < len(u_seq):
        speed_text = font.render(f"Car Speed: {u_seq[step]:.2f}", True, (0, 0, 0))
        screen.blit(speed_text, (10, 10))

    pygame.display.flip()

    step += 1
    if step > T:
        pygame.time.wait(2000)
        running = False

pygame.quit()
