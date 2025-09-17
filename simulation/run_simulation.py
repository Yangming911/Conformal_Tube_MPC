from models.predictor import WalkerActionPredictor
import numpy as np

def run_visual_sim():
    print("🚧 Visualization mode not yet implemented here.")

def run_batch_sim(n=10000, collision_threshold=20):
    model = WalkerActionPredictor()
    collision_count = 0

    for _ in range(n):
        car_x = -40
        car_speed = np.random.uniform(0, 15)
        walker_x = 400
        walker_y = 200

        for _ in range(100):
            car_x += car_speed
            if car_x > 800:
                break

            walker_input = (car_speed, walker_y)
            vx, vy = model.predict(*walker_input)
            walker_x += vx
            walker_y += vy

            if abs(walker_x - car_x) < collision_threshold and abs(walker_y - 320) < collision_threshold:
                collision_count += 1
                break

    prob = collision_count / n
    print(f"[Model Prediction] Collision probability over {n} trials: {prob:.4f}")
