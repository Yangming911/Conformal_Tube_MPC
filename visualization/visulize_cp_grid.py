from visualizer.conformal_viz import plot_eta_vs_car_speed

def main():
    print("📊 Visualizing CP Grid...")

    # First half (walker_y < 300), η_y changes with car speed
    plot_eta_vs_car_speed(dim='y', vy_bin=1, walker_bin=0)

    # Second half, η_y changes with car speed
    plot_eta_vs_car_speed(dim='y', vy_bin=1, walker_bin=1)

    # First half, η_x
    plot_eta_vs_car_speed(dim='x', vy_bin=1, walker_bin=0)

    # Second half, η_x
    plot_eta_vs_car_speed(dim='x', vy_bin=1, walker_bin=1)

    print("✅ Done. You should see 4 plots.")

if __name__ == '__main__':
    main()
