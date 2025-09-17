from visualizer.conformal_viz import plot_eta_vs_car_speed

def main():
    print("📊 Visualizing CP Grid...")

    # 前半段（walker_y < 300），η_y 随 car speed 的变化
    plot_eta_vs_car_speed(dim='y', vy_bin=1, walker_bin=0)

    # 后半段，η_y 随 car speed 的变化
    plot_eta_vs_car_speed(dim='y', vy_bin=1, walker_bin=1)

    # 前半段，η_x
    plot_eta_vs_car_speed(dim='x', vy_bin=1, walker_bin=0)

    # 后半段，η_x
    plot_eta_vs_car_speed(dim='x', vy_bin=1, walker_bin=1)

    print("✅ Done. You should see 4 plots.")

if __name__ == '__main__':
    main()
