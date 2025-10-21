import numpy as np
from scipy.optimize import minimize

# 定义问题参数
circle_center = np.array([2.0, 3.0])  # 圆心坐标
circle_radius = 1.0  # 圆的半径

# 定义目标函数: 最小化点到原点的距离
def objective(x):
    return np.linalg.norm(x)  # 计算欧几里得距离

# 定义约束条件: 点在圆外
def constraint(x):
    # 约束函数在满足约束时应返回非负值
    # 我们希望 ||x - circle_center|| >= circle_radius
    # 所以约束函数为: ||x - circle_center|| - circle_radius >= 0
    return np.linalg.norm(x - circle_center) - circle_radius

# 设置约束条件
def_cons = {
    'type': 'ineq',  # 不等式约束: constraint(x) >= 0
    'fun': constraint
}

# 设置初始猜测值
# 我们可以选择从原点指向圆心的方向上、距离圆心半径位置的点
# 这很可能接近最优解
direction = circle_center / np.linalg.norm(circle_center)  # 单位方向向量
x0 = circle_center - direction * circle_radius  # 初始猜测值

# 求解优化问题
result = minimize(
    objective,      # 目标函数
    x0,             # 初始猜测值
    method='SLSQP', # 序列二次规划法，适合处理约束优化问题
    constraints=[def_cons],  # 约束条件
    options={'disp': True}  # 显示求解过程
)

# 输出结果
print("优化成功: ", result.success)
print("最优解 (点坐标): ", result.x)
print("最小距离: ", result.fun)
print("验证点是否在圆外: ", np.linalg.norm(result.x - circle_center) >= circle_radius - 1e-8)

# 可视化结果（可选）
try:
    import matplotlib.pyplot as plt
    
    # 绘制圆
    circle = plt.Circle(circle_center, circle_radius, fill=False, color='blue', linestyle='--')
    fig, ax = plt.subplots()
    ax.add_patch(circle)
    
    # 绘制原点
    ax.plot(0, 0, 'ro', label='原点')
    
    # 绘制最优解点
    ax.plot(result.x[0], result.x[1], 'go', label='最优解点')
    
    # 绘制连线
    ax.plot([0, result.x[0]], [0, result.x[1]], 'g--', label='到原点的距离')
    ax.plot([circle_center[0], result.x[0]], [circle_center[1], result.x[1]], 'r--', label='到圆心的距离')
    
    # 设置图形属性
    ax.set_aspect('equal')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 5)
    ax.grid(True)
    ax.legend()
    plt.title('点在圆外且最接近原点的优化问题 (scipy求解)')
    plt.savefig('scipy_optimization_result.png')
    plt.show()
    print("结果图已保存为 'scipy_optimization_result.png'")
except ImportError:
    print("无法进行可视化，请安装matplotlib库")