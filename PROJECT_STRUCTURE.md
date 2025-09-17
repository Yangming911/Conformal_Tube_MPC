# 项目结构说明

## 📁 目录结构

```
cp_mpc_coupled_adjust/
├── analysis/                    # 模型分析脚本
│   ├── analyze_model_performance.py    # 详细性能分析
│   └── compare_models.py               # 模型对比分析
├── assets/                      # 资源文件
│   ├── *.pth                    # 训练好的模型文件
│   ├── *.pkl                    # 预处理的网格数据
│   ├── *.csv                    # 数据集文件
│   └── *.json                   # 训练结果文件
├── cbf/                         # 控制屏障函数控制器
│   ├── cp_cbf_controller.py
│   ├── current_cbf_controller.py
│   └── vanilla_cbf_controller.py
├── envs/                        # 环境定义
│   ├── dynamics_social_force.py # 社会力动力学
│   ├── dynamics.py              # 基础动力学
│   └── simulator.py             # 仿真器
├── logs/                        # 日志和输出文件
│   ├── *.log                    # 训练和运行日志
│   ├── *.png                    # 生成的图片
│   ├── *.csv                    # 分析数据
│   └── *.md                     # 分析报告
├── models/                      # 模型定义
│   ├── conformal_grid.py        # 保形网格
│   ├── model_def.py             # 神经网络模型定义
│   └── predictor.py             # 模型预测器
├── mpc/                         # 模型预测控制
│   ├── car_dynamics.py          # 车辆动力学
│   ├── ped_dynamics.py          # 行人动力学
│   ├── tube_utils.py            # 管状MPC工具
│   ├── tubempc_controller.py    # 管状MPC控制器
│   └── vanillampc_controller.py # 标准MPC控制器
├── results/                     # 实验结果
│   ├── *.png                    # 结果图片
│   └── *.pdf                    # 结果文档
├── scripts/                     # 工具脚本
│   └── gen_cp_grid.py           # 生成保形网格
├── simulation/                  # 仿真脚本
│   └── run_simulation.py        # 运行仿真
├── training/                    # 训练脚本
│   ├── test_model.py            # 模型测试
│   └── train_walker_predictor.py # 模型训练
├── utils/                       # 工具函数
│   └── constants.py             # 常量定义
├── visualization/               # 可视化脚本
│   ├── visualize_bin_density.py
│   ├── visualize_model_performance.py
│   ├── visualize_prediction_vs_speed.py
│   └── visulize_cp_grid.py
├── visualizer/                  # 可视化工具
│   └── conformal_viz.py         # 保形可视化
├── evaluation/                  # 评估目录
├── main.py                      # 主程序入口
├── eval.py                      # 评估脚本
├── evalcbf.py                   # CBF评估脚本
├── generate_cp_grid.py          # 生成保形网格
├── pygame_tube_viz.py           # Pygame可视化
├── tube_test.py                 # 管状测试
├── requirements.txt             # 依赖包列表
├── LICENSE                      # 许可证
├── README.md                    # 项目说明
└── PROJECT_STRUCTURE.md         # 本文件
```

## 🚀 使用方法

### 训练模型
```bash
# 进入训练目录
cd training

# 训练基础模型
python train_walker_predictor.py --model WalkerSpeedPredictor --epochs 50

# 训练残差+注意力模型
python train_walker_predictor.py --model WalkerSpeedPredictorV2 --epochs 50

# 测试模型
python test_model.py
```

### 分析模型性能
```bash
# 进入分析目录
cd analysis

# 详细性能分析
python analyze_model_performance.py

# 模型对比分析
python compare_models.py
```

### 可视化结果
```bash
# 进入可视化目录
cd visualization

# 模型性能可视化
python visualize_model_performance.py
```

### 运行仿真
```bash
# 在项目根目录
python main.py
```

## 📂 文件说明

### 核心文件
- `main.py`: 主程序入口，运行完整的仿真
- `models/model_def.py`: 神经网络模型定义
- `models/predictor.py`: 模型预测器接口
- `envs/simulator.py`: 仿真环境

### 训练相关
- `training/train_walker_predictor.py`: 模型训练脚本
- `training/test_model.py`: 模型测试脚本
- `assets/*.pth`: 训练好的模型文件

### 分析相关
- `analysis/analyze_model_performance.py`: 详细性能分析
- `analysis/compare_models.py`: 模型对比分析
- `logs/*.png`: 分析结果图片
- `logs/*.csv`: 分析数据

### 可视化相关
- `visualization/visualize_model_performance.py`: 模型性能可视化
- `results/*.png`: 实验结果图片

## 🔧 路径配置

所有脚本都已配置为相对于项目根目录的路径：
- 训练脚本使用 `../assets/` 访问资源文件
- 分析脚本使用 `../logs/` 保存输出文件
- 可视化脚本使用 `../logs/` 保存图片

## 📝 注意事项

1. 运行脚本时请确保在正确的目录中
2. 所有输出文件都会保存到 `logs/` 目录
3. 模型文件保存在 `assets/` 目录
4. 实验结果保存在 `results/` 目录
