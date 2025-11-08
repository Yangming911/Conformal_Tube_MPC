# Chance-Constrained Neural-Network MPC for Interactive Multi-Agent Systems via Sequential Convex Programming

This project implements a vehicle-pedestrian interaction safety control system based on **Sequential Convex Programming (SCP)** with **Conformal Prediction**, combining neural network-based pedestrian trajectory prediction and robust optimization techniques to ensure safe vehicle operation in complex traffic environments.

## 📖 Project Overview

The system achieves safe vehicle-pedestrian interaction through the following core technologies:

- **Causal Sequence Predictor**: GRU-based neural network that predicts pedestrian trajectories causally based on vehicle control inputs
- **Conformal Prediction**: Quantifies prediction uncertainty and provides statistically valid safety regions
- **Sequential Convex Programming (SCP)**: Iteratively solves the non-convex optimal control problem through convex subproblems
- **Trust Region Method**: Ensures convergence and robustness with exponentially decaying step size limits
- **Social Force Model Simulator**: Simulates realistic pedestrian behavior patterns

---

<!-- ## 🎥 Demo

See the high-definition demo video at `demo_HD/demo_HD.mp4`.

<p align="center">
  <img src="./demo_HD/demo_HD.gif" width="400"/>
</p>

--- -->

## Project Structure

```
├── models_control/              # Control-oriented models and algorithms
│   ├── model_def.py            # Causal pedestrian predictor (GRU-based)
│   ├── train.py                # Training script for control model
│   ├── cp.py                   # Conformal prediction for uncertainty quantification
│   └── scp.py                  # Sequential convex programming optimizer
├── tools/                       # Utilities and evaluation tools
│   ├── collect_control_sequences.py  # Collect training data from simulator
│   └── eval_runs_scp.py        # Closed-loop MPC evaluation with SCP
├── models/                      # Original prediction models
│   ├── model_def.py            # Model definitions
│   ├── predictor.py            # Predictor implementation
│   └── conformal_grid.py       # Conformal prediction grid
├── envs/                       # Environment simulation
│   ├── dynamics_social_force.py # Social force dynamics model
│   ├── dynamics.py             # Basic dynamics model
│   └── simulator.py            # Simulator with single/multi-pedestrian support
├── training/                   # Model training
│   └── train_walker_predictor.py
├── visualization/              # Visualization tools
│   ├── carla_demo_region.py   # CARLA demo visualization
│   └── visualize_model_performance.py
├── assets/                     # Pre-trained models and data
│   ├── control_ped_model.pth  # Trained causal predictor
│   ├── cp_eta.csv             # Conformal prediction error bounds
│   └── control_sequences.csv  # Training sequences
├── logs/                       # Experiment logs
│   └── scp_eval.log           # SCP evaluation logs
└── results/                    # Experimental results
```

## Core Components

### 🚗 SCP Controller (`models_control/scp.py`)

This is the core controller of the project, implementing optimal control based on Sequential Convex Programming with conformal prediction safety guarantees:

#### Main Functions
- **`scp_optimize()`**: Main optimization function that computes optimal control sequence
  - **Outer Loop**: Updates conformal safety regions based on current control
  - **Inner Loop**: Iteratively linearizes constraints and solves QP subproblems with trust region
  - **Verification**: Validates solution against its own safety region
  - **Multi-Pedestrian Support**: Handles M×T inequality constraints simultaneously

#### Key Features
- **Trust Region Method**: Exponentially decaying step size (`R_k = R_0 × decay^k`) prevents large jumps
- **Binning Strategy**: Partitions control space (e.g., [0,5), [5,10), [10,15]) for adaptive safety bounds
- **Finite Difference Gradients**: Computes Jacobian of collision constraints w.r.t. control inputs
- **OSQP Solver**: Efficient convex quadratic programming for real-time performance
- **Reject Statistics**: Tracks bin transition failures for analysis

### 🧠 Causal Pedestrian Predictor (`models_control/model_def.py`)

#### Model Architecture
- **GRU-based Sequence Model**: Processes control sequence `u[0:T-1]` causally
- **Causality Constraint**: `p_ped[t]` depends only on `u[0:t-1]`, not future controls
- **Input**: 
  - Initial states: `p_veh_0` (2D), `p_ped_0` (2D)
  - Control sequence: `u[0:T-1]` (scalar per timestep)
- **Output**: Pedestrian position sequence `p_ped[1:T]` (2D per timestep)

#### Training
```bash
# Collect training data from Social Force simulator
python tools/collect_control_sequences.py --episodes 20000 --T 10

# Train the causal predictor
python models_control/train.py --data assets/control_sequences.csv --epochs 100
```

### 📊 Conformal Prediction (`models_control/cp.py`)

Provides statistically valid uncertainty bounds for safe control:

#### Process
1. **Calibration Data**: Collect sequences from simulator with the trained model
2. **Error Computation**: Calculate L2 prediction errors `||y_true - y_pred||` for each timestep
3. **Binning**: Partition by vehicle speed into bins (e.g., 3 bins: low/mid/high speed)
4. **Quantile Calculation**: Compute α-quantile (e.g., 95%) for each (timestep, bin) pair
5. **Output**: `eta[t, bin]` matrix used as safety radii in SCP

```bash
# Generate conformal prediction bounds
python models_control/cp.py --alpha 0.95 --num_bins 3 --calib_episodes 1000
```

### 🎯 Closed-Loop Evaluation (`tools/eval_runs_scp.py`)

Model Predictive Control (MPC) evaluation in closed-loop with the simulator:

#### Features
- **Receding Horizon**: Re-plans control every T steps based on current state
- **Multi-Episode Testing**: Runs 200+ full trajectories with random initializations
- **Multi-Pedestrian Support**: Configurable number of pedestrians (1 real + N-1 virtual)
- **Comprehensive Metrics**:
  - Average vehicle speed
  - Collision rate
  - Avg outer-loop iterations per plan
  - Avg inner SCP steps per outer iteration
  - Planning computation time
  - Bin transition statistics (accepted/rejected)

```bash
# Run evaluation with default settings (200 episodes, 1 pedestrian)
python tools/eval_runs_scp.py

# Multi-pedestrian scenario (3 pedestrians)
python tools/eval_runs_scp.py --num_pedestrians 3 --episodes 100

# Adjust trust region parameters
python tools/eval_runs_scp.py --trust_region_initial 3.0 --trust_region_decay 0.6
```

### Social Force Model

Based on Helbing and Molnár's social force model, simulates realistic pedestrian behavior:
- **Goal-directed force**: Guides pedestrians toward their destination
- **Vehicle repulsion force**: Avoids collisions with vehicles (situation-dependent)
- **Stochastic perturbations**: Models natural human motion variability

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch (neural network)
- NumPy, Pandas (data handling)
- CVXPY with OSQP (convex optimization)
- tqdm (progress bars)

## 🚀 Quick Start

### 1. Data Collection
```bash
python tools/collect_control_sequences.py --episodes 20000 --T 10
```

### 2. Train Causal Predictor
```bash
python models_control/train.py --data assets/control_sequences.csv
```

### 3. Generate Conformal Bounds
```bash
python models_control/cp.py --alpha 0.95 --num_bins 3
```

### 4. Evaluate SCP Controller
```bash
python tools/eval_runs_scp.py --episodes 200 --num_pedestrians 1
```

## 📈 Technical Features

### 1. Real-time Performance
- Efficient QP solving with OSQP (typically <1 second per T-step plan)
- Fast neural network inference with PyTorch
- Parallel inner SCP iterations converge rapidly (avg 1-4 steps)

### 2. Safety Guarantees
- **Probabilistic Safety**: (1-α) coverage guarantee from conformal prediction
- **Constraint Verification**: Solutions validated against their own safety regions
- **Conservative Rejection**: Returns safe fallback control if verification fails

### 3. Robustness
- **Trust Region**: Prevents optimizer from making unrealistic jumps
- **Adaptive Binning**: Safety bounds adapt to vehicle speed regime
- **Multi-Pedestrian**: Handles multiple concurrent constraints (M×T total)

### 4. Scalability
- Modular design: predictor, conformal prediction, and SCP are independent
- Supports 1 to N pedestrians (computational cost scales linearly)
- Flexible horizon length T (default: 10 steps)

## 📝 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T` | 10 | Planning horizon (time steps) |
| `outer_iters` | 5 | Max outer-loop iterations (C updates) |
| `trust_region_initial` | 5.0 | Initial trust region radius |
| `trust_region_decay` | 0.5 | Trust region decay rate per inner iteration |
| `alpha` | 0.95 | Conformal prediction confidence level |
| `num_bins` | 3 | Number of speed bins for conformal prediction |
| `d_safe` | 1.0 | Safety distance margin (meters) |
| `u_min` / `u_max` | 0.0 / 15.0 | Control input bounds (m/s) |
| `num_pedestrians` | 1 | Number of pedestrians in constraints |

## 📊 Experiment Logs

All experiments are automatically logged to `logs/scp_eval.log` with:
- Experiment start/end timestamps
- All parameter values
- Reject messages with violated timestep details
- CVXPY warnings (if any)
- Final statistics (collision rate, speed, etc.)

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{wang2025chanceconstrainedneuralmpcuncontrollable,
      title={Chance-Constrained Neural MPC under Uncontrollable Agents via Sequential Convex Programming}, 
      author={Shuqi Wang and Mingyang Feng and Yu Chen and Yue Gao and Xiang Yin},
      year={2025},
      eprint={2504.03293},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2504.03293}, 
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
