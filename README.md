# SPARC: A Safe Prediction-Based Robust Controller with Dynamically Coupled Uncontrollable Agents

This project implements a vehicle-pedestrian interaction safety control system based on Control Barrier Functions (CBF), combining neural network prediction and Conformal Prediction techniques to ensure safe vehicle operation in complex traffic environments.

## 📖 Project Overview

The system achieves safe vehicle-pedestrian interaction through the following core technologies:

- **Neural Network Predictor**: Predicts pedestrian motion trajectories
- **Control Barrier Functions (CBF)**: Ensures vehicles maintain safe distances from pedestrians
- **Conformal Prediction**: Quantifies prediction uncertainty and dynamically adjusts safety boundaries
- **Social Force Model Simulator**: Simulates realistic pedestrian behavior patterns
---
## 🎥 Demo


<p align="center">
  <img src="./demo_HD/demo_HD.gif" width="400"/>
</p>

---
## Project Structure

```
├── cbf/                          # Control Barrier Function controllers
│   └── current_cbf_controller.py # Main CBF controller implementation
├── models/                       # Neural network models
│   ├── model_def.py             # Model definitions
│   ├── predictor.py             # Predictor implementation
│   └── conformal_grid.py        # Conformal prediction grid
├── envs/                        # Environment simulation
│   ├── dynamics_social_force.py # Social force dynamics model
│   ├── dynamics.py              # Basic dynamics model
│   └── simulator.py             # Simulator
├── training/                    # Model training
│   └── train_walker_predictor.py
├── visualization/               # Visualization tools
│   ├── carla_demo_region.py     # CARLA demo visualization
│   ├── multi_ped_results.csv    # Multi-pedestrian results
│   └── visualize_model_performance.py
├── assets/                      # Pre-trained models and resources
│   └── best_model.pth          # Best prediction model
└── results/                     # Experimental results
```

## Core Components

### 🚗 CBF Controller (`cbf/current_cbf_controller.py`)

This is the core controller of the project, implementing safety control based on Control Barrier Functions:

#### Main Functions
- **Single Pedestrian Control**: `cbf_controller()` - Handles safety control for a single pedestrian
- **Multi-Pedestrian Control**: `cbf_controller_multi_pedestrian()` - Handles safety control for multiple pedestrians
- **Uncertainty Handling**: Supports conformal prediction uncertainty quantification
- **Real-time Optimization**: Uses SLSQP and OSQP solvers for real-time optimization

 


### Neural Network Predictor

#### Model Architecture
- **Basic Network**: Multi-layer perceptron with batch normalization and dropout
- **Residual Network**: Uses residual connections to improve training stability
- **Attention Mechanism**: Multi-head self-attention mechanism for enhanced feature extraction

#### Input/Output
- **Input**: 7-dimensional vector `[car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy]`
- **Output**: 2-dimensional vector `[next_walker_vx, next_walker_vy]`

### Social Force Model

Based on Helbing and Molnár's social force model, simulates realistic pedestrian behavior:
- **Goal-directed force**: Guides pedestrians toward their destination
- **Social force**: Avoids collisions with other pedestrians
- **Physical force**: Handles physical contacts

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

### Running Simulations


#### Multi-Pedestrian Scenario Testing
```bash
python evalcbf_multi.py --sample_num 500 --max_steps 10000
```

### Model Training

Train pedestrian behavior prediction model:
```bash
python training/train_walker_predictor.py
```

### video demo
```bash
python visualization/carla_demo_region.py
```


## Technical Features

### 1. Real-time Performance
- Optimization solver time limit (0.1 seconds)
- Efficient neural network inference
- Parallel processing support

### 2. Safety
- Strict mathematical safety guarantees
- Multiple safety check mechanisms
- Conservative control strategies

### 3. Robustness
- Uncertainty quantification and handling
- Multiple solver support
- Exception handling

### 4. Scalability
- Modular design
- Support for multi-pedestrian scenarios
- Easy integration of new prediction models

