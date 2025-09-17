# Vehicle-Pedestrian Interaction Safety Control System

This project implements a vehicle-pedestrian interaction safety control system based on Control Barrier Functions (CBF), combining neural network prediction and Conformal Prediction techniques to ensure safe vehicle operation in complex traffic environments.

## Project Overview

The system achieves safe vehicle-pedestrian interaction through the following core technologies:

- **Neural Network Predictor**: Predicts pedestrian motion trajectories
- **Control Barrier Functions (CBF)**: Ensures vehicles maintain safe distances from pedestrians
- **Conformal Prediction**: Quantifies prediction uncertainty and dynamically adjusts safety boundaries
- **Social Force Model**: Simulates realistic pedestrian behavior patterns

## Core Features

### 1. Intelligent Pedestrian Behavior Prediction
- Deep learning-based pedestrian speed prediction models
- Support for multiple network architectures (basic networks, residual networks, attention mechanisms)
- Real-time prediction of pedestrian next-step motion states

### 2. Safety Control Barrier Functions
- Quadratic programming-based CBF controllers
- Support for single and multi-pedestrian scenarios
- Dynamic safety distance adjustment
- Real-time collision detection and avoidance

### 3. Uncertainty Quantification
- Conformal prediction techniques to quantify prediction uncertainty
- Dynamic adjustment of safety boundary parameters
- Enhanced system robustness

## Project Structure

```
├── cbf/                          # Control Barrier Function controllers
│   ├── current_cbf_controller.py # Main CBF controller implementation
│   ├── cp_cbf_controller.py      # Conformal prediction CBF controller
│   └── vanilla_cbf_controller.py # Basic CBF controller
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
├── analysis/                    # Performance analysis
│   ├── analyze_model_performance.py
│   └── compare_models.py
├── visualization/               # Visualization tools
│   ├── plot_cbf_beta_analysis.py
│   ├── plot_cbf_beta_comparison.py
│   └── visualize_model_performance.py
├── assets/                      # Pre-trained models and resources
│   ├── best_model.pth          # Best prediction model
│   └── *.pkl                   # Conformal prediction grid data
└── results/                     # Experimental results
```

## Core Components

### CBF Controller (`cbf/current_cbf_controller.py`)

This is the core controller of the project, implementing safety control based on Control Barrier Functions:

#### Main Functions
- **Single Pedestrian Control**: `cbf_controller()` - Handles safety control for a single pedestrian
- **Multi-Pedestrian Control**: `cbf_controller_multi_pedestrian()` - Handles safety control for multiple pedestrians
- **Uncertainty Handling**: Supports conformal prediction uncertainty quantification
- **Real-time Optimization**: Uses SLSQP and OSQP solvers for real-time optimization

#### Key Parameters
- `d_safe`: Safety distance (default 2.5 meters)
- `gamma`: CBF parameter controlling the strictness of safety boundaries
- `use_eta`: Whether to use conformal prediction uncertainty adjustment
- `cp_alpha`: Confidence level for conformal prediction

#### Usage Example
```python
from cbf.current_cbf_controller import cbf_controller

# Define system state
state = {
    "car_x": 10.0,      # Vehicle x coordinate
    "car_y": 12.0,      # Vehicle y coordinate  
    "car_v": 8.0,       # Vehicle velocity
    "walker_x": 15.0,   # Pedestrian x coordinate
    "walker_y": 8.0,    # Pedestrian y coordinate
    "walker_vx": 1.0,   # Pedestrian x-direction velocity
    "walker_vy": 0.5    # Pedestrian y-direction velocity
}

# Calculate safe control input
control_input = cbf_controller(
    state=state,
    d_safe=2.5,         # Safety distance
    gamma=1.0,          # CBF parameter
    use_eta=True,       # Use uncertainty adjustment
    cp_alpha=0.85       # Confidence level
)
```

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

## Installation and Usage

### Requirements
- Python 3.7+
- PyTorch 1.13+
- CVXPY 1.1+
- NumPy, SciPy, Matplotlib
- Pygame (for visualization)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Running Simulations

#### Batch Simulation Testing
```bash
python main.py --T 1000  # Run 1000 simulations
```

#### Visual Simulation
```bash
python main.py --display  # Launch visualization interface
```

#### CBF Controller Evaluation
```bash
python evalcbf.py --sample_num 1000 --max_steps 10000
```

#### Multi-Pedestrian Scenario Testing
```bash
python evalcbf_multi.py --sample_num 500 --max_steps 10000
```

### Model Training

Train pedestrian behavior prediction model:
```bash
python training/train_walker_predictor.py
```

## Experimental Results

The system has been comprehensively tested in various scenarios:

### Performance Metrics
- **Collision Rate**: Probability of collision in 1000 simulations
- **Average Passage Time**: Average time for vehicles to pass through scenarios
- **Safety Distance Maintenance**: Minimum safe distance from pedestrians

### Comparative Experiments
- **CBF Controller** vs **Constant Speed**: Significantly reduces collision rate
- **With Uncertainty Adjustment** vs **Without Adjustment**: Improves system robustness
- **Single Pedestrian** vs **Multi-Pedestrian**: Validates performance in multi-agent scenarios

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

## File Descriptions

### Main Scripts
- `main.py`: Main program entry point
- `evalcbf.py`: CBF controller single pedestrian evaluation
- `evalcbf_multi.py`: CBF controller multi-pedestrian evaluation


### Configuration Files
- `utils/constants.py`: System parameter configuration
- `requirements.txt`: Dependency package list

<!-- ## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request -->

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## Contact

For questions or suggestions, please contact us through:
- Submit an Issue
- Send email to project maintainers

---

**Note**: This project focuses on vehicle-pedestrian interaction safety control and does not include Model Predictive Control (MPC) related content.
