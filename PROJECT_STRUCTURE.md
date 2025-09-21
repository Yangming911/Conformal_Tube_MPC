# Project Structure Documentation

## 📁 Directory Structure

```
icra_2026/
├── analysis/                    # Model analysis scripts
│   ├── analyze_model_performance.py    # Detailed performance analysis
│   └── compare_models.py               # Model comparison analysis
├── assets/                      # Resource files
│   ├── *.pth                    # Trained model files
│   ├── *.pkl                    # Preprocessed grid data
│   ├── *.csv                    # Dataset files
│   └── *.json                   # Training result files
├── cbf/                         # Control Barrier Function controllers
│   ├── cp_cbf_controller.py
│   ├── current_cbf_controller.py
│   └── vanilla_cbf_controller.py
├── demo_HD/                     # Demo files
│   ├── demo_HD_cover.jpg        # Demo cover image
│   └── demo_HD.mp4              # Demo video
├── envs/                        # Environment definitions
│   ├── dynamics_social_force.py # Social force dynamics
│   ├── dynamics.py              # Basic dynamics
│   └── simulator.py             # Simulator
├── logs/                        # Log and output files
│   ├── *.log                    # Training and runtime logs
│   ├── *.png                    # Generated images
│   ├── *.csv                    # Analysis data
│   └── *.md                     # Analysis reports
├── models/                      # Model definitions
│   ├── conformal_grid.py        # Conformal grid
│   ├── model_def.py             # Neural network model definition
│   └── predictor.py             # Model predictor
├── mpc/                         # Model Predictive Control
│   ├── car_dynamics.py          # Car dynamics
│   ├── ped_dynamics.py          # Pedestrian dynamics
│   ├── tube_utils.py            # Tube MPC utilities
│   ├── tubempc_controller.py    # Tube MPC controller
│   └── vanillampc_controller.py # Standard MPC controller
├── results/                     # Experimental results
│   ├── *.png                    # Result images
│   ├── *.pdf                    # Result documents
│   └── *.gif                    # Animation files
├── simulation/                  # Simulation scripts
│   └── run_simulation.py        # Run simulation
├── tools/                       # Utility tools
│   ├── identify_best_model.py   # Model identification tool
│   └── run_evalcbf_multi_batch.py # Batch evaluation runner
├── training/                    # Training scripts
│   ├── test_model.py            # Model testing
│   └── train_walker_predictor.py # Model training
├── utils/                       # Utility functions
│   └── constants.py             # Constant definitions
├── visualization/               # Visualization scripts
│   ├── carla_demo_region.py     # CARLA demo visualization
│   ├── multi_ped_results.csv    # Multi-pedestrian results
│   ├── plot_cbf_beta_analysis.py # CBF beta analysis plots
│   ├── plot_cbf_beta_comparison.py # CBF beta comparison plots
│   ├── plot_multi_ped_comparison.py # Multi-pedestrian comparison plots
│   ├── pygame_tube_viz.py       # Pygame tube visualization
│   ├── visualize_bin_density.py # Bin density visualization
│   ├── visualize_model_performance.py # Model performance visualization
│   ├── visualize_prediction_vs_speed.py # Prediction vs speed visualization
│   └── visulize_cp_grid.py      # Conformal prediction grid visualization
├── visualizer/                  # Visualization tools
│   └── conformal_viz.py         # Conformal visualization
├── evalAPF_multi.py             # Multi-pedestrian APF evaluation
├── evalcbf_multi.py             # Multi-pedestrian CBF evaluation
├── find_max_errors.py           # Error analysis script
├── main.py                      # Main program entry
├── requirements.txt             # Dependency list
├── README.md                    # Project description
└── PROJECT_STRUCTURE.md         # This file
```

## 🚀 Usage

### Training Models
```bash
# Enter training directory
cd training

# Train basic model
python train_walker_predictor.py --model WalkerSpeedPredictor --epochs 50

# Train residual+attention model
python train_walker_predictor.py --model WalkerSpeedPredictorV2 --epochs 50

# Test model
python test_model.py
```

### Analyze Model Performance
```bash
# Enter analysis directory
cd analysis

# Detailed performance analysis
python analyze_model_performance.py

# Model comparison analysis
python compare_models.py
```

### Visualize Results
```bash
# Enter visualization directory
cd visualization

# Model performance visualization
python visualize_model_performance.py
```

### Run Simulation
```bash
# In project root directory
python main.py
```

### Evaluate Controllers
```bash
# Multi-pedestrian CBF evaluation
python evalcbf_multi.py --sample_num 1000 --num_pedestrians 3

# Multi-pedestrian APF evaluation
python evalAPF_multi.py --sample_num 1000 --num_pedestrians 3

# Batch evaluation with different pedestrian counts
python tools/run_evalcbf_multi_batch.py --nums 1,3,5,7,9 --quick
```

### Error Analysis
```bash
# Find maximum errors in calibration dataset
python find_max_errors.py
```

## 📂 File Descriptions

### Core Files
- `main.py`: Main program entry, runs complete simulation
- `models/model_def.py`: Neural network model definition (WalkerSpeedPredictor, WalkerSpeedPredictorV2)
- `models/predictor.py`: Model predictor interface with legacy and new format support
- `models/conformal_grid.py`: Conformal prediction grid implementation
- `envs/simulator.py`: Multi-pedestrian simulation environment
- `envs/dynamics.py`: Basic dynamics implementation
- `envs/dynamics_social_force.py`: Social force model dynamics

### Controller Implementations
- `cbf/current_cbf_controller.py`: Current CBF controller with conformal prediction
- `cbf/cp_cbf_controller.py`: Conformal prediction CBF controller
- `cbf/vanilla_cbf_controller.py`: Vanilla CBF controller
- `mpc/tubempc_controller.py`: Tube-based MPC controller
- `mpc/vanillampc_controller.py`: Vanilla MPC controller
- `mpc/car_dynamics.py`: Car dynamics for MPC
- `mpc/ped_dynamics.py`: Pedestrian dynamics for MPC
- `mpc/tube_utils.py`: Tube MPC utility functions

### Training Related
- `training/train_walker_predictor.py`: Model training script with support for multiple architectures
- `training/test_model.py`: Model testing and loading verification
- `assets/*.pth`: Trained model files (walker_speed_predictor.pth, walker_speed_predictor_v2_fixed.pth, etc.)
- `assets/*.json`: Training results and metrics

### Evaluation Scripts
- `evalcbf_multi.py`: Multi-pedestrian CBF controller evaluation
- `evalAPF_multi.py`: Multi-pedestrian APF controller evaluation
- `find_max_errors.py`: Calibration dataset error analysis
- `tools/run_evalcbf_multi_batch.py`: Batch evaluation runner for multiple pedestrian counts

### Analysis Related
- `analysis/analyze_model_performance.py`: Detailed model performance analysis
- `analysis/compare_models.py`: Model comparison analysis between different architectures
- `logs/*.png`: Analysis result images and plots
- `logs/*.csv`: Analysis data and simulation results
- `logs/*.md`: Analysis reports and summaries

### Visualization Related
- `visualization/carla_demo_region.py`: CARLA demo with CBF controller integration
- `visualization/visualize_model_performance.py`: Model performance visualization
- `visualization/visualize_prediction_vs_speed.py`: Prediction vs speed analysis plots
- `visualization/pygame_tube_viz.py`: Pygame-based tube visualization
- `visualization/plot_cbf_beta_analysis.py`: CBF beta parameter analysis plots
- `visualization/plot_multi_ped_comparison.py`: Multi-pedestrian comparison plots
- `visualizer/conformal_viz.py`: Conformal prediction grid visualization
- `results/*.png`: Experimental result images and plots
- `results/*.pdf`: Result documents and papers
- `results/*.gif`: Animation files

### Demo and Media
- `demo_HD/demo_HD_cover.jpg`: Demo cover image
- `demo_HD/demo_HD.mp4`: Demo video file

## 🔧 Path Configuration

All scripts are configured with paths relative to the project root directory:
- Training scripts use `../assets/` to access resource files
- Analysis scripts use `../logs/` to save output files
- Visualization scripts use `../logs/` to save images

## 📝 Notes

1. Make sure you are in the correct directory when running scripts
2. All output files will be saved to the `logs/` directory
3. Model files are saved in the `assets/` directory
4. Experimental results are saved in the `results/` directory
