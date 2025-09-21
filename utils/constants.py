CAR_LEFT_LIMIT = 0
CAR_RIGHT_LIMIT = 50
CAR_START_X = 0
CAR_LANE_Y = 12


WALKER_START_X = 30
WALKER_START_Y = 0
WALKER_DESTINATION_Y = 20
WALKER_START_V_X = 0
WALKER_START_V_Y = 0

a_max = 5
v_max = 2.5
dt = 0.1
walker_noise_y_sigma = 0.5
walker_noise_x_sigma = 0.1
mass_pedestrian = 80
num_pedestrians = 3  # Default to use 3 pedestrians for testing


__all__ = [
    "CAR_LEFT_LIMIT",
    "CAR_RIGHT_LIMIT",
    "CAR_START_X",
    "CAR_LANE_Y",
    "WALKER_START_X",
    "WALKER_START_Y",
    "WALKER_DESTINATION_Y",
    "WALKER_START_V_X",
    "WALKER_START_V_Y",
    "a_max",
    "v_max",
    "dt",
    "walker_noise_y_sigma",
    "walker_noise_x_sigma",
    "mass_pedestrian",
    "num_pedestrians",
]   


