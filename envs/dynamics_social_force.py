import sys
from pathlib import Path

# Ensure project root is on sys.path so that `utils` can be imported when running this file directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.constants as C
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

# Vehicle parameters from the Social Force paper (Table 2)
l_r = 1.2      # rear length
l_f = 1.0      # front length  
l_w = 1.2      # width
l_e = 0.2151011    # extension length
d_ox = 0.510985    # extended length along vehicle orientation
alpha_x = 1.394358 # proportional factor for speed extension
# alpha_x = 1.8 # proportional factor for speed extension

def point_to_line_segment_distance(px, py, x1, y1, x2, y2):
    """Calculate closest point on line segment to a given point"""
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return x1, y1, (px - x1)**2 + (py - y1)**2
    
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    dist_sq = (px - closest_x)**2 + (py - closest_y)**2
    
    return closest_x, closest_y, dist_sq

def get_vehicle_contour_and_influence_point(car_x, car_y, car_v, walker_x, walker_y):
    """
    Get vehicle virtual contour and calculate influence point for a walker
    Returns: (contour_points, influence_point, situation_type)
    """
    # Calculate virtual contour dimensions
    total_width = l_w + 2*l_e
    rect_half_width = total_width / 2
    triangle_extension = alpha_x * abs(car_v)
    # Rectangle bounds relative to car center
    rect_left = -(l_r + l_e)
    rect_right = l_f + l_e + d_ox
    rect_top = rect_half_width
    rect_bottom = -rect_half_width
    
    # Walker position relative to car
    dx = walker_x - car_x
    dy = walker_y - car_y
    
    # Define contour points
    # Rectangle part
    rect_points = [
        [car_x + rect_left, car_y + rect_bottom],   # bottom-left
        [car_x + rect_left, car_y + rect_top],      # top-left
        [car_x + rect_right, car_y + rect_top],     # top-right
        [car_x + rect_right, car_y + rect_bottom],  # bottom-right
    ]
    
    # Triangle part (front extension)
    triangle_tip_x = car_x + rect_right + triangle_extension
    triangle_points = [
        [car_x + rect_right, car_y + rect_top],     # top-right of rectangle
        [triangle_tip_x, car_y],                    # triangle tip
        [car_x + rect_right, car_y + rect_bottom],  # bottom-right of rectangle
    ]
    
    # Complete contour
    contour_points = rect_points + triangle_points
    
    # Determine situation and find influence point
    situation_type = ""
    
    if dx > rect_right:
        # In front of vehicle (triangular area)
        triangle_tip_x_rel = rect_right + triangle_extension
        
        if dx <= triangle_tip_x_rel and triangle_extension > 0:
            # Check if inside triangle
            progress = (dx - rect_right) / triangle_extension
            y_top_at_x = rect_top * (1 - progress)
            y_bottom_at_x = rect_bottom * (1 - progress)
            
            if y_bottom_at_x <= dy <= y_top_at_x:
                situation_type = "Inside Triangle (Freezing)"
                influence_point = [walker_x, walker_y]  # No real influence point
                return contour_points, influence_point, situation_type
        
        # Outside triangle - find closest point on triangle perimeter
        situation_type = "Outside Triangle"
        
        # Calculate distances to three edges
        closest_x1, closest_y1, dist_sq1 = point_to_line_segment_distance(
            dx, dy, rect_right, rect_top, triangle_tip_x_rel, 0)
        closest_x2, closest_y2, dist_sq2 = point_to_line_segment_distance(
            dx, dy, rect_right, rect_bottom, triangle_tip_x_rel, 0)
        closest_x3, closest_y3, dist_sq3 = point_to_line_segment_distance(
            dx, dy, rect_right, rect_top, rect_right, rect_bottom)
        
        if dist_sq1 <= dist_sq2 and dist_sq1 <= dist_sq3:
            influence_point = [car_x + closest_x1, car_y + closest_y1]
        elif dist_sq2 <= dist_sq3:
            influence_point = [car_x + closest_x2, car_y + closest_y2]
        else:
            influence_point = [car_x + closest_x3, car_y + closest_y3]
    else:
        # In rectangular area: check if walker is inside the rectangle
        if rect_left <= dx <= rect_right and rect_bottom <= dy <= rect_top:
            situation_type = "Inside Rectangle (Freezing)"
            influence_point = [walker_x, walker_y]  # No real influence point
            return contour_points, influence_point, situation_type
        # Outside but laterally aligned with rectangle: project to rectangle edge
        situation_type = "Rectangular Area"
        closest_x = max(rect_left, min(rect_right, dx))
        closest_y = max(rect_bottom, min(rect_top, dy))
        influence_point = [car_x + closest_x, car_y + closest_y]
    
    return contour_points, influence_point, situation_type

def F_vehicle(car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy):
    """Calculate the vehicle influence force vector"""
    contour_points, influence_point, situation = get_vehicle_contour_and_influence_point(
        car_x, car_y, car_v, walker_x, walker_y)
    
    if situation == "Inside Triangle (Freezing)" or situation == "Inside Rectangle (Freezing)":
        return 0, 0, situation
    
    # Distance from walker to influence point
    d_iv = math.sqrt((walker_x - influence_point[0])**2 + (walker_y - influence_point[1])**2)
    
    if d_iv == 0:
        return 0, 0, situation
    
    # Unit vector from influence point to walker
    n_vi_x = (walker_x - influence_point[0]) / d_iv
    n_vi_y = (walker_y - influence_point[1]) / d_iv
    
    # Calculate anisotropy
    v_walker_mag = math.sqrt(walker_vx**2 + walker_vy**2)
    if v_walker_mag == 0:
        phi_iv = 0
    else:
        dot_product = (-n_vi_x * walker_vx + -n_vi_y * walker_vy) / v_walker_mag
        dot_product = max(-1, min(1, dot_product))
        phi_iv = math.acos(dot_product)
    
    # Force parameters from Table 2
    A_veh = 777.5852
    b_veh = 2.613755
    lambda_veh = 0.3119132
    
    # Calculate force
    f_exp_val = A_veh * math.exp(-b_veh * d_iv)
    A_sin_val = lambda_veh + (1 - lambda_veh) * (1 + math.cos(abs(phi_iv))) / 2
    force_magnitude = f_exp_val * A_sin_val
    
    F_veh_x = force_magnitude * n_vi_x
    F_veh_y = force_magnitude * n_vi_y
    
    return F_veh_x, F_veh_y, situation


def F_destination(walker_x_position, destination_x, walker_y_position, destination_y, F_veh_x, F_veh_y, walker_vx, walker_vy):
    """
    Compute the attractive force of the destination (force acts only along the y-axis).

    Parameters:
    - walker_y_position: current y-coordinate of the pedestrian

    Returns:
    - F_destination_y: destination attraction force along the y-axis
    """
    # Parameters for destination attraction
    # Parameters from Table 2
    k_des = 545.3125      # feedback gain
    v_0 = 1.394293      # desired speed
    sigma_des = 1.0       # parameter for speed adjustment near destination
    F_1 = 199.7455        # threshold parameter
    F_2 = 672.6487        # threshold parameter
    
    F_veh_magnitude = math.sqrt(F_veh_x**2 + F_veh_y**2)
    if F_veh_magnitude <= F_2:
        beta_des = 1.0
    elif F_veh_magnitude >= F_1:
        beta_des = 0.0
    else:
        # Linear interpolation between F_2 and F_1
        beta_des = 1.0 - (F_veh_magnitude - F_2) / (F_1 - F_2)
    
    # Ensure beta_des is in [0, 1]
    beta_des = max(0.0, min(1.0, beta_des))
    
    # Compute desired speed (v_d) pointing to the destination
    desired_velocity_y = v_0 * (destination_y - walker_y_position) / math.sqrt((destination_y - walker_y_position) ** 2 + sigma_des ** 2)
    desired_velocity_x = v_0 * (destination_x - walker_x_position) / math.sqrt((destination_x - walker_x_position) ** 2 + sigma_des ** 2)
    
    delta_v_y = desired_velocity_y - walker_vy
    delta_v_x = desired_velocity_x - walker_vx
    # Compute destination attraction force
    F_destination_y = beta_des * k_des * delta_v_y # adjust speed toward destination
    F_destination_x = beta_des * k_des * delta_v_x
    
    return F_destination_x, F_destination_y

def F_pedestrian(walker_x_position, walker_y_position, walker_v_x_past, walker_v_y_past, other_ped_x_position, other_ped_y_position, other_ped_v_x_past, other_ped_v_y_past):
    """
    Compute the repulsive force between a pedestrian and other pedestrians.

    Parameters:
    - walker_x_position: current x-coordinate of the pedestrian
    - walker_y_position: current y-coordinate of the pedestrian
    - walker_v_x_past: past x-velocity of the pedestrian
    - walker_v_y_past: past y-velocity of the pedestrian
    - other_ped_x_position: x-coordinates of other pedestrians
    - other_ped_y_position: y-coordinates of other pedestrians

    Returns:
    - F_pedestrian_x: repulsive force along the x-axis
    - F_pedestrian_y: repulsive force along the y-axis
    """
    alpha = 982.125
    repulsive_distance = 0.7801
    F_pedestrian_x = 0
    F_pedestrian_y = 0

    
    def flm(d_ij, d_ref, M, sigma):
        """Linear decaying function with smoothing"""
        if d_ij <= d_ref:
            return M * (1 - (d_ij / d_ref))
        else:
            return M * sigma * math.exp(-(d_ij - d_ref) / sigma)


    def Asin(phi_ij, lambda_param):
        """Sine anisotropy function"""
        return lambda_param + (1 - lambda_param) * (1 + math.cos(abs(phi_ij))) / 2


    def Aexp(phi_ij_v, lambda_param):
        """Exponential anisotropy function"""
        return math.exp(-phi_ij_v**2 / (2 * lambda_param**2))

    def virtual_interaction_force(walker_x, walker_y, walker_vx, walker_vy, 
                                other_ped_x, other_ped_y, other_ped_vx, other_ped_vy):
        """
        Calculate virtual interaction force between two pedestrians based on the repulsion & navigation model
        Formula: F_ij,vir = F_ij,rep + F_ij,nav
        """
        # Calculate distance between pedestrians
        dx = walker_x - other_ped_x
        dy = walker_y - other_ped_y
        d_ij = math.sqrt(dx**2 + dy**2)
        
        if d_ij == 0:
            return 0, 0  # No force if pedestrians are at the same position
        
        # Unit vector from other pedestrian to walker
        n_ij_x = dx / d_ij
        n_ij_y = dy / d_ij
        
        # Calculate relative velocity (from other pedestrian to walker in walker's coordinate)
        v_ji_x = walker_vx - other_ped_vx
        v_ji_y = walker_vy - other_ped_vy
        
        # Calculate phi_ij_v: angle between n_ij and v_ji
        # First calculate the dot product
        v_ji_mag = math.sqrt(v_ji_x**2 + v_ji_y**2)
        if v_ji_mag == 0:
            phi_ij_v = 0
        else:
            dot_product = (v_ji_x * n_ij_x + v_ji_y * n_ij_y) / v_ji_mag
            # Clamp to avoid numerical errors
            dot_product = max(-1, min(1, dot_product))
            phi_ij_v = math.acos(dot_product)
            
            # Determine the sign of phi_ij_v using cross product
            cross_product = v_ji_x * n_ij_y - v_ji_y * n_ij_x
            if cross_product < 0:
                phi_ij_v = -phi_ij_v
        
        # Parameters for repulsion force (you may need to adjust these based on your requirements)
        d_rep = 0.7801  # Repulsion distance threshold
        M_rep = 30.028  # Maximum repulsion force magnitude
        sigma_rep = 0.45971243  # Smoothing parameter for repulsion force
        lambda_rep = 0.1  # Anisotropy parameter for repulsion force
        
        # Calculate repulsion force
        F_rep_mag = flm(d_ij, d_rep, M_rep, sigma_rep) * Asin(phi_ij_v, lambda_rep)
        F_rep_x = F_rep_mag * n_ij_x
        F_rep_y = F_rep_mag * n_ij_y
        
        # Parameters for navigation force
        d_nav = 1.5892008  # Navigation distance threshold
        M_nav = 41.875  # Maximum navigation force magnitude
        sigma_nav = 0.41745  # Smoothing parameter for navigation force
        lambda_nav = 1.0  # Anisotropy parameter for navigation force
        
        # Calculate navigation force
        # First get perpendicular unit vector
        n_ij_perp_x, n_ij_perp_y = get_perpendicular_vector(n_ij_x, n_ij_y, phi_ij_v)
        
        F_nav_mag = flm(d_ij, d_nav, M_nav, sigma_nav) * Aexp(phi_ij_v, lambda_nav)
        F_nav_x = F_nav_mag * n_ij_perp_x
        F_nav_y = F_nav_mag * n_ij_perp_y
        
        # Total virtual interaction force
        F_vir_x = F_rep_x + F_nav_x
        F_vir_y = F_rep_y + F_nav_y
        
        return F_vir_x, F_vir_y

    def get_perpendicular_vector(n_ij_x, n_ij_y, phi_ij_v):
        """Calculate perpendicular unit vector to n_ij, direction depends on phi_ij_v"""
        # Left perpendicular vector (counterclockwise 90 degrees)
        n_ij_perp_left_x = -n_ij_y
        n_ij_perp_left_y = n_ij_x
        # Right perpendicular vector (clockwise 90 degrees)
        n_ij_perp_right_x = n_ij_y
        n_ij_perp_right_y = -n_ij_x
        
        # Choose direction based on phi_ij_v
        # If phi_ij_v is positive, the relative velocity is pointing to the left
        # If phi_ij_v is negative, the relative velocity is pointing to the right
        if phi_ij_v >= 0:
            return n_ij_perp_left_x, n_ij_perp_left_y
        else:
            return n_ij_perp_right_x, n_ij_perp_right_y


    for i in range(len(other_ped_x_position)):
        d_ij = math.sqrt((walker_x_position - other_ped_x_position[i])**2 + (walker_y_position - other_ped_y_position[i])**2)
        if d_ij == 0:
            continue
        n_ij_x = (walker_x_position - other_ped_x_position[i]) / d_ij
        n_ij_y = (walker_y_position - other_ped_y_position[i]) / d_ij
        F_col_x = alpha * min(d_ij-repulsive_distance, 0) * n_ij_x
        F_col_y = alpha * min(d_ij-repulsive_distance, 0) * n_ij_y
        F_vir_x, F_vir_y = virtual_interaction_force(walker_x_position, walker_y_position, walker_v_x_past, walker_v_y_past, 
                                other_ped_x_position[i], other_ped_y_position[i], other_ped_v_x_past[i], other_ped_v_y_past[i])
        F_pedestrian_x += F_col_x + F_vir_x
        F_pedestrian_y += F_col_y + F_vir_y
    
    return F_pedestrian_x, F_pedestrian_y
        

def walker_logic_SF(car_v, car_x_position, car_y_position, walker_x_position, walker_y_position, 
                    walker_v_x_past, walker_v_y_past, v_max=2.5, a_max=5, destination_x=C.WALKER_DESTINATION_X, destination_y=C.WALKER_DESTINATION_Y, rng=None):
    """
    Update pedestrian velocity (x and y) based on destination attraction and vehicle influence.

    Parameters:
    - car_v: vehicle speed
    - car_x_position: vehicle x-coordinate
    - car_y_position: vehicle y-coordinate
    - walker_x_position: pedestrian x-coordinate
    - walker_y_position: pedestrian y-coordinate
    - walker_v_x_past: pedestrian past x-velocity
    - walker_v_y_past: pedestrian past y-velocity
    - v_max: maximum speed (optional, default 2.5 m/s)
    - a_max: maximum acceleration (optional, default 5 m/s^2)

    Returns:
    - walker_v_x: updated pedestrian x-velocity
    - walker_v_y: updated pedestrian y-velocity
    """
    
    # 1. Compute vehicle influence on pedestrian (returns force in x and y)
    F_vehicle_x, F_vehicle_y, _ = F_vehicle(car_x_position, car_y_position, car_v, walker_x_position, walker_y_position, walker_v_x_past, walker_v_y_past)
    F_vehicle_x = 100*F_vehicle_x
    F_vehicle_y = 0.2*F_vehicle_y
    # 2. Compute destination attraction (returns only y-direction force)
    F_destination_x, F_destination_y = F_destination(walker_x_position, destination_x, walker_y_position, destination_y, F_vehicle_x, F_vehicle_y, walker_v_x_past, walker_v_y_past)
    # 3. Compute total force components in x and y

    F_total_x = F_vehicle_x + F_destination_x # only vehicle affects x-direction here
    F_total_y =  F_vehicle_y + F_destination_y # destination and vehicle affect y-direction
    
    # 4. Compute acceleration from force (F = m * a => a = F / m)
    a_x = F_total_x / C.mass_pedestrian
    a_y = F_total_y / C.mass_pedestrian
    
    # 5. Limit acceleration to not exceed a_max
    # Compute acceleration magnitude
    total_a = math.sqrt(a_x**2 + a_y**2)
    if total_a > a_max:
        # Scale acceleration proportionally to keep direction but cap magnitude
        a_x = a_x / total_a * a_max
        a_y = a_y / total_a * a_max
    # 6. Update pedestrian velocity (v = v_0 + a * dt)
    if rng is None:
        rng = np.random
    walker_v_x = walker_v_x_past + a_x * C.dt #+ rng.normal(0, C.walker_noise_x_sigma)*car_v*0.3 # update x-velocity
    walker_v_y = walker_v_y_past + a_y * C.dt #+ rng.normal(0, C.walker_noise_y_sigma)*car_v*0.3 # update y-velocity
    
    # 7. Limit speed to not exceed v_max
    total_v = math.sqrt(walker_v_x**2 + walker_v_y**2)
    if total_v > C.v_max:
        # Scale speed proportionally to keep direction but cap magnitude
        walker_v_x = walker_v_x / total_v * C.v_max
        walker_v_y = walker_v_y / total_v * C.v_max
    
    return walker_v_x, walker_v_y

def walker_logic_SF_multi_ped(car_v, car_x_position, car_y_position, walker_x_position, walker_y_position, 
                    walker_v_x_past, walker_v_y_past, other_ped_x_position, other_ped_y_position, other_ped_v_x_past, other_ped_v_y_past, v_max=2.5, a_max=5, destination_x=C.WALKER_DESTINATION_X, destination_y=C.WALKER_DESTINATION_Y, rng=None):
    """
    Update multiple pedestrian velocities (x and y) based on destination attraction, vehicle influence and pedestrian interaction.
    Parameters:
    - car_v: vehicle speed
    - car_x_position: vehicle x-coordinate
    - car_y_position: vehicle y-coordinate
    - walker_x_position: pedestrian x-coordinate
    - walker_y_position: pedestrian y-coordinate
    - walker_v_x_past: pedestrian past x-velocity
    - walker_v_y_past: pedestrian past y-velocity
    - other_ped_x_position: list of other pedestrian x-coordinate
    - other_ped_y_position: list of other pedestrian y-coordinate
    - other_ped_v_x_past: list of other pedestrian past x-velocity
    - other_ped_v_y_past: list of other pedestrian past y-velocity
    - v_max: maximum speed (optional, default 2.5 m/s)
    - a_max: maximum acceleration (optional, default 5 m/s^2)

    Returns:
    - walker_v_x: updated pedestrian x-velocity
    - walker_v_y: updated pedestrian y-velocity
    """
    # 1. Compute vehicle influence on pedestrian (returns force in x and y)
    F_vehicle_x, F_vehicle_y, _ = F_vehicle(car_x_position, car_y_position, car_v, walker_x_position, walker_y_position, walker_v_x_past, walker_v_y_past)
    F_vehicle_x = 100*F_vehicle_x
    F_vehicle_y = 0.2*F_vehicle_y
    # 2. Compute destination attraction (returns only y-direction force)
    F_destination_x, F_destination_y = F_destination(walker_x_position, destination_x, walker_y_position, destination_y, F_vehicle_x, F_vehicle_y, walker_v_x_past, walker_v_y_past)
    # 3. Compute pedestrian interaction force (returns x and y-direction force)
    F_pedestrian_x, F_pedestrian_y = F_pedestrian(walker_x_position, walker_y_position, walker_v_x_past, walker_v_y_past, other_ped_x_position, other_ped_y_position, other_ped_v_x_past, other_ped_v_y_past)

    # 4. Compute total force components in x and y

    F_total_x = F_vehicle_x + F_destination_x + F_pedestrian_x # only vehicle affects x-direction here
    F_total_y =  F_vehicle_y + F_destination_y + F_pedestrian_y # destination and vehicle affect y-direction
    
    # 5. Compute acceleration from force (F = m * a => a = F / m)
    a_x = F_total_x / C.mass_pedestrian
    a_y = F_total_y / C.mass_pedestrian
    
    # 6. Limit acceleration to not exceed a_max
    # Compute acceleration magnitude
    total_a = math.sqrt(a_x**2 + a_y**2)
    if total_a > a_max:
        # Scale acceleration proportionally to keep direction but cap magnitude
        a_x = a_x / total_a * a_max
        a_y = a_y / total_a * a_max
    # 7. Update pedestrian velocity (v = v_0 + a * dt)
    if rng is None:
        rng = np.random
    walker_v_x = walker_v_x_past + a_x * C.dt # + rng.normal(0, C.walker_noise_x_sigma) # update x-velocity
    walker_v_y = walker_v_y_past + a_y * C.dt # + rng.normal(0, C.walker_noise_y_sigma) # update y-velocity
    
    # 8. Limit speed to not exceed v_max
    total_v = math.sqrt(walker_v_x**2 + walker_v_y**2)
    if total_v > C.v_max:
        # Scale speed proportionally to keep direction but cap magnitude
        walker_v_x = walker_v_x / total_v * C.v_max
        walker_v_y = walker_v_y / total_v * C.v_max
    
    return walker_v_x, walker_v_y

#########################################################################################################################################
# The following is visualization testing and unrelated to core functionality
#########################################################################################################################################

def visualize_vehicle_pedestrian_interaction():
    """Create visualization of vehicle-pedestrian interaction"""
    # Use unified font sizes for every text in this figure
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["font.size"] = 25
    plt.rcParams["axes.labelsize"] = 25
    plt.rcParams["xtick.labelsize"] = 25
    plt.rcParams["ytick.labelsize"] = 25
    plt.rcParams["legend.fontsize"] = 20
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Vehicle parameters
    car_x, car_y = 0, 0
    car_v = 2.0  # 2 m/s
    
    # Three pedestrians in different situations
    pedestrians = [
        {"pos": [-2.3, 1], "vel": [-0.5, -0.5], "color": "red", "label": "Ped 1"},      # Rectangular area
        {"pos": [3, 0.1], "vel": [0, -0.5], "color": "blue", "label": "Ped 2"},   # Inside triangle
        {"pos": [4, 1.3], "vel": [0.2, -0.2], "color": "green", "label": "Ped 3"}  # Outside triangle
    ]
    
    # Draw vehicle body (actual car)
    vehicle_rect = patches.Rectangle((car_x - l_r, car_y - l_w/2), l_f + l_r, l_w, 
                                   linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
    ax.add_patch(vehicle_rect)
    ax.text(car_x, car_y, 'CAR', ha='center', va='center', fontweight='bold', fontsize=18)
    
    # Draw vehicle virtual contour
    contour_points, _, _ = get_vehicle_contour_and_influence_point(car_x, car_y, car_v, 0, 0)
    contour_points.append(contour_points[0])  # Close the polygon
    contour_x, contour_y = zip(*contour_points)
    ax.plot(contour_x, contour_y, 'k--', linewidth=2, alpha=0.8, label='Virtual Contour')
    ax.fill(contour_x, contour_y, color='yellow', alpha=0.2)
    
    # Process each pedestrian
    for i, ped in enumerate(pedestrians):
        walker_x, walker_y = ped["pos"]
        walker_vx, walker_vy = ped["vel"]
        color = ped["color"]
        label = ped["label"]
        
        # Draw pedestrian
        circle = plt.Circle((walker_x, walker_y), 0.27, color=color, alpha=0.8)
        ax.add_patch(circle)
        # Adjust label positions slightly per pedestrian for better readability
        label_dx, label_dy = 0.35, 0.3
        if label == 'Ped 3':
            label_dx = 0.55  # move Ped 3 a bit to the right
        ax.text(walker_x + label_dx, walker_y + label_dy, label, ha='center', va='bottom', 
                fontweight='bold', color=color, fontsize=18)
        
        # Get influence point and force
        contour_points, influence_point, situation = get_vehicle_contour_and_influence_point(
            car_x, car_y, car_v, walker_x, walker_y)
        
        F_x, F_y, _ = F_vehicle(car_x, car_y, car_v, walker_x, walker_y, walker_vx, walker_vy)
        
        # Calculate F_des (destination force) - only in y direction
        F_des_x, F_des_y = F_destination(walker_y, -5, F_x, F_y, walker_vx, walker_vy)
        
        print("situation:", situation)
        print("F_veh_x:", F_x, "F_veh_y:", F_y)
        print("F_des_x:", F_des_x, "F_des_y:", F_des_y)
        
        # Draw influence point
        if situation != "Inside Triangle (Freezing)":
            ax.plot(influence_point[0], influence_point[1], 'ko', markersize=8, 
                   markerfacecolor=color, markeredgecolor='black', markeredgewidth=2)
            
            # Draw line from influence point to pedestrian
            ax.plot([influence_point[0], walker_x], [influence_point[1], walker_y], 
                   color=color, linestyle=':', linewidth=2, alpha=0.7)
            
            # Draw F_veh force vector (scaled for visibility)
            if F_x != 0 or F_y != 0:
                force_scale = 0.02  # Scale factor for force visualization
                ax.arrow(walker_x, walker_y, F_x * force_scale, F_y * force_scale,
                        head_width=0.1, head_length=0.15, fc=color, ec=color, linewidth=2)
                
                # Add F_veh force magnitude text
                force_mag = math.sqrt(F_x**2 + F_y**2)
                # ax.text(walker_x + 0.5, walker_y + 0.5, f'F_veh={force_mag:.1f}N', 
                #        color=color, fontsize=10, fontweight='bold')
                # Shift F_veh text left for specific pedestrians
                fveh_dx = -0.5
                if label == 'Ped 1':
                    fveh_dx = -0.7
                elif label == 'Ped 3':
                    fveh_dx = -0.6  # further left by 0.1
                ax.text(walker_x + fveh_dx, walker_y + 0.3, f'F_veh', 
                       color=color, fontsize=18, fontweight='bold')
        
        # Draw F_des force vector (destination force)
        if F_des_y != 0 or F_des_x != 0:
            force_scale = 0.004  # Scale factor for force visualization
            # Use a different color for F_des (darker version of pedestrian color)
            f_des_color = color
            ax.arrow(walker_x, walker_y, F_des_x * force_scale, F_des_y * force_scale,
                    head_width=0.1, head_length=0.15, fc=f_des_color, ec=f_des_color, 
                    linewidth=2, linestyle='-', alpha=0.8)
            f_des_mag = math.sqrt(F_des_x**2 + F_des_y**2)
            # ax.text(walker_x +0.25 , walker_y - 0.5, f'F_des={f_des_mag:.1f}N', 
            #        color=f_des_color, fontsize=10, fontweight='bold')
            # Shift F_des left a bit for Ped 3
            fdes_dx = -0.6
            if label == 'Ped 3':
                fdes_dx = -0.8
            ax.text(walker_x + fdes_dx, walker_y - 0.5, f'F_des', 
                   color=f_des_color, fontsize=18, fontweight='bold')
                 
    
    # Add velocity vectors for pedestrians
    for ped in pedestrians:
        walker_x, walker_y = ped["pos"]
        walker_vx, walker_vy = ped["vel"]
        color = ped["color"]
        
        # Draw velocity vector
        # ax.arrow(walker_x, walker_y, walker_vx, walker_vy,
        #         head_width=0.15, head_length=0.1, fc=color, ec=color, alpha=0.5, linewidth=2)
    
    # Add vehicle velocity vector
    ax.arrow(car_x, car_y + l_w/2 + 0.5, car_v, 0,
            head_width=0.2, head_length=0.2, fc='black', ec='black', linewidth=3)
    ax.text(car_x + car_v/2, car_y + l_w/2 + 0.8, f'v_veh={car_v}m/s', ha='center', fontweight='bold', fontsize=18)
    
    # Set up plot
    ax.set_xlim(-4, 6)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    # ax.set_title('Vehicle-Pedestrian Interaction Visualization\n(Social Force Model)', fontsize=18, fontweight='bold')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Virtual Contour'),
        plt.Line2D([0], [0], marker='s', color='w', markeredgecolor='k', markerfacecolor='gray', markersize=10, label='Vehicle'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Pedestrians'),
        plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='white', markersize=8, label='Influence Points'),
        plt.Line2D([0], [0], color='k', linewidth=3, label='F_veh (Vehicle Force)'),
        plt.Line2D([0], [0], color='k', linewidth=3, linestyle='-', alpha=0.8, label='F_des (Destination Force)'),
        # plt.Line2D([0], [0], color='k', alpha=0.5, linewidth=2, label='Velocity Vectors')
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    
    plt.tight_layout()
    plt.show()

# Run the visualization# test
if __name__ == "__main__":
    visualize_vehicle_pedestrian_interaction()
    
#     car_speeds = np.linspace(0, 10, 100)  # vehicle speed from 0 to 10 m/s
# walker_x_positions = np.linspace(0, 10, 100)  # pedestrian x from 0 to 10 m
# walker_y_positions = np.linspace(0, 10, 100)  # pedestrian y from 0 to 10 m

# X, Y = np.meshgrid(walker_x_positions, walker_y_positions)
# v_x = np.zeros_like(X)
# v_y = np.zeros_like(Y)

# # Compute pedestrian velocity for each position
# for i in range(len(walker_x_positions)):
#     for j in range(len(walker_y_positions)):
#         walker_v_x, walker_v_y = walker_logic_SF(5, 320, 320, walker_x_positions[i], walker_y_positions[j], 0, 0)
#         v_x[i, j] = walker_v_x
#         v_y[i, j] = walker_v_y

# # Plot v_x and v_y under different parameters

# fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# # Plot for v_x
# contour = ax[0].contourf(X, Y, v_x, cmap=cm.viridis)
# ax[0].set_title('v_x vs Walker Position')
# ax[0].set_xlabel('Walker X Position')
# ax[0].set_ylabel('Walker Y Position')
# fig.colorbar(contour, ax=ax[0])

# # Plot for v_y
# contour = ax[1].contourf(X, Y, v_y, cmap=cm.viridis)
# ax[1].set_title('v_y vs Walker Position')
# ax[1].set_xlabel('Walker X Position')
# ax[1].set_ylabel('Walker Y Position')
# fig.colorbar(contour, ax=ax[1])

# plt.tight_layout()
# plt.show()