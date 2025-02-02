from Distance import *
from Connectivity import *
from Time import *
# from WeightedSum import * # mission_time_and_percentage_connectivity_ws, mission_time_and_percentage_connectivity_and_max_mean_tbv_ws

from PathSolution import *

model_metric_info = {
    "Objectives": {
        'Mission Time': (get_mission_time, 1),
        'Percentage Connectivity': (get_percentage_connectivity, -1),
        'Max Disconnected Time': (get_max_disconnected_time, 1),
        'Mean Disconnected Time': (get_mean_disconnected_time, 1),
        'Max Mean TBV': (get_max_mean_tbv, 1),
    },
    "Constraints": {
        'Max Mission Time': max_mission_time,
        'Min Percentage Connectivity': min_perc_conn_constraint,
        'Path Speed Violations as Constraint': path_speed_violations_as_constraint
    }
}


"""    # Objective Functions
    'Total Distance': get_total_distance,
    'Mission Time': get_mission_time,
    'Percentage Connectivity': calculate_percentage_connectivity,# bfs_connectivity
    'Percentage Disconnectivity': calculate_percentage_disconnectivity,
    'Path Smoothness': path_smoothness_penalty,
    'Total Disconnected Time': calculate_total_disconnected_time,
    'Max Disconnected Time': calculate_max_disconnected_time,
    'Mean Disconnected Time': calculate_mean_disconnected_time,
    'Total Drone Speed Violations as Objective': total_drone_speed_violations_as_objective,
    'Path Speed Violations as Objective': path_speed_violations_as_objective,
    'Path Smoothness as Objective': path_smoothness_as_objective,
    'Max Mean TBV': max_tbv_as_objective,
    # "Mission Time and Percentage Connectivity Weighted Sum": mission_time_and_percentage_connectivity_ws,
    # "Mission Time and Percentage Connectivity and Max Mean TBV Weighted Sum": mission_time_and_percentage_connectivity_and_max_mean_tbv_ws,
    # "Mission Time & Percentage Connectivity & Max Disconnected Time & Mean Disconnected Weighted Sum": mission_time_and_percentage_connectivity_and_max_disconnected_time_and_mean_disconnected_ws,
    


    # Inequality Constraints
    'Max Mission Time': max_mission_time,
    'Min Percentage Connectivity': min_perc_conn_constraint,
    'Max Mean TBV as Constraint': max_tbv_as_constraint,
    'Min Cell Visits': min_cell_visits,
    'Max Cell Visits': max_cell_visits,
    'Path Speed Violations as Constraint': path_speed_violations_as_constraint,
    'Total Drone Speed Violations as Constraint': total_drone_speed_violations_as_constraint,
    'Max Drone Speed Violations as Constraint': max_drone_speed_violations_as_constraint,
    'Limit Cell Range': limit_cell_range,
    'Subtour Range':get_subtour_range,
    'Time Penalties': calculate_time_penalty,
    'Longest Subtour': get_longest_subtour,
    'Total Diagonal Steps':get_total_diagonal_steps,
    'Mean Turning Angle': get_mean_turning_angle,
    'Max Number of Visits':calculate_max_visits,
    'Limit Total Traceback Penalty': limit_total_traceback_penalty,
    'Limit Max Traceback Penalty': limit_max_traceback_penalty,
    'Limit Cell per Drone': min_cells_per_drone_constr,
    'Limit Max Longest Subtour': max_longest_subtour_constr,
    'Limit Min Longest Subtour': min_longest_subtour_constr,
    'Limit Subtour Range': max_subtour_range_constr,


    # Equality Constraints
    'Hovering Drones Full Connectivity':enforce_hovering_connectivity,
    'Eliminate Traceback': eliminate_total_traceback,
    'Search Drone Path Smoothness': longest_path_smoothness_penalty,
    'Path Smoothness': eliminate_path_smoothness_penalties,
    'Eliminate Longest Path Traceback': eliminate_longest_path_traceback,
    'Speed Violation Smoothness': speed_violation_smoothness_as_constraint
}

# 'Limit Total Speed Violation', 'Limit Max Speed Violation', 'Limit Total Traceback Penalty' 'Limit Max Traceback Penalty'"""