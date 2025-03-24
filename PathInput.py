from copy import deepcopy

from PathOptimizationModel import *
# from PathSampling import *
# from PathMutation import *
# from PathCrossover import *
# from PathRepair import *
from pymoo.core.duplicate import NoDuplicateElimination


# CHANGE ALGORITHM INPUTS FROM HERE !!!

# MODEL
model = MTSP_TC_SOO_GA

# ALG
pop_size = 300
n_gen = 1000

"""

# OPERATORS
# path_sampling = PathSampling()
# path_mutation = PathMutation()
# path_crossover = PathCrossover()
# path_eliminate_duplicates = NoDuplicateElimination()
# path_repair = PathRepair()


# algorithm = model['Alg']

# if model == distance_soo_model:
#     algorithm = 'GA'
# elif model == moo_model_with_disconn:
#     algorithm = ['NSGA2','NSGA3']

# algorithm = 'NSGA2' if model==moo_model else 'GA'

# One single scenario for testing
test_setup_scenario = {
'grid_size': 8,
'cell_side_length': 50,
'number_of_drones': 8,
'max_drone_speed': 2.5, # m/s
'comm_cell_range': 2,  # 2 cells
'n_visits': 1,  # Minimum number of cell visits
'max_visits':5, # Maximum number of cell visits
'number_of_targets': 1,
'target_positions':12,
'true_detection_probability': 0.99,
'false_detection_probability': 0.01,
'detection_threshold': 0.9,
'max_isolated_time': 0,
}

single_visit_setup_scenarios = [

    {
    'grid_size': 8,
    'cell_side_length': 50,
    'number_of_drones': 4,
    'max_drone_speed': 2.5, # m/s
    'comm_cell_range': 2,  # 2 cells
    'n_visits': 1,  # Minimum number of cell visits
    'max_visits':5, # Maximum number of cell visits
    'number_of_targets': 1,
    'target_positions':12,
    'true_detection_probability': 0.99,
    'false_detection_probability': 0.01,
    'detection_threshold': 0.9,
    'max_isolated_time': 0,
},

    {
    'grid_size': 8,
    'cell_side_length': 50,
    'number_of_drones': 4,
    'max_drone_speed': 2.5, # m/s
    'comm_cell_range': 4,  # 2 cells
    'n_visits': 1,  # Minimum number of cell visits
    'max_visits':5, # Maximum number of cell visits
    'number_of_targets': 1,
    'target_positions':12,
    'true_detection_probability': 0.99,
    'false_detection_probability': 0.01,
    'detection_threshold': 0.9,
    'max_isolated_time': 0,
},

    {
    'grid_size': 8,
    'cell_side_length': 50,
    'number_of_drones': 8,
    'max_drone_speed': 2.5, # m/s
    'comm_cell_range': 2,  # 2 cells
    'n_visits': 1,  # Minimum number of cell visits
    'max_visits':5, # Maximum number of cell visits
    'number_of_targets': 1,
    'target_positions':12,
    'true_detection_probability': 0.99,
    'false_detection_probability': 0.01,
    'detection_threshold': 0.9,
    'max_isolated_time': 0,
},

    {
    'grid_size': 8,
    'cell_side_length': 50,
    'number_of_drones': 8,
    'max_drone_speed': 2.5, # m/s
    'comm_cell_range': 4,  # 2 cells
    'n_visits': 1,  # Minimum number of cell visits
    'max_visits':5, # Maximum number of cell visits
    'number_of_targets': 1,
    'target_positions':12,
    'true_detection_probability': 0.99,
    'false_detection_probability': 0.01,
    'detection_threshold': 0.9,
    'max_isolated_time': 0,
},

    {
    'grid_size': 8,
    'cell_side_length': 50,
    'number_of_drones': 12,
    'max_drone_speed': 2.5, # m/s
    'comm_cell_range': 2,  # 2 cells
    'n_visits': 1,  # Minimum number of cell visits
    'max_visits':5, # Maximum number of cell visits
    'number_of_targets': 1,
    'target_positions':12,
    'true_detection_probability': 0.99,
    'false_detection_probability': 0.01,
    'detection_threshold': 0.9,
    'max_isolated_time': 0,
},

    {
    'grid_size': 8,
    'cell_side_length': 50,
    'number_of_drones': 12,
    'max_drone_speed': 2.5, # m/s
    'comm_cell_range': 4,  # 2 cells
    'n_visits': 1,  # Minimum number of cell visits
    'max_visits':5, # Maximum number of cell visits
    'number_of_targets': 1,
    'target_positions':12,
    'true_detection_probability': 0.99,
    'false_detection_probability': 0.01,
    'detection_threshold': 0.9,
    'max_isolated_time': 0,
},


    {
    'grid_size': 8,
    'cell_side_length': 50,
    'number_of_drones': 16,
    'max_drone_speed': 2.5, # m/s
    'comm_cell_range': 2,  # 2 cells
    'n_visits': 1,  # Minimum number of cell visits
    'max_visits':5, # Maximum number of cell visits
    'number_of_targets': 1,
    'target_positions':12,
    'true_detection_probability': 0.99,
    'false_detection_probability': 0.01,
    'detection_threshold': 0.9,
    'max_isolated_time': 0,
},

    {
    'grid_size': 8,
    'cell_side_length': 50,
    'number_of_drones': 16,
    'max_drone_speed': 2.5, # m/s
    'comm_cell_range': 4,  # 2 cells
    'n_visits': 1,  # Minimum number of cell visits
    'max_visits':5, # Maximum number of cell visits
    'number_of_targets': 1,
    'target_positions':12,
    'true_detection_probability': 0.99,
    'false_detection_probability': 0.01,
    'detection_threshold': 0.9,
    'max_isolated_time': 0,
}

]

two_visits_setup_scenarios = deepcopy(single_visit_setup_scenarios)
three_visits_setup_scenarios = deepcopy(single_visit_setup_scenarios)
four_visits_setup_scenarios = deepcopy(single_visit_setup_scenarios)
five_visits_setup_scenarios = deepcopy(single_visit_setup_scenarios)

for i in range(len(two_visits_setup_scenarios)):
    two_visits_setup_scenarios[i]["n_visits"] = 2
    three_visits_setup_scenarios[i]["n_visits"] = 3
    four_visits_setup_scenarios[i]["n_visits"] = 4
    five_visits_setup_scenarios[i]["n_visits"] = 5

"""