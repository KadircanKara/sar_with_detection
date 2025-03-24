from PathDuplicateElimination import PathDuplicateElimination
from pymoo.core.duplicate import NoDuplicateElimination, HashDuplicateElimination

from PathSampling import PathSampling
from PathMutation import PathMutation
from PathCrossover import PathCrossover
from PathRepair import PathRepair
from PathTermination import PathTermination# , PathDefaultTermination, PathDefaultMultiObjectiveTermination
from PathUnitTest import *

import os
from math import sqrt
from PathInput import *


"""Algoriithms To RUN"""
# TCDT WS
# TCD WS
# TT WS

# !!! DON'T FORGET n=4, r=2*sqrt(2) runs, you skipped those !!!

scenario = {
                        'grid_size': 8,
                        'cell_side_length': 50,
                        'number_of_drones': 4, # n=12, r=2*sqrt(2), n_visits=1
                        'max_drone_speed': 2.5, # m/s
                        'comm_cell_range': 2,  # 4 cells
                        'n_visits': 1,  # Minimum number of cell visits
                        }

number_of_drones_values = [8]
comm_cell_range_values = [2,2*sqrt(2)]
n_visits_values = [1,2,3]


scenarios = []
for v in n_visits_values:
    for n in number_of_drones_values:
        for r in comm_cell_range_values:
            if n==4 and r==2 and v==1:
                 continue
            scenarios.append(
                    {   'grid_size': 8,
                        'cell_side_length': 50,
                        'number_of_drones': n,
                        'max_drone_speed': 2.5, # m/s
                        'comm_cell_range': r,  # 4 cells
                        'n_visits': v,  # Minimum number of cell visits
                        'target_positions': [12],
                        'th': 0.9,
                        'detection_probability':0.7
                    }
                )

# SAMPLING
path_sampling = PathSampling()
# MUTATION
path_mutation = PathMutation({
                    "swap_last_point":(0, 1),
                    "swap": (0.3, 1), # 0.3 0.5
                    "inversion": (0.4, 1), # 0.4
                    "scramble": (0.3, 1), # 0.3 0.6
                    "insertion": (0, 1),
                    "displacement": (0, 1),
                    # "reverse sequence": (0.3, 1),
                    "block inversion": (0, 1),
                    # "shift": (0.3, 1),
                    "random_one_sp_mutation": (0.4, 1), # 1.0 for 4 drones |  0.95 for 8 drones | 0.7 for 12 drones | 0.5 for 16 drones
                    "random_n_sp_mutation": (0.0, 1), # 1.0 for 4 drones |  0.95 for 8 drones | 0.7 for 12 drones | 0.5 for 16 drones
                    "all_sp_mutation": (0.0, 1), # 1.0 for 4 drones |  0.95 for 8 drones | 0.7 for 12 drones | 0.5 for 16 drones
                    "longest_path_sp_mutation": (0.0, 1), # 1.0 for 4 drones |  0.95 for 8 drones | 0.7 for 12 drones | 0.5 for 16 drones
                    "randomly_selected_sp_mutation": (0.0, 1), # 1.0 for 4 drones |  0.95 for 8 drones | 0.7 for 12 drones | 0.5 for 16 drones
                })
# CROSSOVER
path_crossover = PathCrossover(prob=0.9, ox_prob=1.0, n_offsprings=2)
# REPAIR
path_repair = PathRepair()
# DUPLICATES
path_eliminate_duplicates = NoDuplicateElimination()


if __name__ == "__main__":
        
        old_filecount = len(os.listdir(objective_values_filepath))
        # Run scenarios
        for scenario in scenarios:
            test = PathUnitTest(scenario)
            test(save_results=True, animation=False, copy_to_drive=False)
        new_filecount = len(os.listdir(objective_values_filepath))
        filecount_diff = new_filecount - old_filecount
        if filecount_diff == len(scenarios)*2:
            print(f"ALL SCENARIOS RAN SUCCESSFULLY !!!")
        else:
            print(f"ERROR: {len(scenarios)*2-filecount_diff} SCENARIOS DIDN'T CONVERGE !!!")