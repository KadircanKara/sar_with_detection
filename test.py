import numpy as np
import pandas as pd
from PathFileManagement import load_pickle
from FilePaths import *
from PathOptimizationModel import calculate_ws_score_from_ws_objective, obj_name_sol_attr_dict, get_objectives_from_weighted_sum_model
from PathSolution import *

"""nvisit_scenario = f"WS_GA_TCT_g_8_a_50_n_12_v_2.5_r_sqrt(8)_nvisits_1"
one_tour_sol = load_pickle(f"{solutions_filepath}{nvisit_scenario}-SolutionObjects.pkl")[0]
two_tour_sol = load_pickle(f"{solutions_filepath}{nvisit_scenario.replace("nvisits_1",f"ntours_{2}")}-SolutionObjects.pkl")[0]
print(one_tour_sol.real_time_path_matrix.shape[1], two_tour_sol.real_time_path_matrix.shape[1] )
"""

"""nvisit_scenario = f"WS_GA_TCT_g_8_a_50_n_12_v_2.5_r_sqrt(8)_nvisits_1"
one_tour_sol = load_pickle(f"{solutions_filepath}{nvisit_scenario}-SolutionObjects.pkl")[0]
two_tour_sol = produce_n_tour_sol(one_tour_sol, 2)
print(f"Mission Time: {one_tour_sol.mission_time} | {two_tour_sol.mission_time}")
print(f"Percentage Connectivity: {one_tour_sol.percentage_connectivity} | {two_tour_sol.percentage_connectivity}")
print(f"Max Mean TBV: {one_tour_sol.max_mean_tbv} | {two_tour_sol.max_mean_tbv}")
"""

"""nvisit_scenario = f"WS_GA_TCT_g_8_a_50_n_12_v_2.5_r_sqrt(8)_nvisits_1"
nvisit_X = load_pickle(f"{solutions_filepath}{nvisit_scenario}-SolutionObjects.pkl")
nvisit_F = pd.read_pickle(f"{objective_values_filepath}{nvisit_scenario}-ObjectiveValues.pkl")
nvisit_R = load_pickle(f"{runtimes_filepath}{nvisit_scenario}-Runtime.pkl")
for ntour in range(2, 11):
    print(f"{ntour} Tours")
    ntour_scenario = f"WS_GA_TCT_g_8_a_50_n_12_v_2.5_r_sqrt(8)_ntours_{ntour}"
    ntour_X = load_pickle(f"{solutions_filepath}{ntour_scenario}-SolutionObjects.pkl")
    ntour_F = pd.read_pickle(f"{objective_values_filepath}{ntour_scenario}-ObjectiveValues.pkl")
    ntour_R = load_pickle(f"{runtimes_filepath}{ntour_scenario}-Runtime.pkl")
    objectives = get_objectives_from_weighted_sum_model(nvisit_X[0].info.model)

    for i in range(len(ntour_X)):
        nvisits_sol = nvisit_X[i]
        ntours_sol = ntour_X[i]
        for objective in objectives:
            print(f"{objective}: {getattr(nvisits_sol, obj_name_sol_attr_dict[objective])} | {getattr(ntours_sol, obj_name_sol_attr_dict[objective])}")
    print(f"nvisit F: {nvisit_F}\nntour F: {ntour_F}")
    print(f"nvisit R: {nvisit_R}, ntour R: {ntour_R}")
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
"""





# data = [1,2,3,4,5]
# columns = ["test"]
# df = pd.DataFrame(data=data, columns=columns)
# print(df)

# test = np.arange(2,10)
# print(test)

# test_sol = load_pickle("Results/Solutions/WS_GA_TCDT_g_8_a_50_n_16_v_2.5_r_2_nvisits_3-SolutionObjects.pkl")[0]
# # print(test_sol)
# calculate_ws_score_from_ws_objective(test_sol)


# print(np.__version__)
# print(pd.__version__)

# test = np.array([[1,2,3],[4,5,6]])
# print(test)
# import matplotlib

# import matplotlib

"""from math import ceil, log10
import numpy as np
import pandas as pd
# from Results import save_best_solutions
# from PathInfo import *
from FilePaths import *
# from PathFileManagement import load_pickle
# from Connectivity import connected_components, PathSolution"""

"""array = np.array([ [1,2,3], [4,5,6], [7,8,9] ])
flat_array = array.flatten()
print(flat_array)"""

# rows=8
# cols = 10

# array_of_lists = np.empty((rows, cols), dtype=object)
# for i in range(rows):
#     for j in range(cols):
#         array_of_lists[i, j] = [0.5]  # Assign an empty list to each cell

# print(array_of_lists)
# array_of_lists = np.empty(8, dtype=object)  # Creates an array of 5 empty objects
# for i in range(8):
#     array_of_lists[i] = [[0.5]*10]  # Assign empty lists
# print(array_of_lists)


# test_array = np.full(shape=(8,20), fill_value=[0.5], dtype=list)
# print(test_array[0])

"""F = pd.read_pickle("Results/Objectives/MOO_NSGA2_MTSP_TCDT_g_8_a_50_n_8_v_2.5_r_2_nvisits_3-ObjectiveValues.pkl")
for col in F.columns:
    print(f"{col} max val: {F[col].max()}")
"""
"""B_values = [0.9, 0.95] # 0.9, 0.95
p_values = [0.9, 0.8, 0.7] # 0.9, 0.8, 0.7
# q = 1-p
p0 = 0.5

for B in B_values:
    for p in p_values:
        q = 1-p
        m = ceil( log10((p0*(1-B))/(B*(1-p0))) / log10(q/p) )
        print(f"for B: {B}, p: {p}, # visit(s): {m}")
# m = ceil( log10((p0*(1-B))/(B*(1-p0))) / log10(q/p) )"""