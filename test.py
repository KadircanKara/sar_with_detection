from math import ceil, log10
import numpy as np
import pandas as pd
# from Results import save_best_solutions
# from PathInfo import *
from FilePaths import *
# from PathFileManagement import load_pickle
# from Connectivity import connected_components, PathSolution

array = np.array([ [1,2,3], [4,5,6], [7,8,9] ])
flat_array = array.flatten()
print(flat_array)

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