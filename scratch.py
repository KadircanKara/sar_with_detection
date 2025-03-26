from FilePaths import *
from PathFileManagement import *
import os
import copy
# import pandas as pd
import numpy as np
from PathSolution import produce_n_tour_sol
from PathOptimizationModel import *
# import joblib

# data = joblib.load("Results/Objectives/SOO_GA_MTSP_g_8_a_50_n_4_v_2.5_r_sqrt(8)_nvisits_2-ObjectiveValues.pkl")

# with open("Results/Objectives/SOO_GA_MTSP_g_8_a_50_n_4_v_2.5_r_sqrt(8)_nvisits_2-ObjectiveValues.pkl", "rb") as file:
#     data = pickle.load(file)
#     print(data)


test = np.load("Results/Objectives/SOO_GA_MTSP_g_8_a_50_n_4_v_2.5_r_sqrt(8)_nvisits_2-ObjectiveValues.pkl", allow_pickle=False)
# test = pd.read_pickle("Results/Objectives/SOO_GA_MTSP_g_8_a_50_n_4_v_2.5_r_sqrt(8)_nvisits_2-ObjectiveValues.pkl")

"""for filename in os.listdir(solutions_filepath):
    if "WS" in filename:
        print("Scenario:", filename.split("-")[0])
        filepath = f"{solutions_filepath}{filename}"
        X = load_pickle(filepath)
        for sol in X:
            sol.info.model = TCDT_WS
        save_as_pickle(filepath, X)
        # print(load_pickle(filepath)[0].info.model["F"])
"""        

"""for filename in os.listdir(objective_values_filepath):
    if "WS" in filename:
        print("Scenario:", filename.split("-")[0])
        filepath = f"{objective_values_filepath}{filename}"
        F = pd.read_pickle(filepath)
        assert (len(list(F.columns))==1), "Length greater than 1 !"
        old_column = list(F.columns)[0]
        print("Old Column:", old_column)
        new_column = old_column.replace("-", " & ").replace(" & Weighted Sum", " Weighted Sum")
        print("New Column:", new_column)
        F.columns = [new_column]
        F.to_pickle(filepath)
        # print("-->", list(F.columns))
        print("Post-Update Columns:", list(pd.read_pickle(filepath).columns))
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------")
"""
"""obj_dict = {
    "Mission Time":{"attribute":"mission_time", "normalization_factor":1000},
    "Percentage Connectivity": {"attribute":"percentage_connectivity", "normalization_factor":1},
    "Max Mean TBV": {"attribute":"max_mean_tbv", "normalization_factor":1},
    "Max Disconnected Time": {"attribute":"max_disconnected_time", "normalization_factor":1},
    "Mean Disconnected Time": {"attribute":"mean_disconnected_time", "normalization_factor":1},
}

n_tours_list = 2,3,4,5

sols_dir = os.listdir(solutions_filepath)
sols_dir.reverse()
print(sols_dir)

for filename in sols_dir:
    scenario = filename.split("-")[0]
    # Debug: Print the filename being processed
    print(f"Processing filename: {filename}")
    
    # Ensure case-insensitive matching
    if "nvisits_1" in filename.lower():
        print(f"Matched 'nvisits_1': {filename}")
        
        # Skip files containing "Weighted Sum" (case-insensitive)
        # if "weighted sum" in filename.lower():
        #     print(f"Skipping 'Weighted Sum' file: {filename}")
        #     continue
        
        # Proceed with processing
        objs = copy.deepcopy(pd.read_pickle(f"{objective_values_filepath}{scenario}-ObjectiveValues.pkl"))
        if "Weighted Sum" in objs.columns[0]:
            continue
        sols = copy.deepcopy(load_pickle(f"{solutions_filepath}{filename}"))
        runtime = load_pickle(f"{runtimes_filepath}{scenario}-Runtime.pkl")
        
        for n_tours in n_tours_list:
            n_tours_sols = sols.copy()
            n_tours_F = objs.copy()
            for row, sol in enumerate(n_tours_sols):
                sol = produce_n_tour_sol(sol, n_tours)
                if len(n_tours_F.columns) > 1:
                    for col in n_tours_F.columns:
                        obj = obj_dict[col]["attribute"]
                        n_tours_F[col].iloc[row] = getattr(sol, obj)
                else:
                    score = 0
                    objectives = sol.info.model["F"]
                    for objective in objectives:
                        score += (
                            getattr(sol, obj_dict[objective]["attribute"]) *
                            obj_dict[objective]["normalization_factor"]
                        )
                    n_tours_F.iloc[row] = score

            save_as_pickle(f"{solutions_filepath}{filename.replace('nvisits_1', f'ntours_{n_tours}')}", n_tours_sols)
            save_as_pickle(f"{objective_values_filepath}{scenario.replace('nvisits_1', f'ntours_{n_tours}')}-ObjectiveValues.pkl", n_tours_F)
            save_as_pickle(f"{objective_values_filepath}{scenario.replace('nvisits_1', f'ntours_{n_tours}')}-ObjectiveValuesAbs.pkl", abs(n_tours_F))
            save_as_pickle(f"{runtimes_filepath}{scenario.replace('nvisits_1', f'ntours_{n_tours}')}-Runtime.pkl", runtime)"""

"""for filename in sols_dir:
    scenario = filename.split("-")[0]
    if "nvisits_1" in filename:
        print(filename)
        if "Weighted Sum" in filename:
            continue
        sols = copy.deepcopy(load_pickle(f"{solutions_filepath}{filename}"))
        objs = copy.deepcopy(pd.read_pickle(f"{objective_values_filepath}{scenario}-ObjectiveValues.pkl"))
        runtime = load_pickle(f"{runtimes_filepath}{scenario}-Runtime.pkl")
        for n_tours in n_tours_list:
            # if f"{solutions_filepath}{filename.replace("nvisits_1",f"ntours_{n_tours}")}" in os.listdir(solutions_filepath):
            #     continue
            n_tours_sols = sols.copy()
            n_tours_F = objs.copy()
            for row,sol in enumerate(n_tours_sols):
                sol = produce_n_tour_sol(sol, n_tours)
                if len(n_tours_F.columns) > 1:
                    for col in n_tours_F.columns:
                        obj = obj_dict[col]["attribute"]
                        n_tours_F[col].iloc[row] = getattr(sol, obj)
                else:
                    score = 0
                    objectives = sol.info.model["F"]
                    for objective in objectives:

                        score += ( getattr(sol, obj_dict[objective]["attribute"]) * obj_dict[objective]["normalization_factor"] )
                    n_tours_F.iloc[row] = score

            save_as_pickle( f"{solutions_filepath}{filename.replace('nvisits_1',f'ntours_{n_tours}')}", n_tours_sols)
            save_as_pickle( f"{objective_values_filepath}{scenario.replace('nvisits_1',f'ntours_{n_tours}')}-ObjectiveValues.pkl",  n_tours_F)
            save_as_pickle( f'{objective_values_filepath}{scenario.replace("nvisits_1",f"ntours_{n_tours}")}-ObjectiveValuesAbs.pkl',  abs(n_tours_F))
            save_as_pickle( f'{runtimes_filepath}{scenario.replace("nvisits_1",f"ntours_{n_tours}")}-Runtime.pkl',  runtime) # Add runtimes too for consistency
"""



        



"""dirs = [solutions_filepath, objective_values_filepath, runtimes_filepath]

for dir in dirs:
    filenames = os.listdir(dir)
    for filename in filenames:
        split_filename = filename.split("_")
        type_ = split_filename[0]
        exp = filename.split("_")[2]
        if type_ == "SOO" and exp == "TCDT":
            new_filename = filename.replace("SOO","WS")
            # print(f"Original Scenario: {filename.split('-')[0]}\nNew Scenario: {new_filename.split('-')[0]}")
            os.rename(f"{dir}{filename}", f"{dir}{new_filename}")
        # split_filename = filename.split("_")
        # # print(split_filename)
        # split_filename = split_filename[:2] + ["MTSP"] + split_filename[2:]
        # new_filename = ""
        # for i in range(len(split_filename)):
        #     new_filename += split_filename[i] + "_"
        # new_filename = new_filename[:-1]
        # # print(new_filename)
        # # new_filename = filename.replace("minv", "nvisits")
        # os.rename(f"{dir}{filename}", f"{dir}{new_filename}")
        # print(f"Renamed {filename} to {new_filename}")"""