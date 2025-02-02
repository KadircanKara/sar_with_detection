import numpy as np
import pandas as pd
from Results import save_best_solutions
from PathInfo import *
from FilePaths import *
from PathFileManagement import load_pickle

info = PathInfo()
info.number_of_drones = 12
info.comm_cell_range = 2*sqrt(2)
info.n_visits = 2
# scenario = str(info)
scenario = 'MOO_NSGA2_TCDT_g_8_a_50_n_4_v_2.5_r_sqrt(8)_nvisits_3'
objective_values = pd.read_pickle(f'{objective_values_filepath}{scenario}-ObjectiveValues.pkl')
mission_time_values = objective_values['Mission Time']
best_mission_time_sol = load_pickle(f'{solutions_filepath}{scenario}-Best-Mission_Time-Solution.pkl')
print(best_mission_time_sol.mission_time)
# print(min(mission_time_values))


# save_best_solutions(scenario=scenario, copy_to_drive=False)