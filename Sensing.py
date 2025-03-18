import numpy as np
import pandas as pd
from Results import save_best_solutions
from PathInfo import *
from FilePaths import *
from PathFileManagement import load_pickle
from Connectivity import connected_components, PathSolution, get_connected_nodes, connected_nodes
# from Distance import interpolate_between_cities
from array_operations import create_array_of_lists
import itertools
import copy
import math
from math import inf, ceil, log10
from scipy.optimize import linear_sum_assignment

from matplotlib import pyplot as plt
import seaborn as sns

import sys


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0  # Returns 0 if x is exactly 0


def interpolate_between_cities(sol:PathSolution, city_prev, city):

    interpolated_path = [city_prev]

    info = sol.info
    coords_prev = sol.get_coords(city_prev)
    coords = sol.get_coords(city)
    coords_delta = coords - coords_prev
    axis_inc = np.array([sign(coords_delta[0]), sign(coords_delta[1])])

    num_mid_cities = int(max(abs(coords_delta))/info.cell_side_length)

    coords_temp = coords_prev.copy()

    for _ in range(num_mid_cities):
        if coords_temp[0] != coords[0]:
            coords_temp[0] += info.cell_side_length * axis_inc[0]
        if coords_temp[1] != coords[1]:
            coords_temp[1] += info.cell_side_length * axis_inc[1]
        mid_city = sol.get_city(coords_temp)
        interpolated_path.append(mid_city)

    return interpolated_path


    pass


"""def merge_maps(conn_comp, search_map):
    new_search_map = copy.deepcopy(search_map)
    number_of_nodes, number_of_cells = search_map.shape

    for clique in conn_comp:
        for cell in range(number_of_cells):
            clique_cell_info = new_search_map[clique, cell].T.flatten().tolist()
            all_cell_observations = list(itertools.chain(*clique_cell_info))
            unique_cell_observations = list({frozenset(d.items()): d for d in all_cell_observations}.values())
            unique_cell_observations_timesteps = np.array([x["timestep"] for x in unique_cell_observations])
            recent_timestep = np.max(unique_cell_observations_timesteps)
            recent_timestep_indices = np.where(unique_cell_observations_timesteps == recent_timestep)[0]
            recent_cell_observations = [unique_cell_observations[i] for i in recent_timestep_indices]
            n_obs = np.max([x["n_obs"] for x in recent_cell_observations])
            prob = np.mean([x["prob"] for x in recent_cell_observations])
            most_recent_obs = {"timestep": recent_timestep, "n_obs": n_obs, "prob": prob}
            for node_cell_observations in clique_cell_info:
                node_cell_observations_timesteps = np.array([x["timestep"] for x in node_cell_observations])
                recent_timestep = np.max(node_cell_observations_timesteps)
                if most_recent_obs["timestep"] > recent_timestep:
                    node_cell_observations.append(most_recent_obs)
                elif most_recent_obs["timestep"] == recent_timestep:
                    node_recent_timestep = np.where(node_cell_observations_timesteps == recent_timestep)[0]
                    node_recent_cell_observations = [node_cell_observations[i] for i in node_recent_timestep]
                    new_prob = np.mean([x["prob"] for x in node_recent_cell_observations] + [most_recent_obs["prob"]])
                    new_n_obs = np.max([x["n_obs"] for x in node_recent_cell_observations] + [most_recent_obs["n_obs"]])
                    node_cell_observations.append({"timestep": node_recent_timestep, "n_obs": new_n_obs, "prob": new_prob})
    return new_search_map
"""

def merge_maps(conn_comp, search_map):
    new_search_map = copy.deepcopy(search_map)
    number_of_nodes, number_of_cells = search_map.shape
    number_of_nodes = number_of_nodes - 1

    for clique in conn_comp:
        for cell in range(number_of_cells):
            clique_cell_info = new_search_map[clique, cell].T.flatten().tolist()
            # Combine all cell observations from all nodes in the clique
            all_cell_observations = list(itertools.chain(*clique_cell_info))
            # Extract unique observations
            unique_cell_observations = list({frozenset(d.items()): d for d in all_cell_observations}.values())
            # print(unique_cell_observations)
            unique_cell_observations_timesteps = np.array([x["timestep"] for x in unique_cell_observations])
            unique_cell_observations_n_obs = np.array([x["n_obs"] for x in unique_cell_observations])
            recent_timestep = np.max(unique_cell_observations_timesteps)
            recent_timestep_indices = np.where(unique_cell_observations_timesteps==recent_timestep)[0]
            recent_cell_observations = [unique_cell_observations[i] for i in recent_timestep_indices]
            # recent_cell_observations = unique_cell_observations[recent_timestep_indices]
            n_obs = np.max([x["n_obs"] for x in recent_cell_observations])
            prob = np.mean([x["prob"] for x in recent_cell_observations])
            most_recent_obs = {"timestep":recent_timestep, "n_obs":n_obs, "prob":prob}
            # recent_cell_observations = []
            for node_cell_observations in clique_cell_info:
                node_cell_observations_timesteps = np.array([x["timestep"] for x in node_cell_observations])
                node_cell_observations_n_obs = np.array([x["n_obs"] for x in node_cell_observations])
                recent_timestep = np.max(node_cell_observations_timesteps)
                if most_recent_obs["timestep"] > recent_timestep:
                    # Update the node's observation
                    node_cell_observations.append(most_recent_obs)
                elif most_recent_obs["timestep"] == recent_timestep:
                    node_recent_timestep_indices = np.where(node_cell_observations_timesteps==recent_timestep)[0]
                    node_recent_cell_observations = [node_cell_observations[i] for i in node_recent_timestep_indices]
                    # node_recent_cell_observations = node_cell_observations[node_recent_timestep_indices]
                    new_prob = np.mean([x["prob"] for x in node_recent_cell_observations] + [most_recent_obs["prob"]])
                    # print(f"prob list: {[x['prob'] for x in node_recent_cell_observations] + [most_recent_obs['prob']]}, new_prob: {new_prob}")
                    new_n_obs = np.max([x["n_obs"] for x in node_recent_cell_observations] + [most_recent_obs["n_obs"]])
                    # Update the node's observation
                    node_cell_observations[-1] = {"timestep":recent_timestep, "n_obs":new_n_obs, "prob":new_prob}
                else:
                    continue

    return new_search_map

def sensing_and_info_sharing(sol, target_locations=[12], B=0.9, p=0.9, q=0.2):
    # Make a deep copy of the solution object to avoid modifying the original object
    x = copy.deepcopy(sol)
    info = x.info
    final_search_steps = [len(q) - 2 for q in list(x.drone_dict.values())]
    drone_path_matrix = x.real_time_path_matrix[1:, :]
    number_of_drones, timesteps = drone_path_matrix.shape
    connectivity_matrix = x.connectivity_matrix
    search_map = np.empty((sol.info.number_of_nodes, sol.info.number_of_cells), dtype=object)
    rows, cols = search_map.shape
    for i in range(rows):
        for j in range(cols):
            search_map[i, j] = [{"n_obs": 0, "timestep": -1, "prob": 0.5}]
    drone_full_detection_timesteps = np.full(number_of_drones, fill_value=np.inf)
    drones_path_back_to_bs = {"drone_no": i, "path_back_to_bs": []}
    relay_cells = []
    drone_status = ["search"] * number_of_drones
    timestep_bs_knows_at_least_one_target = np.inf
    timestep_bs_knows_all_targets = np.inf
    timestep_at_least_one_drone_knows_all_targets = np.inf
    timestep_all_targets_are_known = np.inf

    for step in range(max(final_search_steps)):
        adj_mat = connectivity_matrix[step]
        conn_comp = connected_components(adj_mat)
        search_map = merge_maps(conn_comp, search_map)
        for drone in range(number_of_drones):
            if step > final_search_steps[drone]:
                continue
            drone_position = drone_path_matrix[drone, step]
            prior_obs = search_map[drone + 1, drone_position][-1]["prob"]
            if drone_position in target_locations:
                new_obs = p * prior_obs / (p * prior_obs + q * (1 - prior_obs))
            else:
                new_obs = (1 - p) * prior_obs / ((1 - p) * prior_obs + (1 - q) * (1 - prior_obs))
            search_map[drone + 1, drone_position].append({"timestep": step, "prob": new_obs, "n_obs": max([x["n_obs"] for x in search_map[drone + 1, drone_position]]) + 1})
        search_map = merge_maps(conn_comp, search_map)
        occupancy_status = np.zeros((info.number_of_nodes, info.number_of_cells))
        for row, col in itertools.product(range(info.number_of_nodes), range(info.number_of_cells)):
            observations = search_map[row, col]
            observation_probs = np.array([x["prob"] for x in observations])
            if len(np.where(observation_probs > B)[0]) >= 1:
                occupancy_status[row, col] = 1
        if timestep_all_targets_are_known == np.inf and len(np.unique(np.where(occupancy_status == 1)[1])) >= len(target_locations):
            timestep_all_targets_are_known = step
        if timestep_bs_knows_at_least_one_target == np.inf and len(np.where(occupancy_status[0] == 1)[0]) >= 1:
            timestep_bs_knows_at_least_one_target = step
        if timestep_bs_knows_all_targets == np.inf and len(np.where(occupancy_status[0] == 1)[0]) >= len(target_locations):
            timestep_bs_knows_all_targets = step
            relay_cells = np.where(occupancy_status[0] == 1)[0]
        for m in range(number_of_drones):
            if timestep_at_least_one_drone_knows_all_targets == np.inf and len(np.where(occupancy_status[m + 1] == 1)[0]) >= len(target_locations):
                timestep_at_least_one_drone_knows_all_targets = step
                drone_full_detection_timesteps[m] = step
                path_to_cell_0 = interpolate_between_cities(x, drone_path_matrix[m, step], 0)
                new_path = path_to_cell_0 + [-1] * (timesteps - len(path_to_cell_0))
                x.real_time_path_matrix[m + 1] = new_path
        if timestep_bs_knows_all_targets == np.inf and len(relay_cells) > 0:
            connected_drones_to_bs = get_connected_nodes(adj_mat, 0)
            unassigned_drones = [i for i in range(number_of_drones) if drone_status[i] == "search"]
            distances = np.array([info.D[drone, relay_cell] for drone in unassigned_drones for relay_cell in relay_cells])
            drone_indices, relay_point_indices = linear_sum_assignment(distances)
            drone_ids = [unassigned_drones[i] for i in drone_indices]
            for i in drone_ids:
                drone_status[i] = "relay"
                path_to_relay = interpolate_between_cities(x, drone_path_matrix[i, step], relay_cells[relay_point_indices[i]])
                new_path = path_to_relay + path_to_relay[-1] * (timesteps - len(path_to_relay))
                x.real_time_path_matrix[i + 1] = new_path
            for filled_relay_idx in relay_point_indices:
                relay_cells.pop(filled_relay_idx)

    detection_time = sum(x.time_elapsed_at_steps[:timestep_all_targets_are_known + 1]) if timestep_all_targets_are_known != np.inf else np.inf
    time_bs_knows_at_least_one_target = sum(x.time_elapsed_at_steps[:timestep_bs_knows_at_least_one_target + 1]) if timestep_bs_knows_at_least_one_target != np.inf else np.inf
    inform_time = sum(x.time_elapsed_at_steps[:timestep_bs_knows_all_targets + 1]) if timestep_bs_knows_all_targets != np.inf else np.inf
    original_detection_time = sum(sol.time_elapsed_at_steps[:max(final_search_steps) + 1])
    for step in range(max(final_search_steps), sol.real_time_path_matrix.shape[1]):
        if len(get_connected_nodes(sol.connectivity_matrix[step], 0)) == info.number_of_drones:
            original_inform_time = sum(sol.time_elapsed_at_steps[:step + 1])
            break
    detection_time_gain = (original_detection_time - detection_time) * 100 / original_detection_time if detection_time != np.inf else np.inf
    inform_time_gain = (original_inform_time - inform_time) * 100 / original_inform_time if inform_time != np.inf else np.inf
    return {"detection time gain": detection_time_gain, "inform time gain": inform_time_gain, "Time BS knows at least one target": time_bs_knows_at_least_one_target}

"""def sensing_and_info_sharing(sol:PathSolution, target_locations=[12], B=0.9, p=0.9, q=0.2):
    
    # Make a deep copy of the solution object to avoid modifying the original object
    x = deepcopy(sol)
    info = x.info
    final_search_steps = [len(q)-2 for q in list(x.drone_dict.values())]
    # hovering_cells = [seq[-2] for seq in list(x.drone_dict.values())]

    drone_path_matrix = x.real_time_path_matrix[1:,:]
    number_of_drones, timesteps = drone_path_matrix.shape
    connectivity_matrix = x.connectivity_matrix
    info = x.info
    # Initialize search maps (Async. and sync.)
    search_map = np.empty((sol.info.number_of_nodes, sol.info.number_of_cells), dtype=object)
    rows, cols = search_map.shape
    for i in range(rows):
        for j in range(cols):
            search_map[i, j] = [{"n_obs":0, "timestep":-1, "prob":0.5}] # Start at timestep=-1 to indicate that this is the prior belief
    drone_full_detection_timesteps = np.full(number_of_drones, fill_value=inf)
    drones_path_back_to_bs = {"drone_no":i, "path_back_to_bs":[]}
    relay_cells = []
    drone_status = ["search"] * number_of_drones
    timestep_bs_knows_at_least_one_target = inf
    timestep_bs_knows_all_targets = inf
    timestep_at_least_one_drone_knows_all_targets = inf
    timestep_all_targets_are_known = inf # Timestep all targets are known overall (a single drone doesn't have to know all targets in this metric)
    timestep_relay_chain_is_formed = inf
    for step in range(max(final_search_steps)):
        adj_mat = connectivity_matrix[step]
        conn_comp = connected_components(adj_mat)
        search_map = merge_maps(conn_comp, search_map)
        # TODO: Add new observations
        for drone in range(number_of_drones):
            # If drone is in hovering state, continue
            if step > final_search_steps[drone]:
                continue
            else:
                # Update occupancy probability in the async. search map
                drone_position = drone_path_matrix[drone, step]
                prior_obs = search_map[drone+1, drone_position][-1]["prob"]
                if drone_position in target_locations:
                    new_obs = p*prior_obs / (p*prior_obs + q*(1-prior_obs))
                else:
                    new_obs = (1-p)*prior_obs / ((1-p)*prior_obs + (1-q)*(1-prior_obs))
                search_map[drone+1, drone_position].append({"timestep":step, "prob":new_obs, "n_obs": max([x["n_obs"] for x in search_map[drone+1, drone_position]])+1})
        # TODO: Info Merging
<<<<<<< HEAD
        for clique in connected_nodes:
            # print(f"Clique: {clique}")
            clique_async_search_map = async_search_map[clique]
            clique_sync_search_map = sync_search_map[clique]
            for cell in range(info.number_of_cells):
                clique_async_cell_info = clique_async_search_map[:,cell]
                clique_sync_cell_info = clique_sync_search_map[:,cell]
                combined_cell_info = SM_clique[:,cell].flatten().tolist()
                print(combined_cell_info)
                # print(SM[clique][cell])
                for node in clique:
                    SM[node][cell] = combined_cell_info
                    # print(f"--> {SM[node][cell]}")
            # print(f"Post-Merge: {SM[clique]}")
        
    #     print(f"step {step} completed")
    # print("Search map generation completed")
    # for loc in target_locations: 
    #     print(f"target cell {loc} search map: {SM[:,loc]}")
=======
        search_map = merge_maps(conn_comp, search_map)
        # print(f"search map updated at step {step}")
        # Extract Time Metrics
        occupancy_status = np.zeros((info.number_of_nodes, info.number_of_cells))
        for row, col in itertools.product(range(info.number_of_nodes), range(info.number_of_cells)):
            observations = search_map[row, col]
            observation_probs = np.array([x["prob"] for x in observations])
            # print(observation_probs)
            if len(np.where(observation_probs > B)[0]) >= 1:
                occupancy_status[row, col] = 1
        # print(f"occupancy status:\n{occupancy_status}")
        # Occupancy Status: if cell is believed to be empty, it is 0, otherwise 1 (Mxn) Matrix
        # Set Time Metrics
        # Timestep all targets are known
        if timestep_all_targets_are_known == inf and len(np.unique(np.where(occupancy_status==1)[1])) >= len(target_locations):
            timestep_all_targets_are_known = step
        # Timestep BS Knows At Least One Target
        if timestep_bs_knows_at_least_one_target == inf and len(np.where(occupancy_status[0]==1)[0]) >= 1:
            timestep_bs_knows_at_least_one_target = step
        # Timestep BS Knows All Targets
        if timestep_bs_knows_all_targets == inf and len(np.where(occupancy_status[0]==1)[0]) >= len(target_locations):
            timestep_bs_knows_all_targets = step
            relay_cells = np.where(occupancy_status[0]==1)[0]
        for m in range(number_of_drones):
            # Timestep At Least One Drone Knows All Targets
            if timestep_at_least_one_drone_knows_all_targets == inf and len(np.where(occupancy_status[m+1]==1)[0]) >= len(target_locations):
                timestep_at_least_one_drone_knows_all_targets = step
                drone_full_detection_timesteps[m] = step
                # Reroute drone to BS
                path_to_cell_0 = interpolate_between_cities(x, drone_path_matrix[m, step], 0)
                new_path = path_to_cell_0 + [-1] * (timesteps - len(path_to_cell_0))
                x.real_time_path_matrix[m+1] = new_path
        if timestep_bs_knows_all_targets == inf and len(relay_cells) > 0 :
            connected_drones_to_bs = get_connected_nodes(adj_mat, 0)
            unassigned_drones = [i for i in range(number_of_drones) if drone_status[i] == "search"]
            distances = np.array([info.D[drone, relay_cell] for drone in unassigned_drones for relay_cell in relay_cells])
            drone_indices, relay_point_indices = linear_sum_assignment(distances)
            drone_ids = [unassigned_drones[i] for i in drone_indices]
            for i in drone_ids:
                drone_status[i] = "relay"
                path_to_relay = interpolate_between_cities(x, drone_path_matrix[i, step], relay_cells[relay_point_indices[i]])
                new_path = path_to_relay + path_to_relay[-1] * (timesteps - len(path_to_relay))
                x.real_time_path_matrix[i+1] = new_path
            for filled_relay_idx in relay_point_indices:
                relay_cells.pop(filled_relay_idx)
            # for drone_idx, in drone_indices:
            #     drone_status[unassigned_drones[drone_idx]] = "relay"
>>>>>>> 504d702 (Sensing Completed)

    detection_time = sum(x.time_elapsed_at_steps[:timestep_all_targets_are_known+1]) if timestep_all_targets_are_known != inf else inf
    time_bs_knows_at_least_one_target = sum(x.time_elapsed_at_steps[:timestep_bs_knows_at_least_one_target+1]) if timestep_bs_knows_at_least_one_target != inf else inf
    inform_time = sum(x.time_elapsed_at_steps[:timestep_bs_knows_all_targets+1]) if timestep_bs_knows_all_targets != inf else inf
    # time_at_least_one_drone_knows_all_targets = sum(x.time_elapsed_at_steps[:timestep_at_least_one_drone_knows_all_targets+1])
    # Now find original path matrix inform and detection times
    original_detection_time = sum(sol.time_elapsed_at_steps[:max(final_search_steps)+1])
    for step in range(max(final_search_steps), sol.real_time_path_matrix.shape[1]):
        # print(get_connected_nodes(sol.connectivity_matrix[step], 0))
        if len(get_connected_nodes(sol.connectivity_matrix[step], 0))==info.number_of_drones:
            original_inform_time = sum(sol.time_elapsed_at_steps[:step+1])
            break
    detection_time_gain = (original_detection_time - detection_time)*100 / original_detection_time if detection_time != inf else inf
    inform_time_gain = (original_inform_time - inform_time)*100 / original_inform_time if inform_time != inf else inf
    # print(original_detection_time, original_inform_time)
    # print(detection_time, inform_time)
    return {"detection time gain": detection_time_gain, "inform time gain": inform_time_gain, "Time BS knows at least one target": time_bs_knows_at_least_one_target}
"""
    # print(time_all_targets_are_known, time_bs_knows_at_least_one_target, time_bs_knows_all_targets, time_at_least_one_drone_knows_all_targets)







            

        # Print BS search map
        # print(search_map[0, :])

    # print("Search map generation completed")
    # print(f"Target Cells: {target_locations}")
    # Print all cells' search maps
    # for loc in range(info.number_of_cells):
    #     print("---------------------------------------------------------------------------------------------------")
    #     print(f"cell {loc}")
    #     print(f"cell {loc} assigned drone ids: {np.where(x.real_time_path_matrix==loc)[0]}")
    #     print(f"cell {loc} search map: {search_map[:,loc]}")
    #     print("---------------------------------------------------------------------------------------------------")

    # Print only target cells' search maps
    # for loc in target_locations: 
    #     print("---------------------------------------------------------------------------------------------------")
    #     print(f"target cell {loc}")
    #     print(f"target cell {loc} assigned drone ids: {np.unique(np.where(x.real_time_path_matrix[:,:max(final_search_steps)]==loc)[0])}")
    #     print(f"target cell {loc} visit steps: {np.unique(np.where(x.real_time_path_matrix[:,:max(final_search_steps)]==loc)[1])}")
    #     print(f"target cell {loc} search map: {search_map[:,loc]}")
    
# PLOT
n_runs = 1000
p0 = 0.5
for B in [0.9, 0.95]:
    for p in [0.8, 0.9]:
        q = 1-p
        m = ceil( log10((p0*(1-B))/(B*(1-p0))) / log10(q/p) )
        for comm_range in ["2", "sqrt(8)"]:
            for n_targets in [3,4,5]:
                # fig, ax = plt.subplots(1, 4, figsize=(12, 6))
                # fig, ax = plt.subplots(figsize=(8, 10))
                y_detection_time_gain_list = [0, 0, 0, 0]
                y_inform_time_gain_list = [0, 0, 0, 0]
                y_time_at_least_one_drone_knows_all_targets_list = [0, 0, 0, 0]
                y_successful_runs = [0, 0, 0, 0]
                for run_no in range(n_runs):
                    print(f"Run {run_no+1}")
                    target_locations = np.random.choice(range(1, 64), n_targets, replace=False)
                    for i,number_of_drones in enumerate([4, 8, 12, 16]):
                        sol_set = copy.deepcopy(load_pickle(f"{solutions_filepath}MOO_NSGA2_MTSP_TCDT_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{m}-SolutionObjects.pkl")).flatten().tolist()
                        F = copy.deepcopy(pd.read_pickle(f"{objective_values_filepath}MOO_NSGA2_MTSP_TCDT_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{m}-ObjectiveValues.pkl"))
                        best_sol = sol_set[F["Mission Time"].idxmin()] # Best connectivity solution of specificed scenario
                        best_sol_time_metrics = sensing_and_info_sharing(best_sol, target_locations=target_locations, B=B, p=p, q=1-p)
                        y_detection_time_gain_list[i] += best_sol_time_metrics["detection time gain"] if best_sol_time_metrics["detection time gain"] != inf else 0
                        y_inform_time_gain_list[i] += best_sol_time_metrics["inform time gain"] if best_sol_time_metrics["inform time gain"] != inf else 0
                        y_time_at_least_one_drone_knows_all_targets_list[i] += best_sol_time_metrics["Time BS knows at least one target"] if best_sol_time_metrics["Time BS knows at least one target"] != inf else 0
                        if best_sol_time_metrics["inform time gain"] != inf:
                            y_successful_runs[i] += 1
                y_detection_time_gain_list = [x/n_runs for x in y_detection_time_gain_list]
                y_inform_time_gain_list = [x/n_runs for x in y_inform_time_gain_list]
                y_time_at_least_one_drone_knows_all_targets_list = [x/n_runs for x in y_time_at_least_one_drone_knows_all_targets_list]
                y_success_rate = [x/n_runs for x in y_successful_runs]

                # Plot Detection Time Gain
                fig,ax = plt.subplots(figsize=(8, 6))
                fig.suptitle(f"p: {np.round(p,2)}, B: {np.round(B,2)}, q: {np.round(q,2)}, " + "$n_{visits}$: " + str(m) + f", T: {n_targets}, $r_c$: {comm_range}")
                plt.plot([4, 8, 12, 16], y_detection_time_gain_list, label="Detection Time Gain")
                plt.title("Detection Time Gain")
                plt.xlabel("Number of Drones")
                plt.ylabel("Detection Time Gain (%)")
                plt.xticks([4, 8, 12, 16])
                plt.grid()
                plt.legend()
                plt.savefig(f"Figures/Sensing/p_{round(p,2)}_B_{round(B,2)}_q_{round(q,2)}_nvisits_{m}_T_{n_targets}_r_{comm_range}_detection_time_gain.png")
                plt.show()
                plt.close()

                # Plot Inform Time Gain
                fig,ax = plt.subplots(figsize=(8, 6))
                fig.suptitle(f"p: {np.round(p,2)}, B: {np.round(B,2)}, q: {np.round(q,2)}, " + "$n_{visits}$: " + str(m) + f", T: {n_targets}, $r_c$: {comm_range}")
                plt.plot([4, 8, 12, 16], y_inform_time_gain_list, label="Inform Time Gain")
                plt.title("Inform Time Gain (%)")
                plt.xlabel("Number of Drones")
                plt.ylabel("Inform Time Gain (%)")
                plt.xticks([4, 8, 12, 16])
                plt.grid()
                plt.legend()
                plt.savefig(f"Figures/Sensing/p_{round(p,2)}_B_{round(B,2)}_q_{round(q,2)}_nvisits_{m}_T_{n_targets}_r_{comm_range}_inform_time_gain.png")
                plt.show()
                plt.close()

                # Plot Time BS Knows At Least One Target
                fig,ax = plt.subplots(figsize=(8, 6))
                fig.suptitle(f"p: {np.round(p,2)}, B: {np.round(B,2)}, q: {np.round(q,2)}, " + "$n_{visits}$: " + str(m) + f", T: {n_targets}, $r_c$: {comm_range}")
                plt.plot([4, 8, 12, 16], y_time_at_least_one_drone_knows_all_targets_list, label="Time BS knows at least one target")
                plt.title("Time BS knows at least one target (s)")
                plt.xlabel("Number of Drones")
                plt.ylabel("Time BS knows at least one target (s)")
                plt.xticks([4, 8, 12, 16])
                plt.grid()
                plt.legend()
                plt.savefig(f"Figures/Sensing/p_{round(p,2)}_B_{round(B,2)}_q_{round(q,2)}_nvisits_{m}_T_{n_targets}_r_{comm_range}_time_bs_knows_at_least_one_target.png")
                plt.show()
                plt.close()

                # Plot Success Rate
                fig,ax = plt.subplots(figsize=(8, 6))
                fig.suptitle(f"p: {np.round(p,2)}, B: {np.round(B,2)}, q: {np.round(q,2)}, " + "$n_{visits}$: " + str(m) + f", T: {n_targets}, $r_c$: {comm_range}")
                plt.plot([4, 8, 12, 16], y_success_rate, label="Success Rate")
                plt.title("Success Rate")
                plt.xlabel("Number of Drones")
                plt.ylabel("Success Rate")
                plt.xticks([4, 8, 12, 16])
                plt.grid()
                plt.legend()
                plt.savefig(f"Figures/Sensing/p_{round(p,2)}_B_{round(B,2)}_q_{round(q,2)}_nvisits_{m}_T_{n_targets}_r_{comm_range}_success_rate.png")



                # ax[0].plot([4, 8, 12, 16], y_detection_time_gain_list, label="Detection Time Gain")
                # ax[0].set_title("Detection Time Gain")
                # ax[0].set_xlabel("Number of Drones")
                # ax[0].set_ylabel("Detection Time Gain (%)")
                # ax[0].set_xticks([4, 8, 12, 16])
                # ax[0].grid()
                # ax[0].legend()
                # ax[1].plot([4, 8, 12, 16], y_inform_time_gain_list, label="Inform Time Gain")
                # ax[1].set_title("Inform Time Gain")
                # ax[1].set_xlabel("Number of Drones")
                # ax[1].set_ylabel("Inform Time Gain (%)")
                # ax[1].set_xticks([4, 8, 12, 16])
                # ax[1].grid()
                # ax[1].legend()
                # ax[2].plot([4, 8, 12, 16], y_time_at_least_one_drone_knows_all_targets_list, label="Time BS knows at least one target")
                # ax[2].set_title("Time BS knows at least one target")
                # ax[2].set_xlabel("Number of Drones")
                # ax[2].set_ylabel("Time BS knows at least one target")
                # ax[2].set_xticks([4, 8, 12, 16])
                # ax[2].grid()
                # ax[2].legend()
                # ax[3].plot([4, 8, 12, 16], y_success_rate, label="Success Rate")
                # ax[3].set_title("Success Rate")
                # ax[3].set_xlabel("Number of Drones")
                # ax[3].set_ylabel("Success Rate")
                # ax[3].set_xticks([4, 8, 12, 16])
                # ax[3].grid()
                # ax[3].legend()
                # plt.show()
                        


    
# Metrics: Detection Time Gain, Inform Time Gain, Time at least one drone knows all targets, Time BS knows at least one target
# 1. Detection Time Gain: (Original Detection Time - New Detection Time) * 100 / Original Detection Time
# 2. Inform Time Gain: (Original Inform Time - New Inform Time) * 100 / Original Inform Time
# 3. Time BS knows at least one target

# test
number_of_drones = 16
nvisits = 3
comm_range = "sqrt(8)"
model = "TCDT"
sol_set = deepcopy(load_pickle(f"{solutions_filepath}MOO_NSGA2_MTSP_{model}_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{nvisits}-SolutionObjects.pkl").flatten().tolist())
F = deepcopy(pd.read_pickle(f"{objective_values_filepath}MOO_NSGA2_MTSP_{model}_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{nvisits}-ObjectiveValues.pkl"))
best_conn_idx = F["Percentage Connectivity"].idxmin()
best_conn_sol = sol_set[best_conn_idx]
random_sol = np.random.choice(sol_set)

# generate_sar_paths(sample_sol, merging_strategy="belief", target_locations=[12,15,8], B=0.9, p=0.9, q=0.2)
print(sensing_and_info_sharing(best_conn_sol, target_locations=[2,15,61], B=0.9, p=0.9, q=0.2))