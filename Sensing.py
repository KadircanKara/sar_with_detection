import numpy as np
import pandas as pd
from Results import save_best_solutions
from PathInfo import *
from FilePaths import *
from PathFileManagement import load_pickle
from Connectivity import connected_components, PathSolution, connected_nodes_at_step # get_connected_nodes
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

"""def merge_maps(conn_comp, search_map, merging_strategy):
    new_search_map = copy.deepcopy(search_map)
    number_of_nodes, number_of_cells = search_map.shape

    for clique in conn_comp:
        for cell in range(number_of_cells):
            clique_cell_info = new_search_map[clique, cell].T.flatten().tolist()
            all_cell_observations = list(itertools.chain(*clique_cell_info))
            print(type(all_cell_observations[0]))
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
    return new_search_map"""



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




"""def merge_maps(conn_comp, search_map, merging_strategy="ondrone"):
    number_of_nodes, number_of_cells = search_map.shape
    number_of_drones = number_of_nodes - 1

    for clique in conn_comp:
        if merging_strategy == "gcs" and 0 not in clique:
            continue
        for cell in range(number_of_cells):
            # print(f"cell: {cell}")
            clique_cell_observations = search_map[clique]
            # print(f"clique cell observations: {clique_cell_observations}")
            clique_recent_cell_observations = [ x[cell][-1] for x in clique_cell_observations ]
            clique_unique_recent_cell_observations = list({frozenset(d.items()): d for d in clique_recent_cell_observations}.values())
            # print(f"recent cell observations: {clique_recent_cell_observations}\nunique recent cell observations:\n{clique_unique_recent_cell_observations}")
            for node in clique:
                # print(f"node: {node}", end=" ")
                node_cell_observations = search_map[node, cell]
                # print(f"node {node} cell {cell} observations: {node_cell_observations}")
                recent_node_cell_observation = node_cell_observations[-1]
                # print(f"recent node cell observation: {recent_node_cell_observation}", end=" ")
                # print(f"recent node cell observation: {recent_node_cell_observation}")
                # node_cell_observations = new_search_map[node, cell]
                for observation in clique_unique_recent_cell_observations:
                    # print(f"clique observation: {observation}", end=" ")
                    if observation["timestep"] > recent_node_cell_observation["timestep"]:
                        # print("appending...")
                        # print(node_cell_observations[-1], end=" ")
                        node_cell_observations = node_cell_observations + [observation]
                        # print(node_cell_observations[-1])
                    elif observation["timestep"] == recent_node_cell_observation["timestep"]:
                        node_cell_observations = node_cell_observations + [observation]
                        node_cell_observations[-1]["timestep"] += 1
                    print(f"new cell observations: {node_cell_observations}")
                # print(f"node {node} cell {cell} observations after merge: {node_cell_observations}")
    return search_map"""


def gcs_merging(bs_adj, search_map):
    number_of_nodes, number_of_cells = search_map.shape
    number_of_drones = number_of_nodes - 1
    new_search_map = copy.deepcopy(search_map)
    one_hop_neighbors = np.where(bs_adj == 1)[0]
    if len(one_hop_neighbors) != 0:
        for cell in range(number_of_cells):
            nbs_recent_cell_observations = [ x[cell][-1] for x in new_search_map[one_hop_neighbors, :] ]
            unique_nbs_recent_cell_observations = list({frozenset(d.items()): d for d in nbs_recent_cell_observations}.values())
            print(f"original: {nbs_recent_cell_observations}\nunique: {unique_nbs_recent_cell_observations}")
            # nbs_recent_cell_observations = []
            # for nb in one_hop_neighbors:
            #     nbs_recent_cell_observations.append(new_search_map[nb, cell][-1])
            # nbs_unique_recent_cell_observations = list({frozenset(d.items()): d for d in nbs_recent_cell_observations}.values())




            nb_cell_info = new_search_map[one_hop_neighbors, cell].T.flatten().tolist()
            # Combine all cell observations from all neighbors
            all_cell_observations = list(itertools.chain(*nb_cell_info))
            # Extract unique observations
            unique_cell_observations = list({frozenset(d.items()): d for d in all_cell_observations}.values())
            # print(unique_cell_observations)
            unique_cell_observations_timesteps = np.array([x["timestep"] for x in unique_cell_observations])
            recent_timestep = np.max(unique_cell_observations_timesteps)
            recent_timestep_indices = np.where(unique_cell_observations_timesteps==recent_timestep)[0]
            recent_cell_observations = [unique_cell_observations[i] for i in recent_timestep_indices]
            recent_cell_observations_n_obs_values = [x["n_obs"] for x in recent_cell_observations]
            recent_cell_observations_max_n_obs = np.max([x["n_obs"] for x in recent_cell_observations])
            recent_cell_observations_max_n_obs_indices = np.where(recent_cell_observations_n_obs_values==recent_cell_observations_max_n_obs)[0]


def merge_maps(conn_comp, search_map, merging_strategy="ondrone"):
    """ONDRONE MERGING"""
    new_search_map = copy.deepcopy(search_map)
    number_of_nodes, number_of_cells = search_map.shape
    number_of_nodes = number_of_nodes - 1

    for clique in conn_comp:
        if merging_strategy == "gcs" and 0 not in clique:
            continue
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
                if most_recent_obs["timestep"] >= recent_timestep:
                    # Update the node's observation
                    node_cell_observations.append(most_recent_obs)
                # elif most_recent_obs["timestep"] == recent_timestep:
                #     node_recent_timestep_indices = np.where(node_cell_observations_timesteps==recent_timestep)[0]
                #     node_recent_cell_observations = [node_cell_observations[i] for i in node_recent_timestep_indices]
                #     # node_recent_cell_observations = node_cell_observations[node_recent_timestep_indices]
                #     new_prob = np.mean([x["prob"] for x in node_recent_cell_observations] + [most_recent_obs["prob"]])
                #     # print(f"prob list: {[x['prob'] for x in node_recent_cell_observations] + [most_recent_obs['prob']]}, new_prob: {new_prob}")
                #     new_n_obs = np.max([x["n_obs"] for x in node_recent_cell_observations] + [most_recent_obs["n_obs"]])
                #     # Update the node's observation
                #     node_cell_observations[-1] = {"timestep":recent_timestep, "n_obs":new_n_obs, "prob":new_prob}
                else:
                    continue

    return new_search_map


def sensing_and_info_sharing(sol, merging_strategy="ondrone", target_locations=[12], B=0.9, p=0.9, q=0.2):
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
    # timestep_bs_knows_at_least_one_target = np.inf
    timestep_bs_knows_all_targets = np.inf
    timestep_at_least_one_drone_knows_all_targets = np.inf
    timestep_all_targets_are_known = np.inf

    for step in range(max(final_search_steps)):

        if timestep_bs_knows_all_targets != np.inf and timestep_at_least_one_drone_knows_all_targets != np.inf and timestep_all_targets_are_known != np.inf:
            break

        adj_mat = connectivity_matrix[step]
        conn_comp = connected_components(adj_mat)
        search_map = merge_maps(conn_comp, search_map, merging_strategy)
        for drone in range(number_of_drones):
            if step > final_search_steps[drone]:
                continue
            drone_position = drone_path_matrix[drone, step]
            prior_prob = search_map[drone + 1, drone_position][-1]["prob"]
            if drone_position in target_locations:
                new_prob = p * prior_prob / (p * prior_prob + q * (1 - prior_prob))
            else:
                new_prob = (1 - p) * prior_prob / ((1 - p) * prior_prob + (1 - q) * (1 - prior_prob))
            # if drone_position in target_locations:
                # print(f"Drone {drone} at cell {drone_position} with prior prob {prior_prob} and new prob {new_prob}")
            # search_map[drone + 1, drone_position].append({"n_obs": max([x["n_obs"] for x in search_map[drone + 1, drone_position]]) + 1, "timestep": step, "prob": new_prob})
            search_map[drone + 1, drone_position].append({"n_obs": np.count_nonzero(drone_path_matrix[drone,:step+1]==drone_position), "timestep": step, "prob": new_prob})
        search_map = merge_maps(conn_comp, search_map, merging_strategy)
        # print(f"Target cells SM:\n{search_map[:,target_locations]}")
        

        # MISSION TIME METRICS

        occupancy_status = np.zeros((info.number_of_nodes, info.number_of_cells))
        for row, col in itertools.product(range(info.number_of_nodes), range(info.number_of_cells)):
            observations = search_map[row, col]
            observation_probs = np.array([x["prob"] for x in observations])
            # print(observation_probs)
            if len(np.where(observation_probs > B)[0]) >= 1:
                occupancy_status[row, col] = 1

        # if merging_strategy == "gcs":
        #     occupancy_status = np.zeros((1,info.number_of_cells))
        #     for cell in range(info.number_of_cells):
        #         # print(search_map.shape)
        #         observations = search_map[0, cell]
        #         observation_probs = np.array([x["prob"] for x in observations])
        #         if len(np.where(observation_probs > B)[0]) >= 1:
        #             occupancy_status[cell] = 1
        # elif merging_strategy == "ondrone":
        #     occupancy_status = np.zeros((info.number_of_nodes, info.number_of_cells))
        #     for row, col in itertools.product(range(info.number_of_nodes), range(info.number_of_cells)):
        #         observations = search_map[row, col]
        #         observation_probs = np.array([x["prob"] for x in observations])
        #         # print(observation_probs)
        #         if len(np.where(observation_probs > B)[0]) >= 1:
        #             occupancy_status[row, col] = 1

        # print("-->", occupancy_status)

        # print(f"BS: {np.where(occupancy_status[0]==1)[0]}", end=" ")

        if timestep_all_targets_are_known == np.inf and len(np.unique(np.where(occupancy_status == 1)[1])) >= len(target_locations):
            timestep_all_targets_are_known = step
        # if timestep_bs_knows_at_least_one_target == np.inf and len(np.where(occupancy_status[0] == 1)[0]) >= 1:
        #     timestep_bs_knows_at_least_one_target = step
        if timestep_bs_knows_all_targets == np.inf and len(np.where(occupancy_status[0] == 1)[0]) >= len(target_locations):
            timestep_bs_knows_all_targets = step
            relay_cells = np.where(occupancy_status[0] == 1)[0]
        for m in range(number_of_drones):
            # print(f"Drone {m+1}: {np.where(occupancy_status[m+1]==1)[0]}" , end=" ") if m < number_of_drones - 1 else print(f"Drone {m+1}: {np.where(occupancy_status[m+1]==1)[0]}" , end="\n")
            if timestep_at_least_one_drone_knows_all_targets == np.inf and len(np.where(occupancy_status[m + 1] == 1)[0]) >= len(target_locations):
                timestep_at_least_one_drone_knows_all_targets = step
                drone_full_detection_timesteps[m] = step
                path_to_cell_0 = interpolate_between_cities(x, drone_path_matrix[m, step], 0)
                new_path = path_to_cell_0 + [-1] * (timesteps - len(path_to_cell_0))
                x.real_time_path_matrix[m + 1] = new_path
        if timestep_bs_knows_all_targets == np.inf and len(relay_cells) > 0:
            # connected_drones_to_bs = get_connected_nodes(adj_mat, 0)
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
    inform_time = sum(x.time_elapsed_at_steps[:timestep_bs_knows_all_targets + 1]) if timestep_bs_knows_all_targets != np.inf else np.inf
    time_at_least_one_drone_knows_all_targets = sum(x.time_elapsed_at_steps[:timestep_at_least_one_drone_knows_all_targets + 1]) if timestep_at_least_one_drone_knows_all_targets != np.inf else np.inf
    # time_bs_knows_at_least_one_target = sum(x.time_elapsed_at_steps[:timestep_bs_knows_at_least_one_target + 1]) if timestep_bs_knows_at_least_one_target != np.inf else np.inf
        
        # original_detection_time = sum(sol.time_elapsed_at_steps[:max(final_search_steps) + 1])
        # for step in range(max(final_search_steps), sol.real_time_path_matrix.shape[1]):
        #     if connected_nodes_at_step(sol.connectivity_matrix[step], 0) == info.number_of_drones:
        #         original_inform_time = sum(sol.time_elapsed_at_steps[:step + 1])
        #         break
        # detection_time_gain = (original_detection_time - detection_time) * 100 / original_detection_time if detection_time != np.inf else np.inf
        # inform_time_gain = (original_inform_time - inform_time) * 100 / original_inform_time if inform_time != np.inf else np.inf
    return {"detection time": detection_time, "inform time": inform_time, "Time at least one drone knows all targets":time_at_least_one_drone_knows_all_targets}


def run_tests(test="test", merging_strategy="ondrone", n_runs=1000, p0=0.5, B_list=[0.9, 0.95], p_list=[0.8, 0.9], comm_range_list=["sqrt(8)", 2], number_of_drones_list=[4, 8, 12, 16], n_targets_list=[3, 4, 5]):

    if test == "TC_GA":
        problem = "SOO"
        alg = "GA"
        obj = "TC"
    elif test == "TC_NSGA2":
        problem = "MOO"
        alg = "NSGA2"
        obj = "TC"
    elif test == "T_GA":
        problem = "SOO"
        alg = "GA"
        obj = "T"
    elif test == "TCDT_NSGA2":
        problem = "MOO" 
        alg = "NSGA2"
        obj = "TCDT"
    elif test == "TCDT_GA":
        problem = "SOO" 
        alg = "GA"
        obj = "TCDT"


    if test == "test":
        scenario = f"MOO_NSGA2_MTSP_TCDT_g_8_a_50_n_{16}_v_2.5_r_{'sqrt(8)'}_nvisits_{3}"
        sol_set = copy.deepcopy(load_pickle(f"{solutions_filepath}{scenario}-SolutionObjects.pkl")).flatten().tolist()
        F = copy.deepcopy(pd.read_pickle(f"{objective_values_filepath}{scenario}-ObjectiveValues.pkl"))
        best_sol = sol_set[F["Percentage Connectivity"].idxmin()] # Best connectivity solution of specificed scenario
        print(f"Solution Percentage Connectivity: {best_sol.percentage_connectivity}")
        best_sol_time_metrics = sensing_and_info_sharing(best_sol, merging_strategy="gcs", target_locations=[8,12], B=0.9, p=0.9, q=0.1)
        print(best_sol_time_metrics)
    else:
        # Put different B, p values on the same plots
        for B in B_list:
            for p in p_list:
                fig, ax = plt.subplots()
                q = 1-p
                m = ceil( log10((p0*(1-B))/(B*(1-p0))) / log10(q/p) )
                for comm_range in comm_range_list:
                    for n_targets in n_targets_list:
                        target_locations = np.random.choice(range(1, 64), n_targets, replace=False)
                        plot_details = f"T: {n_targets}, {target_locations},  $r_c$: {comm_range}" + "$n_{visits}$: " + str(m)
                        y_detection_time_list, y_inform_time_list, y_time_at_least_one_drone_knows_all_targets_list, y_success_rate = [], [], [], []
                        for i,number_of_drones in enumerate(number_of_drones_list):
                            sol_set = copy.deepcopy(load_pickle(f"{solutions_filepath}{problem}_{alg}_MTSP_{obj}_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{m}-SolutionObjects.pkl")).flatten().tolist()
                            F = copy.deepcopy(pd.read_pickle(f"{objective_values_filepath}{alg}_MTSP_{obj}_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{m}-ObjectiveValues.pkl"))
                            cols = F.columns()
                            for col in cols:
                                sol = sol_set[F[col].idxmin()]
                                time_metrics = sensing_and_info_sharing(sol=sol, merging_strategy=merging_strategy, target_locations=target_locations, B=B, p=p, q=1-p)

                    








        for B in B_list:
            for p in p_list:
                q = 1-p
                m = ceil( log10((p0*(1-B))/(B*(1-p0))) / log10(q/p) )
                for comm_range in comm_range_list:
                    for n_targets in n_targets_list:
                        target_locations = np.random.choice(range(1, 64), n_targets, replace=False)
                        y_detection_time_list, y_inform_time_list, y_time_at_least_one_drone_knows_all_targetslist, y_success_rate = [], [], [], []
                        for i,number_of_drones in enumerate(number_of_drones_list):
                            plot_details = f"B: {np.round(B,2)}, p: {np.round(p,2)},  q: {np.round(q,2)}, T: {n_targets}, {target_locations},  $r_c$: {comm_range}" + "$n_{visits}$: " + str(m)
                            sol_set = copy.deepcopy(load_pickle(f"{solutions_filepath}{problem}_{alg}_MTSP_{obj}_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{m}-SolutionObjects.pkl")).flatten().tolist()
                            F = copy.deepcopy(pd.read_pickle(f"{objective_values_filepath}{alg}_MTSP_{obj}_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{m}-ObjectiveValues.pkl"))
                            cols = F.columns()
                            best_sol_dict = {col:sol_set[F[col].idxmin()] for col in cols}
                            best_sol_time_metric_dict = {col:sensing_and_info_sharing(best_sol_dict[col], merging_strategy="ondrone", target_locations=target_locations, B=B, p=p, q=1-p) for col in cols}
                            for run_no in range(n_runs):
                                print(f"Run {run_no+1}")

                                for col in F.columns():
                                    y_detection_time_dict[col] = np.zeros(len(number_of_drones_list))
                                        
                                    best_sol_dict[col] = sol_set[F[col].idxmin()] # Best connectivity solution of specificed scenario
                                    best_sol_time_metric_dict[col] = sensing_and_info_sharing(best_sol_dict[col], merging_strategy="ondrone", target_locations=target_locations, B=B, p=p, q=1-p)
                                    y_detection_time_dict[col][i] += best_sol_time_metrics["detection time"] if best_sol_time_metrics["detection time"] != inf else 0
                                    y_inform_time_list[i] += best_sol_time_metrics["inform time"] if best_sol_time_metrics["inform time"] != inf else 0
                                    y_successful_runs_list[i] = y_successful_runs_list[i] + 1 if best_sol_time_metrics["inform time"] != inf else y_successful_runs_list[i]

                                            # best_sol_time_metrics = sensing_and_info_sharing(best_sol, merging_strategy="ondrone", target_locations=target_locations, B=B, p=p, q=1-p)

                                # best_sol_time_metrics = sensing_and_info_sharing(best_sol, merging_strategy="ondrone", target_locations=[8,12], B=B, p=p, q=1-p)
                                    # print(best_sol_time_metrics)
                                    # y_time_at_least_one_drone_knows_all_targets_list[i] += best_sol_time_metrics["Time BS knows at least one target"] if best_sol_time_metrics["Time BS knows at least one target"] != inf else 0
                                    if best_sol_time_metrics["inform time"] != inf:
                                        y_successful_runs[i] += 1
                            y_detection_time_list = [x/n_runs for x in y_detection_time_list]
                            y_inform_time_list = [x/n_runs for x in y_inform_time_list]
                            y_time_at_least_one_drone_knows_all_targets_list = [x/n_runs for x in y_time_at_least_one_drone_knows_all_targets_list]
                            y_success_rate = [x/n_runs for x in y_successful_runs]


                            # Plot Detection Time
                            fig,ax = plt.subplots(figsize=(8, 6))
                            fig.suptitle(f"{plot_details}")
                            plt.plot(number_of_drones_list, y_detection_time_list, label="Detection Time")
                            plt.title("Detection Time Gain")
                            plt.xlabel("Number of Drones")
                            plt.ylabel("Detection Time (s)")
                            plt.xticks(number_of_drones_list)
                            plt.grid()
                            plt.legend()
                            plt.savefig(f"Figures/Sensing/{plot_details}_detection_time.png")
                            plt.show()
                            plt.close()

                            # Plot Inform Time
                            fig,ax = plt.subplots(figsize=(8, 6))
                            fig.suptitle(f"{plot_details}")
                            plt.plot(number_of_drones_list, y_inform_time_list, label="Inform Time")
                            plt.title("Inform Time (s)")
                            plt.xlabel("Number of Drones")
                            plt.ylabel("Inform Time (s)")
                            plt.xticks(number_of_drones_list)
                            plt.grid()
                            plt.legend()
                            plt.savefig(f"Figures/Sensing/{plot_details}_inform_time.png")
                            plt.show()
                            plt.close()

                            # Plot Time at least one drone knows all targets
                            fig,ax = plt.subplots(figsize=(8, 6))
                            fig.suptitle(f"{plot_details}")
                            plt.plot(number_of_drones_list, y_time_at_least_one_drone_knows_all_targets_list, label="Time at least one drone knows all targets")
                            plt.title("Time at least one drone knows all targets (s)")
                            plt.xlabel("Number of Drones")
                            plt.ylabel("Time at least one drone knows all targets (s)")
                            plt.xticks(number_of_drones_list)
                            plt.grid()
                            plt.legend()
                            plt.savefig(f"Figures/Sensing/{plot_details}_time_at_least_one_drone_knows_all_targets.png")
                            plt.show()
                            plt.close()

                            # Plot Success Rate
                            fig,ax = plt.subplots(figsize=(8, 6))
                            fig.suptitle(f"{plot_details}")
                            plt.plot([4, 8, 12, 16], y_success_rate, label="Success Rate")
                            plt.title("Success Rate")
                            plt.xlabel("Number of Drones")
                            plt.ylabel("Success Rate")
                            plt.xticks(number_of_drones_list)
                            plt.grid()
                            plt.legend()
                            plt.savefig(f"Figures/Sensing/{plot_details}_success_rate.png")



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
                        

run_tests("multi")