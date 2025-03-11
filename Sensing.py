import numpy as np
import pandas as pd
from Results import save_best_solutions
from PathInfo import *
from FilePaths import *
from PathFileManagement import load_pickle
from Connectivity import connected_components, PathSolution
# from Distance import interpolate_between_cities
from array_operations import create_array_of_lists

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

def belief_merging(sol:PathSolution, conn_comp, search_map):
    """Update coordinated search map according to the last measurement made"""
    for clique in conn_comp:
        clique_search_maps = search_map[clique]
        for cell in range(sol.info.number_of_cells):
            clique_cell_beliefs = clique_search_maps[:, cell]
            num_cell_obs = [len(x) for x in clique_cell_beliefs]
            max_cell_obs_drone_ids = np.where(num_cell_obs == max(num_cell_obs))[0]
            if len(max_cell_obs_drone_ids) > 1:
                cell_belief_to_brodcast = np.mean(clique_cell_beliefs[max_cell_obs_drone_ids][:,-1], axis=0)
            else:
                cell_belief_to_brodcast = clique_cell_beliefs[max_cell_obs_drone_ids[0]][-1]
            for node_cell_beliefs in clique_cell_beliefs:
                node_cell_beliefs[-1] = cell_belief_to_brodcast
            

def average_merging():
    pass

def occ_grid_merging():
    pass

def sensed_data_merging():
    pass


def info_sharing(sol:PathSolution, target_locations=[12], B=0.9, p=0.9, q=0.2):
    """
    Integrate search map to already generated pathplans
    sol: PathSolution object
    target_locations: list of target locations
    B: occupancy threshold
    p: probability of detection
    q: probability of false alarm
    """
    # Make a deep copy of the solution object to avoid modifying the original object
    x = deepcopy(sol)
    info = x.info
    # Slice path matrix in such a way that only steps between the 1st step and the first hovering steps are taken because we do'nt take measurements while hovering
    # and at the first step because each drone goes to cell 0 at the first step to avoid going outside the map
    hovering_cells = [seq[-2] for seq in list(x.drone_dict.values())]
    final_search_steps = [np.where(x.real_time_path_matrix[i+1,2:]==hovering_cells[i])[0][info.n_visits-1]+2 for i in range(info.number_of_drones)]
    # path_seq = np.array(x.path)
    # cell_0_indices = np.where(path_seq==0)[0]
    # drones_with_cell_0_as_first_step = []
    # for idx in cell_0_indices:
    #     drones_with_cell_0_as_first_step.append(x.start_points.tolist().index(idx)) if idx in x.start_points else None

    drone_path_matrix = x.real_time_path_matrix[1:,:]
    number_of_drones, timesteps = drone_path_matrix.shape
    connectivity_matrix = x.connectivity_matrix
    info = x.info
    # Initialize search maps (Async. and sync.)
    async_search_map = np.empty((sol.info.number_of_nodes, sol.info.number_of_cells), dtype=object)
    rows, cols = async_search_map.shape
    for i in range(rows):
        for j in range(cols):
            async_search_map[i, j] = {"n_obs":0, "timestep":-1, "prob":0.5} # Start at timestep=-1 to indicate that this is the prior belief
    sync_search_map = async_search_map.copy()
    for step in range(max(final_search_steps)):
        adj_mat = connectivity_matrix[step]
        connected_nodes = connected_components(adj_mat)
        # TODO: Asynchronous Search Map Update
        for drone in range(number_of_drones):
            # If drone is in hovering state, continue
            if step > final_search_steps[drone]:
                continue
            else:
                async_search_map[drone+1, drone_position]["timestep"] = step # Update timestep
                async_search_map[drone+1, drone_position]["n_obs"] += 1 # Increment number of observations (i.e. n_obs)
                # Update occupancy probability in the async. search map
                drone_position = drone_path_matrix[drone, step]
                prior_obs = async_search_map[drone+1, drone_position]["prob"]
                if drone_position in target_locations:
                    new_obs = p*prior_obs / (p*prior_obs + q*(1-prior_obs))
                else:
                    new_obs = (1-p)*prior_obs / ((1-p)*prior_obs + (1-q)*(1-prior_obs))
                async_search_map[drone+1, drone_position] = {"timestep":step, "occ_prob":new_obs}
        # TODO: Info Merging
        for clique in connected_nodes:
            # print(f"Clique: {clique}")
            SM_clique = SM[clique]
            # print(f"Pre-Merge: {SM_clique}")
            for cell in range(info.number_of_cells):
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

                        






def generate_sar_paths(sol:PathSolution, merging_strategy="belief", target_locations=[12], B=0.9, p=0.9, q=0.2):
    x = deepcopy(sol)
    real_time_path_matrix = x.real_time_path_matrix
    connectivity_matrix = x.connectivity_matrix
    info = x.info
    # Initialize search map
    search_map = np.empty((sol.info.number_of_nodes, sol.info.number_of_cells), dtype=object)
    rows, cols = search_map.shape
    for i in range(rows):
        for j in range(cols):
            search_map[i, j] = {"timestep":-1, "belief":0.5, "num_obs":0}  # Assign an empty list to each cell
    # search_map = create_array_of_lists(rows=info.number_of_nodes, cols=info.number_of_cells, fill_value=0.5)
    # search_map = np.full(shape=(info.number_of_nodes, info.number_of_cells), fill_value=[0.5], dtype=list)    # np.full(shape=(info.n_visits, info.number_of_nodes, info.number_of_cells), fill_value=0.5, dtype=float) # Occupancy grid initialization
    for step in range(real_time_path_matrix.shape[1]):
        node_positions = real_time_path_matrix[:,step]
        drone_positions = real_time_path_matrix[1:,step]
        # First, do uncoordinated search map updates
        for drone in range(info.number_of_drones):
            drone_position = drone_positions[drone]
            # First update the timestep of the drone's last observation
            search_map[drone+1, drone_position]["timestep"] = step
            # Then, calculate new cell occupancy belief according to whether the drone is at a target location or not
            prior_obs = search_map[drone+1, drone_position]["belief"]
            if drone_position in target_locations:
                new_obs = p*prior_obs / (p*prior_obs + q*(1-prior_obs))
            else:
                new_obs = (1-p)*prior_obs / ((1-p)*prior_obs + (1-q)*(1-prior_obs))
            # Set the new observation in drone's search map
            search_map[drone+1, drone_position]["belief"] = new_obs
            # Update the number of observations attribute
            search_map[drone+1, drone_position]["num_obs"] += 1

        # Share information
        adj = connectivity_matrix[step]
        conn_comp = connected_components(adj)
        # print(f"Adj Mat:\n{adj}\nConn Comp:\n{conn_comp}")
        for clique in conn_comp:
            # nodes_in_cliuqe = clique.copy()
            clique_positions = real_time_path_matrix[clique, step]
            if len(np.unique(clique_positions)) == len(clique_positions):
                print(f"Unique visits - clique: {clique}, clique positions: {clique_positions}")
                # If all drones in the clique are in different cells, share their search maps
                for node in clique:
                    receivers = [x for x in clique if x != node]
                    cell_no = node_positions[node]
                    msg = search_map[node][node_positions[node]] 
                    for receiver_node in receivers:
                        search_map[receiver_node, cell_no] = msg
            else:
                print(f"Concurrent visits detected ! - clique: {clique}, clique positions: {clique_positions}")

    #     print(f"step {step} completed")
    # print("Search map generation completed")
    # for loc in target_locations: 
    #     print(f"target cell {loc} search map: {search_map[:,loc]}")



"""
        if merging_strategy == "belief":
            belief_merging(sol=sol, conn_comp=conn_comp, search_map=search_map)
        elif merging_strategy == "average":
            average_merging(sol=sol, conn_comp=conn_comp, search_map=search_map)
        elif merging_strategy == "occ_grid":
            occ_grid_merging(sol=sol, conn_comp=conn_comp, search_map=search_map)
        elif merging_strategy == "sensed_data":
            sensed_data_merging(sol=sol, conn_comp=conn_comp, search_map=search_map)
"""
        

def update_path_for_detection(self):

    real_time_path_matrix = self.real_time_path_matrix.copy()
    connectivity_matrix = self.connectivity_matrix.copy()
    hovering_cells = np.unique([self.drone_dict[i][-3] for i in range(self.info.number_of_drones)])
    hovering_cells_dists_to_bs = [self.info.D[hovering_cell, -1] for hovering_cell in hovering_cells]
    hovering_cells_dists_to_bs_sorted = np.unique(sorted(hovering_cells_dists_to_bs))
    closest_hovering_cells_to_bs = []
    for i in range(len(hovering_cells_dists_to_bs_sorted)):
        indices = np.where(hovering_cells_dists_to_bs == hovering_cells_dists_to_bs_sorted[i])[0]
        for ind in indices:
            closest_hovering_cells_to_bs.append(hovering_cells[ind])
        # closest_hovering_cells_to_bs_indices.extend(np.where(hovering_cells_dists_to_bs == hovering_cells_dists_to_bs_sorted[i])[0])
    available_hovering_cells = closest_hovering_cells_to_bs.copy()
    # print(closest_hovering_cells_to_bs)
    
    info = self.info
    # print(f"Path before detection:\n{self.real_time_path_matrix
    occ_grid = np.full(shape=(info.number_of_nodes, info.number_of_cells), fill_value=0.5, dtype=float) # Occupancy grid initialization
    new_drone_paths_dict = self.drone_dict.copy()
    drone_search_status = {i: True for i in range(info.number_of_drones)}
    for step in range(real_time_path_matrix.shape[1]):
        # Check if all drones have found all target locations and they've returned to BS
        if True not in drone_search_status.values():
            break
        adj = connectivity_matrix[step]
        connected_nodes = connected_components(adj)
        drone_positions = real_time_path_matrix[1:,step]
        for drone_no, drone_position in enumerate(drone_positions):
            drone_bs_connectivity = False
            # First, update individual occupancy grid for each drone
            if drone_position in info.target_locations:
                # This is where we would apply the bayesian occ prob update, for now we just set it to 1
                occ_grid[drone_no + 1, drone_position] = 1
            # Next, update the occupancy grids for each connected component (for now, take the average)
            for conn_comp in connected_nodes:
                drone_bs_connectivity = True if drone_no+1 in conn_comp and 0 in conn_comp else None
                # if drone_no+1 in conn_comp and 0 in conn_comp:
                #     drone_bs_connectivity = True
                new_occ_grids = np.zeros(info.number_of_cells)
                for node in conn_comp:
                    new_occ_grids += occ_grid[node]
                new_occ_grids /= len(conn_comp)
                for node in conn_comp:
                    occ_grid[node] = new_occ_grids
            # If drone knows all target locations, fly back to BS
            if len(np.where(occ_grid[drone_no+1]>=info.th)[0]) >= len(info.target_locations) or drone_bs_connectivity and len(np.where(occ_grid[0]>=info.th)[0]) >= len(info.target_locations) and drone_search_status[drone_no]: # If drone knows all target locations OR BS knows all target locations and is connected to the drone, return to BS
                drone_search_status[drone_no] = False
                if len(available_hovering_cells) != 0:
                    relay_position = available_hovering_cells[0]
                    available_hovering_cells.pop(0)
                    path_to_relay_position = interpolate_between_cities(self, drone_position, relay_position)
                    path_back_to_bs = np.hstack(( path_to_relay_position, interpolate_between_cities(self, relay_position, 0), -1 ))
                    # print(path_back_to_bs)
                else:
                    if drone_position == 0:
                        path_back_to_bs = np.array([-1])
                    elif drone_position == -1:
                        path_back_to_bs = np.array([])
                    else:
                        path_back_to_bs = np.hstack(( interpolate_between_cities(self, drone_position, 0), -1 ))
                new_drone_path = np.hstack(( real_time_path_matrix[drone_no+1,:step], path_back_to_bs ))
                # print(new_drone_path)
                new_drone_paths_dict[drone_no] = new_drone_path
    self.drone_dict = new_drone_paths_dict
    self.get_pathplan()
    print(f"Path after detection:\n{self.real_time_path_matrix}")
                
                
                # path_back_to_bs.append(-1)
                # print(drone_position)
                # print(path_back_to_bs)
                # continue
                # Interpolate between current position and BS (0. cell)
                # real_time_path_matrix[drone_no+1, step+1:] = -1

# test
number_of_drones = 8
nvisits = 3
comm_range = "sqrt(8)"
model = "TCDT"
sol_set = deepcopy(load_pickle(f"{solutions_filepath}MOO_NSGA2_MTSP_{model}_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{nvisits}-SolutionObjects.pkl").flatten().tolist())
F = deepcopy(pd.read_pickle(f"{objective_values_filepath}MOO_NSGA2_MTSP_{model}_g_8_a_50_n_{number_of_drones}_v_2.5_r_{comm_range}_nvisits_{nvisits}-ObjectiveValues.pkl"))
best_conn_idx = F["Percentage Connectivity"].idxmin()
best_conn_sol = sol_set[best_conn_idx]
random_sol = np.random.choice(sol_set)

# generate_sar_paths(sample_sol, merging_strategy="belief", target_locations=[12,15,8], B=0.9, p=0.9, q=0.2)
info_sharing(best_conn_sol, target_locations=[12,15,8], B=0.9, p=0.9, q=0.2)