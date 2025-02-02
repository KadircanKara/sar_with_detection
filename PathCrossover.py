# Comprehensive Sequential Constructive Crossover (CSCX)
from typing import Dict
import numpy as np
import random
from time import sleep
import os
from scipy.signal import convolve

from PathSolution import *

from pymoo.core.crossover import Crossover
from pymoo.core.variable import Real, get



def find_subarray_convolve(arr, subarr):
    subarr_len = len(subarr)
    if subarr_len > len(arr):
        return False

    # Convolve the array with the original and reversed subarray
    conv_result_original = convolve(arr, subarr[::-1], mode='valid')
    conv_result_reverse = convolve(arr, subarr, mode='valid')

    # Check for the sum of squares match
    match_original = (conv_result_original == np.sum(subarr ** 2))
    match_reverse = (conv_result_reverse == np.sum(subarr[::-1] ** 2))

    # Return True if either match is found
    return np.any(match_original) or np.any(match_reverse)


# Pymoo Defined Crossovers

def random_sequence(n):
    start, end = np.sort(np.random.choice(n, 2, replace=False))
    return tuple([start, end])


def ox(receiver, donor, seq=None, shift=False):
    """
    The Ordered Crossover (OX) as explained in http://www.dmi.unict.it/mpavone/nc-cs/materiale/moscato89.pdf.

    Parameters
    ----------
    receiver : numpy.array
        The receiver of the sequence. The array needs to be repaired after the donation took place.
    donor : numpy.array
        The donor of the sequence.
    seq : tuple (optional)
        Tuple with two problems defining the start and the end of the sequence. Please note in our implementation
        the end of the sequence is included. The sequence is randomly chosen if not provided.

    shift : bool
        Whether during the repair the receiver should be shifted or not. Both version of it can be found in the
        literature.

    Returns
    -------

    y : numpy.array
        The offspring which was created by the ordered crossover.

    """
    assert len(donor) == len(receiver)

    # print(f"donor: {donor}\nreceiver: {receiver}")

    # the sequence which shall be use for the crossover
    seq = seq if not None else random_sequence(len(receiver))
    start, end = seq

    # print(f"seq: {seq}")

    # the donation and a set of it to allow a quick lookup
    donation = np.copy(donor[start:end + 1])
    donation_as_set = set(donation)

    # the final value to be returned
    y = []

    for k in range(len(receiver)):

        # print(f"iteration {k}: {y}")

        # do the shift starting from the swapped sequence - as proposed in the paper
        i = k if not shift else (start + k) % len(receiver)
        v = receiver[i]

        if v not in donation_as_set:
            y.append(v)

    # now insert the donation at the right place
    y = np.concatenate([y[:start], donation, y[start:]]).astype(copy=False, dtype=int).tolist()

    # print(f"Offspring: {y}")

    return y

# My Crossovers

# Update Crossover Probabilities

def scx_crossover_test(p1:PathSolution, p2:PathSolution):

    info = p1.info

    p1_path = p1.path
    p2_path = p2.path

    size = len(p1_path)
    offspring = [-1] * size  # Initialize offspring with -1 indicating unvisited cities
    offspring[0] = p1_path[0]  # Start with the first city of parent1

    current_city_index = 0  # Start from the first city
    for i in range(1, size):
        # Look for the next city in parent1 that is not already in offspring
        next_city1 = p1_path[(current_city_index + 1) % size]
        while next_city1 in offspring:
            current_city_index = (current_city_index + 1) % size
            next_city1 = p1_path[(current_city_index + 1) % size]

        # Look for the next city in parent2 that is not already in offspring
        next_city2 = p2_path[(current_city_index + 1) % size]
        while next_city2 in offspring:
            current_city_index = (current_city_index + 1) % size
            next_city2 = p2_path[(current_city_index + 1) % size]

        # Select the next city based on a predefined criterion (e.g., alternating between parents)
        # print("-->",info.D[offspring[-1] % info.number_of_cells])
        if info.D[offspring[-1] % info.number_of_cells, next_city1 % info.number_of_cells] <= info.D[offspring[-1] % info.number_of_cells, next_city2 % info.number_of_cells]:
            offspring[i] = next_city1
        else:
            offspring[i] = next_city2
        # if i % 2 == 0:
        #     offspring[i] = next_city1
        # else:
        #     offspring[i] = next_city2

        current_city_index = (current_city_index + 1) % size

    p1_sp_sol, p2_sp_sol = PathSolution(offspring, p1.start_points, info), PathSolution(offspring, p2.start_points, info)

    return p1_sp_sol, p2_sp_sol

    # if p1_sp_sol.total_distance <= p2_sp_sol.total_distance:
    #     return p1_sp_sol
    # else:
    #     return p2_sp_sol

    # return offspring


# Sequential Combination Crossover (1-offspring)
'''def scx_crossover(p1: PathSolution, p2: PathSolution):

    info = p1.info

    p1_path = p1.path
    p2_path = p2.path

    size = len(p1_path)
    offspring = [-1] * size  # Initialize offspring with -1 indicating unvisited cities
    offspring[0] = p1_path[0]  # Start with the first city of parent1

    current_city_index = 0  # Start from the first city

    # Keep track of the number of times each city has appeared so far
    p1_city_count = {city: p1_path.count(city) for city in set(p1_path)}
    p2_city_count = {city: p2_path.count(city) for city in set(p2_path)}
    offspring_city_count = {city: 0 for city in set(p1_path).union(set(p2_path))}

    for i in range(1, size):
        last_city = offspring[i-1]

        # Look for the next city in parent1 that is allowed in offspring
        next_city1_index = (current_city_index + 1) % size
        next_city1 = p1_path[next_city1_index]
        while offspring_city_count[next_city1] >= p1_city_count[next_city1]:
            next_city1_index = (next_city1_index + 1) % size
            next_city1 = p1_path[next_city1_index]

        # Look for the next city in parent2 that is allowed in offspring
        next_city2_index = (current_city_index + 1) % size
        next_city2 = p2_path[next_city2_index]
        while offspring_city_count[next_city2] >= p2_city_count[next_city2]:
            next_city2_index = (next_city2_index + 1) % size
            next_city2 = p2_path[next_city2_index]

        # Choose the city with the shorter distance from the last added city
        if info.D[last_city % info.number_of_cells, next_city1 % info.number_of_cells] <= info.D[last_city % info.number_of_cells, next_city2 % info.number_of_cells]:
            offspring[i] = next_city1
            offspring_city_count[next_city1] += 1
        else:
            offspring[i] = next_city2
            offspring_city_count[next_city2] += 1

        current_city_index = (current_city_index + 1) % size

    # Double-check if all offspring positions are filled
    for idx, city in enumerate(offspring):
        if city == -1:
            raise ValueError(f"Offspring city at index {idx} is not filled properly.")
        
    print(f"offspring: {offspring}, len unique: {len(np.unique(offspring))}")

    return PathSolution(offspring, p1.start_points, info), PathSolution(offspring, p2.start_points, info)
'''
def scx_crossover(p1:PathSolution, p2:PathSolution):
    info = p1.info

    p1_path = p1.path
    p2_path = p2.path

    size = len(p1_path)
    offspring = [-1] * size  # Initialize offspring with -1 indicating unvisited cities
    offspring[0] = p1_path[0]  # Start with the first city of parent1

    current_city_index = 0  # Start from the first city
    for i in range(1, size):
        # Look for the next city in parent1 that is not already in offspring
        next_city1 = p1_path[(current_city_index + 1) % size]
        while next_city1 in offspring:
            current_city_index = (current_city_index + 1) % size
            next_city1 = p1_path[(current_city_index + 1) % size]

        # Look for the next city in parent2 that is not already in offspring
        next_city2 = p2_path[(current_city_index + 1) % size]
        while next_city2 in offspring:
            current_city_index = (current_city_index + 1) % size
            next_city2 = p2_path[(current_city_index + 1) % size]

        # Select the next city based on a predefined criterion (e.g., alternating between parents)
        if info.D[offspring[-1] % info.number_of_cells, next_city1 % info.number_of_cells] <= info.D[offspring[-1] % info.number_of_cells, next_city2 % info.number_of_cells]:
            offspring[i] = next_city1
        else:
            offspring[i] = next_city2

        current_city_index = (current_city_index + 1) % size

    p1_sp_sol, p2_sp_sol = PathSolution(offspring, p1.start_points, info), PathSolution(offspring, p2.start_points, info)

    return p1_sp_sol, p2_sp_sol


# Ordered Crossover (2-offsprings)
def ox_crossover(p1: PathSolution, p2: PathSolution, n_offsprings:int):
    
    info = p1.info
    p1_path = p1.path
    p2_path = p2.path
    
    # Randomly select the crossover points
    start, end = random_sequence(len(p1_path))

    # Get the subsequences from both parents
    p1_seq = p1_path[start:end]
    p2_seq = p2_path[start:end]

    # Initialize offspring arrays with None values
    offspring_1 = [None] * len(p1_path)
    offspring_2 = [None] * len(p2_path)

    # Insert the selected subsequences into the offspring
    offspring_1[start:end] = p1_seq
    offspring_2[start:end] = p2_seq

    def fill_offspring(offspring, parent_path, start, end):
        parent_index = 0
        for i in range(len(offspring)):
            # Only fill the positions outside the selected subsequence
            if i < start or i >= end:
                # Find the next element from the parent that maintains the full order
                while offspring.count(parent_path[parent_index]) >= p1_path.count(parent_path[parent_index]):
                    parent_index += 1
                offspring[i] = parent_path[parent_index]
                parent_index += 1

    # Fill the remaining elements for both offspring, preserving the order and allowing duplicates
    fill_offspring(offspring_1, p2_path, start, end)
    fill_offspring(offspring_2, p1_path, start, end)

    if n_offsprings == 2:
        return PathSolution(offspring_1, p1.start_points, info), PathSolution(offspring_2, p2.start_points, info)# , PathSolution(offspring_1, p2.start_points, info), PathSolution(offspring_2, p1.start_points, info)
    elif n_offsprings == 4:
        return PathSolution(offspring_1, p1.start_points, info), PathSolution(offspring_2, p2.start_points, info), PathSolution(offspring_1, p2.start_points, info), PathSolution(offspring_2, p1.start_points, info)

'''def ox_crossover(p1:PathSolution, p2:PathSolution):

    
    info = p1.info

    p1_path = p1.path
    p2_path = p2.path

    seq = random_sequence(len(p1_path))
    start, end = seq

    p1_seq = p1_path[start:end]
    p2_seq = p2_path[start:end]

    offspring_1 = [None] * len(p1_path) # path 1 seq, rest from path 2
    offspring_2 = [None] * len(p2_path) # path 2 seq, rest from path 1

    offspring_1[start:end] = p1_seq
    offspring_2[start:end] = p2_seq

    for i in range(len(offspring_1)):

        path_1_ind, path_2_ind = 0, 0

        if i < start or i >= end:

            cell_from_path_2 = p2_path[path_2_ind]
            while(cell_from_path_2 in offspring_1):
                path_2_ind += 1
                print(f"p2_path: {p2_path}")
                cell_from_path_2 = p2_path[path_2_ind]
            offspring_1[i] = cell_from_path_2

            cell_from_path_1 = p1_path[path_1_ind]
            while(cell_from_path_1 in offspring_2):
                path_1_ind += 1
                cell_from_path_1 = p1_path[path_1_ind]
            offspring_2[i] = cell_from_path_1

    return PathSolution(offspring_1, p1.start_points, info), PathSolution(offspring_2, p2.start_points, info)'''


'''
# Partially Mapped Crossover (2-offsprings) (NOT USED AT THE MOMENT !)
def pmx_crossover(p1:PathSolution, p2:PathSolution):

    info = p1.info

    p1_path = p1.path
    p2_path = p2.path

    offspring_1 = [None] * len(p1.path)
    offspring_2 = offspring_1.copy()

    ox_point = random.choice(np.arange(1, len(p1.path)-1, 1))
    offspring_1 = np.hstack((p1.path[:ox_point], p2.path[ox_point:]))
    offspring_2 = np.hstack((p2.path[:ox_point], p1.path[ox_point:]))

    # IF THERE ARE DUPLICATE CELLS, REPEAT THE PROCESS !!!
    while(len(np.unique(offspring_1)) != len(p1.path) and len(np.unique(offspring_2)) != len(p2.path)):
        print("IN PMX !!!")
        ox_point = random.choice(np.arange(1, len(p1.path)-1, 1))
        offspring_1 = np.hstack((p1.path[:ox_point], p2.path[ox_point:]))
        offspring_2 = np.hstack((p2.path[:ox_point], p1.path[ox_point:]))

    # print(f"1st component length: {len(p1.path[:ox_point])} 2nd component length: {len(p2.path[ox_point:])}")

    # print(f"ox point: {ox_point}\np1: {p1.path}\np2: {p2.path}\noffspring 1: {offspring_1}\noffspring 2: {offspring_2}")

    return PathSolution(offspring_1, p1.start_points, info), PathSolution(offspring_2, p2.start_points, info)

# Edge Recombination Crossover (1-offspring)
def erx_crossover(p1:PathSolution, p2:PathSolution):

    info = p1.info

    p1_path, p2_path = np.asarray(p1.path), np.asarray(p2.path)

    # Step 1: Get the neighbors for each cell for each parents (utilize dicts)
    # Initialize edge dicts for both parents
    p1_edges = dict()
    p2_edges = p1_edges.copy()
    edges_set = p1_edges.copy()
    # print(f"p1: {p1_path} p1 duplicate check: {np.unique(p1_path)}\np2: {p2_path} p2 len: {len(p2_path)==len(np.unique(p2_path))}")
    for cell in range(info.number_of_cells):
        # print(f"cell: {cell}")
        # print(np.where(p1_path==cell), np.where(p2_path==cell))
        p1_cell_ind, p2_cell_ind = np.where(p1_path==cell)[0][0], np.where(p2_path==cell)[0][0]
        # p1_cell_ind, p2_cell_ind = p1_path.index(cell), p2_path.index(cell)
        # Check if cell is located at the end-points or not (edge-case)
        # p1_isedge, p2_isedge = bool(p1_path.index(cell)==len(p1_path)-1 or p1_path.index(cell)==0), bool(p2_path.index(cell)==len(p2_path)-1 or p2_path.index(cell)==0)
        # Neighbors for parent 1
        if p1_cell_ind==len(p1_path)-1:
            p1_n1, p1_n2 = p1_path[0], p1_path[p1_cell_ind-1]
        elif p1_cell_ind==0:
            p1_n1, p1_n2 = p1_path[-1], p1_path[1]
        else:
            p1_n1, p1_n2 = p1_path[p1_cell_ind-1], p1_path[p1_cell_ind+1]
        # Neighbors for parent 2
        if p2_cell_ind==len(p2_path)-1:
            p2_n1, p2_n2 = p2_path[0], p2_path[p2_cell_ind-1]
        elif p1_cell_ind==0:
            p2_n1, p2_n2 = p2_path[-1], p2_path[1]
        else:
            p2_n1, p2_n2 = p2_path[p2_cell_ind-1], p2_path[p2_cell_ind+1]
        # Set dict values for both parents
        p1_edges[cell] = [p1_n1, p1_n2]
        p2_edges[cell] = [p2_n1, p2_n2]
        edges_set[cell] = list(set([p1_n1, p1_n2, p2_n1, p2_n2]))
    # Edge dicts are set, now pick a starting node at random (either the first parent's or the second parent's)
    if random.random() <= 0.5:
        offspring = [p1_path[0]]
    else:
        offspring = [p2_path[0]]
    # print(f"starting cell: {offspring[-1]}")
    # print(f"initial edges set:\n{edges_set}")
    # Remove starting node from the edge lists for each cell key
    for cell in edges_set:
        edges_set[cell] = [neighbor_list for neighbor_list in edges_set[cell] if neighbor_list != offspring[-1]]
    # print(f"edges set first update:\n{edges_set}")
    # Do the recombination from the edges dict
    while(len(offspring) < len(p1_path)):
        # print(f"offspring duplicate check: {len(offspring)==len(np.unique(offspring))}")
        # print(f"offspring: {offspring}")
        # print(f"-->", edges_set[offspring[-1]])
        neighbors = edges_set[offspring[-1]]
        # If neighbor list is empty, pick a random cell that has neighbors
        if len(neighbors) == 0:
            # Get cells that still have neighbors
            non_empty_edges_set = {c: n for c, n in edges_set.items() if n}
            neighbors = random.choice(list(non_empty_edges_set.values()))
            # neighbors = non_empty_edges_set[]
        neighbor_choice = neighbors[0]
        for i in range(len(neighbors)-1):
            if len(edges_set[neighbors[i+1]]) < len(edges_set[neighbors[i]]):
                neighbor_choice = neighbors[i]
        # If neighbor edge lengths are all equal, choose at random
        if neighbor_choice == neighbors [0]:
            neighbor_choice = random.choice(neighbors)
        # print(f"neighbor choice: {neighbor_choice}")
            # if info.D[offspring[-1] % info.number_of_cells, n % info.number_of_cells] < info.D[offspring[-1] % info.number_of_cells, neighbor_choice % info.number_of_cells]:
            #     neighbor_choice = n
        # Remove all occurances of neighbor_choice from the edge set
        for cell in edges_set:
            edges_set[cell] = [neighbor_list for neighbor_list in edges_set[cell] if neighbor_list != neighbor_choice]
        # print(f"edges set later update:\n{edges_set}")
        # Update offspring
        offspring.append(neighbor_choice)
    # print(f"offspring length {len(offspring)}")
    # Try solutions with both start_points, return the one with less total_distance
    p1_sp_sol, p2_sp_sol = PathSolution(offspring, p1.start_points, info), PathSolution(offspring, p2.start_points, info)
    # print(p1_sp_sol, p2_sp_sol)

    return p1_sp_sol, p2_sp_sol

    # if p1_sp_sol.total_distance <= p2_sp_sol.total_distance:
    #     return p1_sp_sol
    # else:
    #     return p2_sp_sol

# Distance Preserving Crossover (1-offspring) (NOT USED AT THE MOMENT !)
def dpx_crossover(p1:PathSolution, p2:PathSolution):
    # Get common edges of both parents and shuffle them to generate the offspring
    info = p1.info
    # fragments = np.array([]) # initialize empty array
    fragments = []
    p1_path, p2_path = p1.path, p2.path
    # print(f"p1: {p1_path}\np2: {p2_path}")
    i = 0
    j = 0
    for i in range(len(p1_path)-1):
        fragment = np.array([p1_path[i]])
        fragments.append(fragment)
        # print(f"initial fragment: {fragment}")
        while(find_subarray_convolve(p2_path, np.hstack((fragment, p1_path[i+1])))):
            fragment = np.hstack((fragment, p2_path[j]))
            i += 1
        # for j in range(i+1,len(p1_path)):
        #     # subpath = fragment + p1_path[j]
        #     subpath = np.hstack((fragment, p1_path[j])) # fragment.append(p1_path[j])
        #     if find_subarray_convolve(p2_path, subpath):
        #         # fragment.append(p1_path[j])
        #         fragment = np.hstack((fragment, p2_path[j]))
        #     else:
        #         break
        # print(f"final fragment {i+1}: {fragment}")
        # fragments = np.hstack((fragments, fragment))
        fragments.append(fragment)
    # print(f"fragments: {fragments}")
    if p1_path[-1] not in fragments[-1]:
        # fragments = np.hstack((fragments, np.array([p1_path[-1]])))
        fragments.append([p1_path[-1]])
    # Now shuffle the fragments
    # Create a list of indices
    indices = list(range(len(fragments)))
    # Shuffle the list of indices
    random.shuffle(indices)
    # Reorder the sublists using the shuffled indices
    shuffled_fragments = [fragments[i] for i in indices]
    offspring = [item for sublist in shuffled_fragments for item in sublist]

    if not len(np.unique(offspring))==len(p1_path):
        print("Offspring Invalid !")
        print(f"p1: {p1_path}\np2: {p2_path}\nfragments: {fragments}\nshuffled fragments: {shuffled_fragments}\noffspring: {offspring}")
        # sleep(0.5)
    else:
        print("Offspring Valid !")
        print(f"p1: {p1_path}\np2: {p2_path}\nfragments: {fragments}\nshuffled fragments: {shuffled_fragments}\noffspring: {offspring}")
        # sleep(0.5)
    #
    # print(f"offspring duplicate check: {len(np.unique(offspring))==len(p1_path)}")
    # print(f"p1: {p1_path}\np2: {p2_path}\nfragments: {fragments}\nshuffled fragments: {shuffled_fragments}\noffspring: {offspring}")
    # Try solutions with both start_points, return the one with less total_distance
    p1_sp_sol, p2_sp_sol = PathSolution(offspring, p1.start_points, info), PathSolution(offspring, p2.start_points, info)
    if p1_sp_sol.total_distance <= p2_sp_sol.total_distance:
        return p1_sp_sol
    else:
        return p2_sp_sol


# Just OX Crossover

class PathCrossover(Crossover):
    def __init__(self, n_parents=2, n_offsprings=2, prob=0.9, **kwargs):
        super().__init__(n_parents=n_parents, n_offsprings=n_offsprings, **kwargs)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), None, dtype=PathSolution)

        for i in range(n_matings):
            # selected_operator_func = ox_crossover
            if random.random() <= self.prob:
                Y[0,i,0], Y[1,i,0] = ox_crossover(X[0, i, 0],X[1, i, 0])
            else:
                Y[0,i,0], Y[1,i,0] = X[0, i, 0],X[1, i, 0]

        return Y


        # crossover_indices = np.random.rand(n_matings) <= self.prob

        # for i in range(n_matings):
        #     if crossover_indices[i]:
        #         Y[0, i, 0], Y[1, i, 0] = ox_crossover(X[0, i, 0], X[1, i, 0])
        #     else:
        #         Y[0, i, 0], Y[1, i, 0] = X[0, i, 0], X[1, i, 0]

        # return Y

'''

class PathCrossover(Crossover):


    def __init__(self, prob=0.9, ox_prob=0.5, n_parents=2, n_offsprings=2, **kwargs):
        super().__init__(n_parents=n_parents, n_offsprings=n_offsprings, **kwargs)

        self.n_offsprings = n_offsprings

        self.prob = prob # 0.9

        self.ox_prob = ox_prob


    def _do(self, problem, X, **kwargs):


        _, n_matings, n_var = X.shape

        Y = np.full((self.n_offsprings, n_matings, n_var), None, dtype=PathSolution)

        for i in range(n_matings):

            # print("Crossover")

            # ox_1, ox_2 = ox_crossover(X[0, i, 0],X[1, i, 0])
            # scx_1, scx_2 = scx_crossover(X[0, i, 0],X[1, i, 0])
            # ox_perf = ((ox_1.percentage_connectivity + ox_2.percentage_connectivity)/2) # + ((ox_1.total_distance + ox_2.total_distance)/2)
            # scx_perf = ((scx_1.percentage_connectivity + scx_2.percentage_connectivity)/2) # + ((scx_1.total_distance + scx_2.total_distance)/2)
            # self.ox_prob = ox_perf / (ox_perf + scx_perf)
            
            if random.random() <= self.prob:
                if random.random() <= self.ox_prob:
                    if self.n_offsprings == 2:
                        Y[0,i,0], Y[1,i,0] = ox_crossover(X[0, i, 0],X[1, i, 0], n_offsprings=self.n_offsprings)
                    elif self.n_offsprings == 4:
                        Y[0,i,0], Y[1,i,0], Y[2,i,0], Y[3,i,0], = ox_crossover(X[0, i, 0],X[1, i, 0], n_offsprings=self.n_offsprings)
                else:
                    if self.n_offsprings == 2:
                        Y[0,i,0], Y[1,i,0] = scx_crossover(X[0, i, 0],X[1, i, 0])
                    elif self.n_offsprings == 4:
                        Y[0,i,0], Y[1,i,0], Y[2,i,0], Y[3,i,0] = scx_crossover(X[0, i, 0],X[1, i, 0])
            else:
                if self.n_offsprings == 2:
                    Y[0,i,0], Y[1,i,0] = X[0, i, 0],X[1, i, 0]
                elif self.n_offsprings == 4:
                    Y[0,i,0], Y[1,i,0], Y[2,i,0], Y[3,i,0] = X[0, i, 0],X[1, i, 0], X[0, i, 0],X[1, i, 0]

        return Y