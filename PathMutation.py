import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.ox import random_sequence
from pymoo.operators.mutation.inversion import inversion_mutation
from scipy.spatial import distance
from typing import List, Dict
import random
from copy import copy, deepcopy

from PathSolution import *
from PathInfo import *
from PathProblem import PathProblem

def random_one_sp_mutation(mut_path, mut_start_points):
    sp_ind = random.randint(1, len(mut_start_points)-1)
    # if sp_ind == len(mut_start_points) - 1:
    #     if mut_start_points[sp_ind] - mut_start_points[sp_ind-1] > 1:
    #         mut_start_points[sp_ind] -= 1
    # else:
    #     if np.random.random() < 0.5:
    #         if mut_start_points[sp_ind] - mut_start_points[sp_ind-1] > 1:
    #             mut_start_points[sp_ind] -= 1
    #     else:
    #         if mut_start_points[sp_ind+1] - mut_start_points[sp_ind] > 1:
    #             mut_start_points[sp_ind] += 1
    prev_sp = mut_start_points[sp_ind-1]
    if sp_ind < len(mut_start_points) - 1:
        next_sp = mut_start_points[sp_ind+1]
    else:
        next_sp = len(mut_path)-1
    new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
    if len(new_sp_choices) != 0:
        mut_start_points[sp_ind] = np.random.choice(new_sp_choices)
    return mut_start_points


def random_n_sp_mutation(mut_path, mut_start_points):
    num_mut = random.randint(0, len(mut_start_points))
    possible_sp_inds = np.arange(1, len(mut_start_points)).tolist()
    for _ in range(num_mut):
        if len(possible_sp_inds) != 0:
            sp_ind = np.random.choice(possible_sp_inds)
            possible_sp_inds.remove(sp_ind)
            prev_sp = mut_start_points[sp_ind-1]
            if sp_ind < len(mut_start_points) - 1:
                next_sp = mut_start_points[sp_ind+1]
            else:
                next_sp = len(mut_path)-1
            new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
            if len(new_sp_choices) != 0:
                mut_start_points[sp_ind] = np.random.choice(new_sp_choices)
    return mut_start_points


def all_sp_mutation(mut_path, mut_start_points):
    sp_indices_perm = np.random.permutation(range(1, len(mut_start_points)))
    # print(f"perm: {sp_indices_perm}")
    for sp_ind in sp_indices_perm:
        prev_sp = mut_start_points[sp_ind-1]
        if sp_ind < len(mut_start_points) - 1:
            next_sp = mut_start_points[sp_ind+1]
        else:
            next_sp = len(mut_path)-1
        new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
        if len(new_sp_choices) != 0:
            mut_start_points[sp_ind] = np.random.choice(new_sp_choices)
    return mut_start_points


def longest_path_sp_mutation(mut_path, mut_start_points):
        path_lens = np.append(np.diff(np.array(mut_start_points)), len(mut_path)-mut_start_points[-1])
        max_len_drone_id = np.argmax(path_lens)
        sp_ind = max_len_drone_id
        prev_sp = mut_start_points[sp_ind-1]
        if sp_ind < len(mut_start_points) - 1:
            next_sp = mut_start_points[sp_ind+1]
        else:
            next_sp = len(mut_path)-1
        new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
        # new_sp_choices = np.arange(start=prev_sp+1, stop=mut_start_points[sp_ind]) # Shortens the sp corresponding to the longest path
        if len(new_sp_choices) != 0:
            mut_start_points[sp_ind] = np.random.choice(new_sp_choices)
        return mut_start_points


class PathMutation(Mutation):

    def __init__(self,
                mutation_info={
                    "swap_last_point":(0, 1),
                    "swap": (0.3, 1), # 0.3 0.5
                    "inversion": (0.4, 1), # 0.4
                    "scramble": (0.3, 1), # 0.3 0.6
                    "insertion": (0, 1),
                    "displacement": (0, 1),
                    # "reverse sequence": (0.3, 1),
                    "block inversion": (0, 1),
                    # "shift": (0.3, 1),
                    "sp_mutation": (0.95, 1), # 1.0 for 4 drones |  0.95 for 8 drones | 0.7 for 12 drones | 0.5 for 16 drones
                    "longest_path_swap": (0.3,1),
                    "longest_path_inversion": (0.4,1),
                    "longest_path_scramble": (0.3,1),
                }
    ) -> None:

        super().__init__()

        self.mutation_info = mutation_info

    def _do(self, problem : PathProblem, X, **kwargs):

        # print("Mutation")

        Y = X.copy()

        for i, y in enumerate(X):
            # print("-->", y)
            sol : PathSolution = y[0]

            start_points = sol.start_points
            path = np.copy(sol.path)
            mut_path = path
            mut_start_points = np.copy(start_points)

            # print("Original Start Points:",start_points)
            #


            # PATH MUTATIONS

            if np.random.random() <= self.mutation_info["swap_last_point"][0] and "Percentage Connectivity" in sol.info.model["F"]:
                    # hovering_cells = [sol.drone_dict[i][-2] for i in range(sol.info.number_of_drones)]
                    hovering_cells = [mut_path[mut_start_points[x]-1] for x in range(1,len(mut_start_points))]
                    hovering_cells.append(mut_path[-1])
                    # hovering_cell_indexes =[np.where(sol.path==cell) for cell in hovering_cells]
                    random_hovering_cell = random.choice(hovering_cells)
                    random_hovering_cell_ind = np.where(mut_path==random_hovering_cell)[0][0]
                    # print("-->",list(np.arange(len(sol.path))))
                    all_cell_indices = list(np.arange(len(mut_path)))
                    # print(f"all_cell_indices: {all_cell_indices}")
                    for hovering_cell in hovering_cells:
                        # print(f"where hovering cell: {np.where(mut_path==hovering_cell)[0][0]}")
                        all_cell_indices.remove(np.where(mut_path==hovering_cell)[0][0])
                    # all_cell_indices.pop(random_hovering_cell_ind)
                    swap_ind = random.choice(all_cell_indices)
                    seq = sorted([random_hovering_cell_ind, swap_ind])
                    # print(f"seq: {seq}")
                    mut_path = np.hstack((
                        mut_path[:seq[0]], np.array([mut_path[seq[1]]]), mut_path[seq[0]+1:seq[1]], np.array([mut_path[seq[0]]]), mut_path[seq[1]+1:]
                    ))
                    # print(f"new path len: {len(mut_path)}")


            if np.random.random() <= self.mutation_info["swap"][0]:
                for _ in range(self.mutation_info["swap"][1]):
                    # Exclude hovering cells
                    hovering_cells = [mut_path[mut_start_points[x]-1] for x in range(1,len(mut_start_points))]
                    hovering_cells.append(mut_path[-1])
                    seq = random_sequence(len(path))
                    # print(f"seq: {seq}")
                    # print(f"swapped cells: {mut_path[seq[0]]} and {mut_path[seq[1]]}")
                    # print(f"pre-swap path: {mut_path}")
                    mut_path = np.hstack((
                        mut_path[:seq[0]], np.array([mut_path[seq[1]]]), mut_path[seq[0]+1:seq[1]], np.array([mut_path[seq[0]]]), mut_path[seq[1]+1:]
                    ))
                    # print(f"post-swap path len: {len(mut_path)}")


            if np.random.random() <= self.mutation_info["inversion"][0]:
                for _ in range(self.mutation_info["inversion"][1]):
                    seq = random_sequence(len(path))
                    mut_path = inversion_mutation(mut_path, seq, inplace=True)


            if np.random.random() <= self.mutation_info["scramble"][0]:
                for _ in range(self.mutation_info["scramble"][1]):
                    seq = random_sequence(len(path))
                    random.shuffle(mut_path[seq[0]:seq[1]])


            if np.random.random() <= self.mutation_info["insertion"][0]:
                for _ in range(self.mutation_info["insertion"][1]):
                    cell = np.random.choice(mut_path)
                    cell_ind = np.where(mut_path == cell)[0][0]
                    mut_path = np.delete(mut_path, cell_ind)
                    new_position = np.random.choice(np.array([i for i in range(len(mut_path) + 1) if i != cell_ind]))
                    mut_path = np.insert(mut_path, new_position, cell)


            if np.random.random() <= self.mutation_info["displacement"][0]:
                for _ in range(self.mutation_info["displacement"][1]):
                    start, end = random_sequence(len(path))
                    seq = mut_path[start:end]
                    indices = np.arange(start, end)
                    mut_path = np.delete(mut_path, indices)
                    new_position = np.random.choice(np.array([i for i in range(len(mut_path) + 1) if i < start or i > start]))
                    mut_path = np.insert(mut_path, new_position, seq)


            if np.random.random() <= self.mutation_info["block inversion"][0]:
                for _ in range(self.mutation_info["block inversion"][1]):
                    start, end = random_sequence(len(path))
                    seq = np.flip(mut_path[start:end])
                    indices = np.arange(start, end)
                    mut_path = np.delete(mut_path, indices)
                    new_position = np.random.choice(np.array([i for i in range(len(mut_path) + 1) if i < start or i > start]))
                    mut_path = np.insert(mut_path, new_position, seq)


            # START POINTS MUTATIONS

            if np.random.random() < self.mutation_info["random_one_sp_mutation"][0]:
                for _ in range(self.mutation_info["random_one_sp_mutation"][1]):
                    # random_one_sp_mutation(mut_path, mut_start_points)
                    sp_ind = random.randint(1, len(mut_start_points)-1)
                    # # if sp_ind == len(mut_start_points) - 1:
                    # #     if mut_start_points[sp_ind] - mut_start_points[sp_ind-1] > 1:
                    # #         mut_start_points[sp_ind] -= 1
                    # # else:
                    # #     if np.random.random() < 0.5:
                    # #         if mut_start_points[sp_ind] - mut_start_points[sp_ind-1] > 1:
                    # #             mut_start_points[sp_ind] -= 1
                    # #     else:
                    # #         if mut_start_points[sp_ind+1] - mut_start_points[sp_ind] > 1:
                    # #             mut_start_points[sp_ind] += 1
                    prev_sp = mut_start_points[sp_ind-1]
                    if sp_ind < len(mut_start_points) - 1:
                        next_sp = mut_start_points[sp_ind+1]
                    else:
                        next_sp = len(mut_path)-1
                    new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
                    if len(new_sp_choices) != 0:
                        mut_start_points[sp_ind] = np.random.choice(new_sp_choices)

            if np.random.random() < self.mutation_info["random_n_sp_mutation"][0]:
                for _ in range(self.mutation_info["random_n_sp_mutation"][1]):
                    mut_start_points = random_n_sp_mutation(mut_path, mut_start_points)
                    # # sp = np.random.choice(mut_start_points[1:]) # To exclude "0"
                    # sp_ind = -1
                    # prev_sp = mut_start_points[sp_ind-1]
                    # if sp_ind < len(mut_start_points) - 1:
                    #     next_sp = mut_start_points[sp_ind+1]
                    # else:
                    #     next_sp = len(mut_path)-1
                    # new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
                    # if len(new_sp_choices) != 0:
                    #     mut_start_points[sp_ind] = np.random.choice(new_sp_choices)                    


            if np.random.random() < self.mutation_info["all_sp_mutation"][0]:
                for _ in range (self.mutation_info["all_sp_mutation"][1]):
                    mut_start_points = all_sp_mutation(mut_path, mut_start_points)
                    # num_mut = random.randint(0, problem.info.number_of_drones)
                    # possible_sp_inds = np.arange(1, floor(len(mut_start_points)/4)).tolist()
                    # for _ in range(num_mut):
                    #     if len(possible_sp_inds) != 0:
                    #         sp_ind = np.random.choice(possible_sp_inds)
                    #         possible_sp_inds.remove(sp_ind)
                    #         prev_sp = mut_start_points[sp_ind-1]
                    #         if sp_ind < len(mut_start_points) - 1:
                    #             next_sp = mut_start_points[sp_ind+1]
                    #         else:
                    #             next_sp = len(mut_path)-1
                    #         new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
                    #         if len(new_sp_choices) != 0:
                    #             mut_start_points[sp_ind] = np.random.choice(new_sp_choices)

            if np.random.random() < self.mutation_info["longest_path_sp_mutation"][0]:
                for _ in range(self.mutation_info["longest_path_sp_mutation"][1]):
                    mut_start_points = longest_path_sp_mutation(mut_path, mut_start_points)
                    # path_lens = np.append(np.diff(np.array(mut_start_points)), sol.info.number_of_cells*sol.info.n_visits-mut_start_points[-1])
                    # max_len_drone_id = np.argmax(path_lens)
                    # sp_ind = max_len_drone_id
                    # prev_sp = mut_start_points[sp_ind-1]
                    # if sp_ind < len(mut_start_points) - 1:
                    #     next_sp = mut_start_points[sp_ind+1]
                    # else:
                    #     next_sp = len(mut_path)-1
                    # # new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
                    # new_sp_choices = np.arange(start=prev_sp+1, stop=mut_start_points[sp_ind]) # Shortens the sp corresponding to the longest path
                    # if len(new_sp_choices) != 0:
                    #     mut_start_points[sp_ind] = np.random.choice(new_sp_choices)

            
            if np.random.random() < self.mutation_info["randomly_selected_sp_mutation"][0]:
                for _ in range(self.mutation_info["randomly_selected_sp_mutation"][1]):

                    rnd = np.random.random()

                    if rnd < 0.25:
                        mut_start_points = random_one_sp_mutation(mut_path, mut_start_points)
                        # sp_ind = random.randint(1,problem.info.number_of_drones-1)
                        # prev_sp = mut_start_points[sp_ind-1]
                        # if sp_ind < len(mut_start_points) - 1:
                        #     next_sp = mut_start_points[sp_ind+1]
                        # else:
                        #     next_sp = len(mut_path)-1
                        # new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
                        # if len(new_sp_choices) != 0:
                        #     mut_start_points[sp_ind] = np.random.choice(new_sp_choices)


                    elif 0.25 <= rnd < 0.50:
                        mut_start_points = random_n_sp_mutation(mut_path, mut_start_points)
                        # num_mut = random.randint(0, problem.info.number_of_drones)
                        # possible_sp_inds = np.arange(1, floor(len(mut_start_points)/4)).tolist()
                        # for _ in range(num_mut):
                        #     if len(possible_sp_inds) != 0:
                        #         sp_ind = np.random.choice(possible_sp_inds)
                        #         possible_sp_inds.remove(sp_ind)
                        #         prev_sp = mut_start_points[sp_ind-1]
                        #         if sp_ind < len(mut_start_points) - 1:
                        #             next_sp = mut_start_points[sp_ind+1]
                        #         else:
                        #             next_sp = len(mut_path)-1
                        #         new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
                        #         if len(new_sp_choices) != 0:
                        #             mut_start_points[sp_ind] = np.random.choice(new_sp_choices)


                    elif 0.50 <= rnd < 0.75:
                        mut_start_points = all_sp_mutation(mut_path, mut_start_points)
                        # path_lens = np.append(np.diff(np.array(mut_start_points)), sol.info.number_of_cells*sol.info.n_visits-mut_start_points[-1])
                        # max_len_drone_id = np.argmax(path_lens)
                        # sp_ind = max_len_drone_id
                        # prev_sp = mut_start_points[sp_ind-1]
                        # if sp_ind < len(mut_start_points) - 1:
                        #     next_sp = mut_start_points[sp_ind+1]
                        # else:
                        #     next_sp = len(mut_path)-1
                        # # new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
                        # new_sp_choices = np.arange(start=prev_sp+1, stop=mut_start_points[sp_ind]) # Shortens the sp corresponding to the longest path
                        # if len(new_sp_choices) != 0:
                        #     mut_start_points[sp_ind] = np.random.choice(new_sp_choices)


                    elif 0.75 <= rnd <= 1.0:
                        mut_start_points = longest_path_sp_mutation(mut_path, mut_start_points)
                        # sp_ind = -1
                        # prev_sp = mut_start_points[sp_ind-1]
                        # if sp_ind < len(mut_start_points) - 1:
                        #     next_sp = mut_start_points[sp_ind+1]
                        # else:
                        #     next_sp = len(mut_path)-1
                        # new_sp_choices = np.arange(start=prev_sp+1, stop=next_sp)
                        # if len(new_sp_choices) != 0:
                        #     mut_start_points[sp_ind] = np.random.choice(new_sp_choices)


                        # print(f"original_start_points: {start_points}, sp: {sp}, new_sp_choices: {new_sp_choices}, new_sp: {sp_new}, new_start_points: {mut_start_points}")


            
            Y[i][0] = PathSolution(mut_path, mut_start_points, problem.info)


        return Y