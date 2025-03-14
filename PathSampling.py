from pymoo.core.sampling import Sampling
import numpy as np
from math import floor
import random
from PathSolution import *
# from PathProblem import *

'''
TODO

Redo the path sampling i.e. start at -1. cell return to -1.cell (start and end the tours at BS)
Visit each cell (0.-63.) at least "n_visits" times at most "max_visits" times

UPDATES tick
DONE not yet

'''

class AdaptiveRelaySampling(Sampling):

    def __init__(self) -> None:
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):


        # X = np.full((n_samples, problem.n_var), 0, dtype=int)
        X = np.full((n_samples, problem.n_var), None, dtype=PathSolution)

        cells = np.arange(problem.info.number_of_cells)

        for i in range(n_samples):
            X[i] = np.random.choice(cells, size=problem.info.number_of_drones-1, replace=False)

        return X


class PathSampling(Sampling):

    def __init__(self) -> None:
        super().__init__()

    def _do(self, problem, n_samples, **kwargs):


        # X = np.full((n_samples, problem.n_var), 0, dtype=int)
        X = np.full((n_samples, 1), None, dtype=PathSolution)

        for i in range(n_samples):
            # Random path
            path = np.random.permutation(problem.info.n_visits * problem.info.number_of_cells)%problem.info.number_of_cells
            # print(f"path: {min(path), max(path)}")
            # Random start points
            start_points = sorted(random.sample([i for i in range(1, len(path))], problem.info.number_of_drones - 1))
            start_points.insert(0, 0)
            # print(f"Path: {path}\nStart Points: {start_points}")
            # print("----------------------------------------------------------------------------------------------------")
            # print(f"Sample {i}")
            # print("----------------------------------------------------------------------------------------------------")
            # print(f"Path: {path}\nStart Points: {start_points}")
            # print(f"Path: {path}")
            # print("Start Points: ", start_points)

            X[i, :] = PathSolution(path, start_points, problem.info, calculate_pathplan=False, calculate_tbv=False, calculate_connectivity=False, calculate_disconnectivity=False)

        return X

'''
            start_points = []
            cpd = problem.info.Nc * problem.info.n_visits // problem.info.Nd  # Cells per Drone
            cpd_bias = 2  # Cells per Drone Bias
            for d in range(problem.info.Nd - 1):
                if d == 0:
                    start_points.append(random.randrange(cpd - cpd_bias, cpd + cpd_bias))  # Generate random index
                else:
                    start_points.append(start_points[-1] + random.randrange(cpd - cpd_bias, cpd + cpd_bias))
            start_points.insert(0, 0)

'''

'''
start_points = sorted(random.sample([i for i in range(1,len(path))], problem.info.Nd-1))
start_points.insert(0, 0)
'''
