from pymoo.core.repair import Repair
import numpy as np
from copy import deepcopy

from PathSolution import *
from PathInfo import *

from Time import get_real_paths
from Distance import get_city

import numpy as np
from pymoo.core.repair import Repair

from PathSolution import *
from PathProblem import *
from Distance import *
from PathAnimation import *


# from functools import lru_cache
# from concurrent.futures import ThreadPoolExecutor, as_completed

def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0  # Returns 0 if x is exactly 0

class PathRepair(Repair):

    def _do(self, problem, X, **kwargs):

        calculate_connectivity = True
        calculate_disconnectivity = False
        calculate_tbv = True

        for obj in problem.model["F"]:
            if "Disconnected Time" in obj:
                calculate_disconnectivity = True
            if "TBV" in obj:
                calculate_tbv = True

        # calculate_disconnectivity = "Max Disconnected Time" in problem.model["F"] or "Mean Disconnected Time" in problem.model["F"]
        # calculate_tbv = "Max Mean TBV" in problem.model["F"] or "Max Mean TBV as Constraint" in problem.model["G"]

        # calculate_connectivity = True
        # calculate_connectivity = "Percentage Connectivity" in problem.model["F"] or "Mission Time and Percentage Connectivity Weighted Sum" in problem.model["F"]
        # calculate_disconnectivity = "Mean Disconnected Time" in problem.model["F"] or "Max Disconnected Time" in problem.model["F"]
        # calculate_tbv = "Max Mean TBV" in problem.model["F"] or "Max Mean TBV as Constraint" in problem.model["G"] or "Mission Time and Percentage Connectivity and Max Mean TBV Weighted Sum" in problem.model["F"]

        # calculate_connectivity, calculate_disconnectivity, calculate_tbv = True, True, True
        # print(calculate_connectivity,calculate_disconnectivity)
        # print("Repair Handling")
        
        for k in range(len(X)):
            sol : PathSolution = X[k, 0]

            new_path = self.interpolate_path(sol)
            # print(f"Old Path: {sol.path}")
            # print(f"New Path: {new_path}")
            # print(f"New Path Length: {len(new_path)}")

            # old_anim = PathAnimation(sol)
            # new_anim = PathAnimation(PathSolution(new_path, np.copy(sol.start_points), sol.info))
            # old_anim()
            # new_anim()

            # print(f"new path: {new_path}")

            X[k, 0] = PathSolution(new_path, np.copy(sol.start_points), sol.info, calculate_pathplan=True, calculate_tbv=calculate_tbv, calculate_connectivity=calculate_connectivity, calculate_disconnectivity=calculate_disconnectivity)

            # print(f"New Path Length: {len(new_path)}")

            # print(f"Frequency:\n{pd.Series(new_path).value_counts()}")

            #X[k, 0] = PathSolution(new_path, np.copy(sol.start_points), problem.info, calculate_conn=True, calculate_dist=True)

        return X
    
    def interpolate_path(self, sol:PathSolution):

        # new_path = []

        # for i in range(sol.info.number_of_drones):
        #     subpath = list(sol.drone_dict[i])
        #     new_subpath = []
        #     city_prev = subpath[0]
        #     subpath.pop(0)
        #     for _ in range(len(subpath)-1):
        #         city = subpath[0]
        #         subpath.pop(0)
        #         if city not in new_path:
        #             # Interpolate cities
        #             mid_cities = self.interpolate_between_cities(sol, city_prev, city)
        #             for city_mid in mid_cities:
        #                 if city_mid not in new_path and city_mid != -1:
        #                     new_path.append(city_mid)
        #         # print(f"city prev: {city_prev}, city: {city}, new path: {new_path}")
        #         city_prev = city
        #     # print(f"drone {i} interpolated path: {new_subpath} drone dict: {sol.drone_dict[i]}")
            
        # return new_path


        copy_path = list(sol.path).copy()

        new_path = []

        city_prev = copy_path[0]

        copy_path.pop(0)

        # print(f"min visits: {sol.info.n_visits}")

        while(len(new_path) < sol.info.number_of_cells * sol.info.n_visits):

            # print(len(new_path) < sol.info.number_of_cells * sol.info.n_visits)
            city = copy_path[0]

            copy_path.pop(0)

            # if city not in new_path:
            # print(f"new path: {new_path}, city: {city}, count: {new_path.count(city)}")
            # print("-->", new_path.count(city))
            if new_path.count(city) < sol.info.n_visits:
                # Interpolate cities
                mid_cities = self.interpolate_between_cities(sol, city_prev, city)
                for city_mid in mid_cities:
                    # if city_mid not in new_path:
                    if new_path.count(city_mid) < sol.info.n_visits:
                        new_path.append(city_mid)

            # print(f"city prev: {city_prev}, city: {city}, new path: {new_path}")
            # print(f"New Path Length: {len(new_path)}")

            city_prev = city

            # if len(new_path) == sol.info.number_of_cells * sol.info.n_visits:
            # if len(copy_path) == 1:
                # break

            # print(len(new_path))

        return new_path



    def interpolate_between_cities(self, sol:PathSolution, city_prev, city):

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
            # print(f"Iteration {_+1} coords: {coords_temp}, corresponding city: {mid_city}")
            interpolated_path.append(mid_city)

        # interpolated_path.pop(-1)

        # print(f"city prev: {city_prev}, city: {city}, mid cities: {interpolated_path}")
        
        return interpolated_path



'''
# def sign(x):
#     return (x > 0) - (x < 0)  # Returns 1, -1, or 0

# class PathRepair(Repair):
    
#     def __init__(self, max_workers=4):
#         super().__init__()
#         self.max_workers = max_workers

#     def _do(self, problem, X, **kwargs):
#         # Parallelizing the repair operation
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = [executor.submit(self.repair_solution, sol) for sol in X[:, 0]]
#             for i, future in enumerate(as_completed(futures)):
#                 X[i, 0] = future.result()
#         return X
    
#     def repair_solution(self, sol):
#         # Interpolates and repairs a single solution
#         new_path = self.interpolate_path(sol)
#         return PathSolution(new_path, sol.start_points, sol.info, calculate_pathplan=True, calculate_connectivity=True, calculate_disconnectivity=True)

#     def interpolate_path(self, sol: PathSolution):
#         new_path_set = set()
#         new_path = []
#         city_prev = sol.path[0]

#         for city in sol.path[1:]:
#             if city not in new_path_set:
#                 mid_cities = self.interpolate_between_cities(sol, city_prev, city)
#                 for city_mid in mid_cities:
#                     if city_mid not in new_path_set:
#                         new_path.append(city_mid)
#                         new_path_set.add(city_mid)
#             city_prev = city
#             if len(new_path) >= sol.info.number_of_cells:
#                 break
#         return new_path

#     @lru_cache(maxsize=None)  # Cache results for interpolations
#     def interpolate_between_cities(self, sol: PathSolution, city_prev, city):
#         interpolated_path = [city_prev]
#         info = sol.info
#         coords_prev = sol.get_coords(city_prev)
#         coords = sol.get_coords(city)
#         coords_delta = coords - coords_prev
#         axis_inc = np.sign(coords_delta)

#         num_mid_cities = int(np.max(np.abs(coords_delta)) / info.cell_side_length)
#         coords_temp = coords_prev.copy()

#         for _ in range(num_mid_cities):
#             coords_temp += info.cell_side_length * axis_inc * (coords_temp != coords)
#             mid_city = sol.get_city(tuple(coords_temp))  # Ensure the coordinates are hashable for caching
#             interpolated_path.append(mid_city)
        
#         return tuple(interpolated_path)  # Return a tuple so it can be cached


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0  # Returns 0 if x is exactly 0

class PathRepair(Repair):

    def _do(self, problem, X, **kwargs):
        
        for k in range(len(X)):
            sol : PathSolution = X[k, 0]
            repaired_path = self.interpolate_path(sol)
            calculate_connectivity = bool("Percentage Connectivity" in sol.info.model["F"])
            calculate_disconnectivity = bool("Percentage Disconnectivity" in sol.info.model["F"])
            repaired_solution = PathSolution(repaired_path, np.copy(sol.start_points), sol.info, calculate_pathplan=True, calculate_connectivity=calculate_connectivity, calculate_disconnectivity=calculate_disconnectivity)
            X[k, 0] = repaired_solution

            speed_violations = calculate_speed_violations(repaired_solution)

            if speed_violations:
                print(f"Old Path: {sol.path}\nNew Path: {repaired_path}\nSpeed Violations: {speed_violations}")


        return X
    
    def interpolate_path(self, sol:PathSolution):

        copy_path = list(sol.path).copy()

        new_path = []

        city_prev = copy_path[0]

        copy_path.pop(0)

        while(True):
            
            city = copy_path[0]
            copy_path.pop(0)

            if city not in new_path:
                # Interpolate cities
                mid_cities = self.interpolate_between_cities(sol, city_prev, city)
                for city_mid in mid_cities:
                    if city_mid not in new_path:
                        new_path.append(city_mid)

            city_prev = city

            if len(new_path) == sol.info.number_of_cells:
                break

        return new_path



    def interpolate_between_cities(self, sol:PathSolution, city_prev, city):

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
'''