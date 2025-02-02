from statistics import median, median_low, median_high
from typing import Any
import pickle
from PathAlgorithm import *
from PathOutput import *
from pymoo.operators.crossover.nox import NoCrossover
from pymoo.operators.mutation.nom import NoMutation
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.optimize import minimize
from Results import save_paths_and_anims_from_scenario, animate_extreme_point_paths, save_best_solutions
# from PathTermination import PathTermination
from PathAnimation import PathAnimation
from Time import *
import os
import shutil
from math import inf
# from google.colab import drive
# drive.mount('/content/drive')

from PathOptimizationModel import *
from PathInput import *
from WeightedSum import PathCallback
# from main import *
from FilePaths import *
from PathFileManagement import save_as_pickle, load_pickle

from pymoo.termination.default import DefaultTermination, DefaultMultiObjectiveTermination
from pymoo.core.termination import Termination, NoTermination
# from PathTermination import PathDefaultMultiObjectiveTermination
from pymoo.termination import get_termination
from PathTermination import WeightedSumTermination

from GoogleDriveUpload import authenticate, upload_file, PARENT_FOLDER_ID_DICT

class PathUnitTest(object):

    def __init__(self, scenario) -> None:

        self.model = model
        self.algorithm = self.model["Alg"] # From PathInput

        self.info = [PathInfo(scenario)] if not isinstance(scenario, list) else list(map(lambda x: PathInfo(x), scenario))

    def __call__(self, save_results=True, animation=False, copy_to_drive=True, *args: Any, **kwds: Any) -> Any:

        for info in self.info:
            scenario = str(info)
            print(f"Scenario: {str(info)}")
            res, F, F_abs, X, R = self.run_optimization(info)
            if X is not None:
                if save_results:
                    # save_as_pickle(f"{res_filepath}{str(info)}-Res.pkl", X)
                    # Save PathSolutions
                    save_as_pickle(f"{solutions_filepath}{scenario}-SolutionObjects.pkl", X)
                    upload_file(f"{solutions_filepath}{scenario}-SolutionObjects.pkl", PARENT_FOLDER_ID_DICT["Solutions"]) if copy_to_drive else None
                    # Save Objective Values
                    F.to_pickle(f"{objective_values_filepath}{scenario}-ObjectiveValues.pkl")
                    upload_file(f"{objective_values_filepath}{scenario}-ObjectiveValues.pkl", PARENT_FOLDER_ID_DICT["Objectives"]) if copy_to_drive else None
                    F_abs.to_pickle(f"{objective_values_filepath}{scenario}-ObjectiveValuesAbs.pkl")
                    upload_file(f"{objective_values_filepath}{scenario}-ObjectiveValuesAbs.pkl", PARENT_FOLDER_ID_DICT["Objectives"]) if copy_to_drive else None
                    # Save Runtimes
                    save_as_pickle(f"{runtimes_filepath}{scenario}-Runtime.pkl", R)
                    upload_file(f"{runtimes_filepath}{scenario}-Runtime.pkl", PARENT_FOLDER_ID_DICT["Runtimes"]) if copy_to_drive else None
                    # save_paths_and_anims_from_scenario(str(info))
                    # save_best_solutions(scenario, copy_to_drive)
                
                if animation:
                    animate_extreme_point_paths(info)

                print(f"Scenario: {str(info)} COMPLETED !!!")
            else:
                print(f"Scenario: {str(info)} NO SOLUTION FOUND !!!")

            
    def run_optimization(self, info):

        problem = PathProblem(info)
        algorithm = PathAlgorithm(self.algorithm)()
        termination = ("n_gen", n_gen)
        # termination = NoTermination()
        output = PathOutput(problem)

        res, F, F_abs, X, R = None, None, None, None, None

        t = time.time()
        t_start = time.time()

        res = minimize(problem=PathProblem(info),
                        algorithm=algorithm,
                        termination=termination,
                        save_history=True,
                        seed=1,
                        output=output,
                        verbose=True,
                        # callback=PathCallback(model)
                        )
        
        t_end = time.time()
        t_elapsed_seconds = t_end - t_start

        if res.X is not None:
            # print(res.X)
            X = res.X.flatten() # FLATTEN NEW !
            F = pd.DataFrame(res.F, columns=model['F'])
            F_abs= abs(F)
            R = t_elapsed_seconds
            # If certain attributes are missing from the solution objects, add them here
            sample_sol = X[0][0] if isinstance(X[0], np.ndarray) else X[0]
            # Add TBV and Disconnecivity attributes to the solution objects if they are not already calculated
            for row in X:
                if isinstance(row, np.ndarray):
                    sol = row[0]
                else:
                    sol = row
                if not sample_sol.calculate_tbv:
                    sol.get_visit_times()
                    sol.get_tbv()
                    sol.get_mean_tbv()
                if not sample_sol.calculate_disconnectivity:
                    sol.do_disconnectivity_calculations()


        return res, F, F_abs, X, R