from Sensing import merge_maps, sensing_and_info_sharing
from PathSolution import *
from PathFileManagement import load_pickle
from FilePaths import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from math import log10

# model_name_model_object_dict = {
#     "MTSP": MTSP,
#     "TC_MOO": TC_MOO_NSGA2,
#     "TC_SOO": TC_SOO_GA,
#     "TCDT_MOO": TCDT_MOO_NSGA2,
#     "TCDT_SOO": TCDT_SOO_GA
# }

def initialize_figure(_title=None, _suptitle=None, xlabel=None, ylabel=None, xticks=None, yticks=None, grid_on=True):
    fig,ax = plt.subplots()
    if _title is not None: ax.set_title(_title)
    if _suptitle is not None: fig.suptitle(_suptitle)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if xticks is not None: ax.set_xticks(xticks)
    if yticks is not None: ax.set_yticks(yticks)
    if grid_on: ax.grid()

    return fig, ax


def plot_time_metrics(models=[TCDT_MOO_NSGA2, TC_MOO_NSGA2], merging_strategy="ondrone", n_runs=50, p0=0.5, B_list=[0.9, 0.95], p_list=[0.8, 0.9], comm_range_list=["sqrt(8)", 2], number_of_drones_list=[4, 8, 12, 16], n_targets_list=[3, 4, 5], show=True, save=False):

    # Plot for every r, B, p, n_tar combination.
    # Put all 'best objective paths' on the same plots.
    # You should have len(r)*len(B)*len(p)*len(n_tar) plots. Default: 2*2*2*3 = 24 plots

    info = PathInfo()
    for comm_range in comm_range_list:
        info.comm_cell_range = comm_range
        for B in B_list:
            for p in p_list:
                q = 1-p
                m = ceil( log10((p0*(1-B))/(B*(1-p0))) / log10(q/p) )
                for n_targets in n_targets_list:
                    parameters = f"B: {B}, p: {p}, q: {q}, T: {n_targets},  $r_c$: {comm_range}" + "$n_{visits}$: " + str(m)
                    # Initialize the figures and y_data
                    detection_time_fig, detection_time_ax = initialize_figure(_title=parameters, _suptitle="Detection Time", xlabel="Number of Drones", ylabel="Detection Time (s)", xticks=number_of_drones_list, yticks=None, grid_on=True)
                    inform_time_fig, inform_time_ax = initialize_figure(_title=parameters, _suptitle="Inform (Mission) Time", xlabel="Number of Drones", ylabel="Inform (Mission) Time(s)", xticks=number_of_drones_list, yticks=None, grid_on=True)
                    time_at_least_one_drone_knows_all_targets_fig, time_at_least_one_drone_knows_all_targets_ax = initialize_figure(_title=parameters, _suptitle="Time at Least One Drone Knows All Targets (s)", xlabel="Number of Drones", ylabel="Time at Least One Drone Knows All Targets (s)", xticks=number_of_drones_list, yticks=None, grid_on=True)
                    success_rate_fig, success_rate_ax = initialize_figure(_title=parameters, _suptitle="Mission Success Rate", xlabel="Number of Drones", ylabel="Mission Success Rate (%)", xticks=number_of_drones_list, yticks=None, grid_on=True)
                    figs = [detection_time_fig, inform_time_fig, time_at_least_one_drone_knows_all_targets_fig, success_rate_fig]
                    axes = [detection_time_ax, inform_time_ax, time_at_least_one_drone_knows_all_targets_ax, success_rate_ax]
                    #Mission Time - Percentage Connectivity - Weighted Sum
                    for model in models:
                        info.model = model
                        if model["Type"]=="WS": # WS
                            label = f"{model['Exp']}-WS Best Path"
                        elif model["Type"]=="SOO":
                            label = f"{model['Exp']} Best Path"
                        for objective_name in model["F"]:
                            if model["Type"]=="MOO":
                                label = f"{model['Exp']}-{model['Type']} Best {objective_name} Path"
                            y_detection_time = np.zeros(len(number_of_drones_list))
                            y_inform_time = y_detection_time.copy()
                            y_time_at_least_one_drone_knows_all_targets = y_detection_time.copy()
                            y_successful_runs = y_detection_time.copy()
                            for i,number_of_drones in enumerate(number_of_drones_list):
                                info.number_of_drones = number_of_drones
                                scenario = str(info)
                                X = load_pickle(f"{solutions_filepath}{scenario}-SolutionObjects.pkl")
                                print(scenario)
                                F = load_pickle(f"{objective_values_filepath}{scenario}-ObjectiveValues.pkl")
                                # F = np.load(f"{objective_values_filepath}{scenario}-ObjectiveValues.pkl", allow_pickle=True) if model["Type"]=="SOO" else load_pickle(f"{objective_values_filepath}{scenario}-ObjectiveValues.pkl")
                                opt_sol = X[F[objective_name].idxmin()] if not len(model["F"]) > 1 and model["Type"]=="SOO" else X[0]
                                for run in range(n_runs):
                                    print(f"Run # {run}")
                                    target_locations = np.random.choice(range(1, 64), n_targets, replace=False)
                                    time_metrics = sensing_and_info_sharing(sol=opt_sol, merging_strategy="ondrone", target_locations=target_locations, B=B, p=p, q=q)
                                    y_detection_time[i] += time_metrics["detection time"] if time_metrics["detection time"] != np.inf else  y_detection_time[i]
                                    y_inform_time[i] += time_metrics["inform time"] if time_metrics["inform time"] != np.inf else  y_inform_time[i]
                                    y_time_at_least_one_drone_knows_all_targets[i] += time_metrics["time at least one drone knows all targets"] if time_metrics["time at least one drone knows all targets"] != np.inf else  y_time_at_least_one_drone_knows_all_targets[i]
                                    y_successful_runs[i] += 1 if time_metrics["detection time"] != np.inf else  y_successful_runs[i]
                            detection_time_ax.plot(number_of_drones_list, y_detection_time/n_runs, label=label, linewidth=0.5)
                            inform_time_ax.plot(number_of_drones_list, y_inform_time/n_runs, label=label, linewidth=0.5)
                            time_at_least_one_drone_knows_all_targets_ax.plot(number_of_drones_list, y_time_at_least_one_drone_knows_all_targets/n_runs, label=label, linewidth=0.5)
                            success_rate_ax.plot(number_of_drones_list, y_successful_runs/n_runs, label=label, linewidth=0.5)
                    
                    for ax in axes:
                        ax.legend()

                    if show:
                        plt.show()

                    if save:
                        for fig in figs:
                            fig.savefig(f"Figures/Sensing/{parameters}_{fig.suptitle().get_text()}.png")

plot_time_metrics(models=[MTSP, TCDT_WS], merging_strategy="ondrone", n_runs=50, p0=0.5, B_list=[0.9, 0.95], p_list=[0.8, 0.9], comm_range_list=["sqrt(8)", 2], number_of_drones_list=[4, 8, 12], n_targets_list=[3, 4, 5], show=True)