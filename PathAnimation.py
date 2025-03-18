import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from Time import get_real_paths, get_real_connectivity_matrix

from PathSolution import *

class PathAnimation:

    def __init__(self, sol:PathSolution, fig, ax):
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'mediumseagreen'] * 3  # Colors for 24 drones

        self.sol = sol
        self.fig = fig
        self.ax = ax
        self.paths = get_real_paths(self.sol)  # np.array([x_matrix, y_matrix])
        self.real_time_x_matrix, self.real_time_y_matrix = self.paths
        self.real_time_connectivity_matrix = get_real_connectivity_matrix(
            self.real_time_x_matrix, self.real_time_y_matrix, self.sol)

        # Set the title for the subplot
        # self.ax.set_title(str(sol.info))

        # Set ticks and labels
        x_ticks_values = [i for i in range(-self.sol.info.cell_side_length, 
                                           (self.sol.info.grid_size + 1) * self.sol.info.cell_side_length, 
                                           self.sol.info.cell_side_length)]
        y_tick_values = x_ticks_values.copy()
        x_ticks_labels = [i for i in range(-1, self.sol.info.grid_size + 1)]
        y_tick_labels = x_ticks_labels.copy()

        if ax:
            self.ax.set_xticks(x_ticks_values)
            self.ax.set_xticklabels(x_ticks_labels)
            self.ax.set_yticks(y_tick_values)
            self.ax.set_yticklabels(y_tick_labels)
            self.ax.grid(linestyle='--')
        else:
            plt.set_xticks(x_ticks_values)
            plt.set_xticklabels(x_ticks_labels)
            plt.set_yticks(y_tick_values)
            plt.set_yticklabels(y_tick_labels)
            plt.grid(linestyle='--')


    def initialize_figure(self):
        """Initialize the plot elements for animation."""
        self.drone_animations = self.ax.scatter([], [], marker="o")
        self.drone_path_lines = [self.ax.plot([], [], color=self.colors[_], marker="", linewidth=0.5)[0] 
                                 for _ in range(self.sol.info.number_of_nodes)]
        self.connectivity_lines = [self.ax.plot([], [], color='k', marker="", linewidth=3)[0] 
                                   for _ in range(self.sol.info.number_of_nodes)]

        self.drone_animations.set_offsets(np.empty((0, 2)))

        for connectivity_line in self.connectivity_lines:
            connectivity_line.set_data([], [])

        for drone_no, drone_path_line in enumerate(self.drone_path_lines):
            drone_x_path, drone_y_path = self.real_time_x_matrix[drone_no], self.real_time_y_matrix[drone_no]
            drone_path_line.set_data([drone_x_path], [drone_y_path])

        return self.drone_animations, *self.connectivity_lines

    def update(self, frame):
        """Update function for each frame of the animation."""
        # Update Drone Paths
        x_path = self.paths[0][:, frame]
        y_path = self.paths[1][:, frame]
        data = np.stack((x_path, y_path), axis=-1)
        self.drone_animations.set_offsets(data)

        # Update Connectivity Lines
        for node_no, node_connectivity_lines in enumerate(self.connectivity_lines):
            connectivity_lines_xdata = []
            connectivity_lines_ydata = []
            for node_no_2 in range(node_no + 1, self.sol.info.number_of_nodes):
                connectivity_array = self.real_time_connectivity_matrix[frame, node_no, :]
                if connectivity_array[node_no_2]:
                    connectivity_lines_xdata.extend([self.paths[0][node_no, frame], self.paths[0][node_no_2, frame]])
                    connectivity_lines_ydata.extend([self.paths[1][node_no, frame], self.paths[1][node_no_2, frame]])
            node_connectivity_lines.set_data(connectivity_lines_xdata, connectivity_lines_ydata)

        return self.drone_animations, *self.connectivity_lines

    def __call__(self):
        """Create and return the animation."""
        anim = FuncAnimation(self.fig, self.update, frames=self.paths[0].shape[1],
                             init_func=self.initialize_figure, blit=True, interval=50)
        plt.show()
        return anim


'''from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import subprocess
from time import sleep
import pickle

from PathSolution import PathSolution
from Time import get_real_paths, get_real_connectivity_matrix
from FileManagement import save_as_pickle, load_pickle

class PathAnimation:

    def __init__(self, sol:PathSolution):
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'mediumseagreen'] * 3  # 24 drones

        self.sol = sol
        # self.info = sol.info
        self.paths = get_real_paths(self.sol)  # np.array([x_matrix, y_matrix])
        self.real_time_x_matrix, self.real_time_y_matrix = self.paths
        self.real_time_connectivity_matrix = get_real_connectivity_matrix(self.real_time_x_matrix, self.real_time_y_matrix, self.sol)

        fig, axis = plt.subplots()

        # Set figure to full-screen
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()

        self.fig, self.axis = fig, axis
        axis.set_title(str(sol.info))

        # Set ticks and labels
        x_ticks_values = [i for i in range(-self.sol.info.cell_side_length, (self.sol.info.grid_size + 1) * self.sol.info.cell_side_length, self.sol.info.cell_side_length)]
        y_tick_values = x_ticks_values.copy()
        x_ticks_labels = [i for i in range(-1, self.sol.info.grid_size + 1)]
        y_tick_labels = x_ticks_labels.copy()

        plt.xticks(x_ticks_values, x_ticks_labels)
        plt.yticks(y_tick_values, y_tick_labels)
        plt.grid(linestyle='--')


    def initialize_figure(self):
        self.drone_animations = self.axis.scatter([], [], marker="o")
        self.drone_path_lines = [self.axis.plot([], [], color=self.colors[_], marker="", linewidth=0.5)[0] for _ in range(self.sol.info.number_of_nodes)]
        # print("-->",self.drone_path_lines)
        self.connectivity_lines = [self.axis.plot([], [], color='k', marker="", linewidth=3)[0] for _ in range(self.sol.info.number_of_nodes)]

        self.drone_animations.set_offsets(np.empty((0, 2)))

        for connectivity_line in self.connectivity_lines:
            connectivity_line.set_data([], [])

        for drone_no, drone_path_line in enumerate(self.drone_path_lines):
            drone_x_path, drone_y_path = self.real_time_x_matrix[drone_no], self.real_time_y_matrix[drone_no]
            drone_path_line.set_data([drone_x_path], [drone_y_path])

        return self.drone_animations, *self.connectivity_lines

    def update(self, frame):

        # print("IN UPDATE !!!")

        # Update Drone Paths
        x_path = self.paths[0][:, frame]
        y_path = self.paths[1][:, frame]
        data = np.stack((x_path, y_path), axis=-1)
        # print("-->", data)
        self.drone_animations.set_offsets(data)

        # Update Connectivity Lines (assuming some logic for connectivity)
        # For demonstration, just showing connectivity to origin (0,0)
        for node_no, node_connectivity_lines in enumerate(self.connectivity_lines):
            connectivity_lines_xdata = []
            connectivity_lines_ydata = []
            for node_no_2 in range(node_no+1, self.sol.info.number_of_nodes):
                connectivity_array = self.real_time_connectivity_matrix[frame,node_no,:]
                if connectivity_array[node_no_2]:
                    connectivity_lines_xdata.append( [self.paths[0][node_no, frame], self.paths[0][node_no_2, frame]] )
                    connectivity_lines_ydata.append( [self.paths[1][node_no, frame], self.paths[1][node_no_2, frame]] )
            # line_xdata = [0, xdata[node_no]]
            # line_ydata = [0, ydata[node_no]]
            node_connectivity_lines.set_data(connectivity_lines_xdata, connectivity_lines_ydata)

        # sleep(0.40)

        return self.drone_animations, *self.connectivity_lines

    def __call__(self):
        anim = FuncAnimation(self.fig, self.update, frames=self.paths[0].shape[1],
                             init_func=self.initialize_figure, blit=True, interval=50)
        return plt
        
        # plt.show()

# Load your sample solution and paths
# sample_sol = np.load("Results/Solutions/SOO_GA_g_8_a_50_n_4_v_2.5_r_2_minv_1_maxv_5_Nt_1_tarPos_12_ptdet_0.99_pfdet_0.01_detTh_0.9_maxIso_0_SolutionObjects.npy", allow_pickle=True)[0]
# anim = PathAnimation(sample_sol)
# anim()
'''

# from PathFileManagement import load_pickle
# sol = load_pickle("Results/Solutions/MOO_NSGA2_MTSP_TCDT_g_8_a_50_n_8_v_2.5_r_2_nvisits_1-SolutionObjects.pkl")[0]
# print(sol.drone_dict)
# new_sol = produce_n_tour_sol(sol, 2)
# fig, ax = plt.subplots(figsize=(10, 10))
# new_anim = PathAnimation(new_sol, fig, ax)
# new_anim()