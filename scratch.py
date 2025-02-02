from FilePaths import *
import os

dirs = [solutions_filepath, objective_values_filepath, runtimes_filepath]

for dir in dirs:
    filenames = os.listdir(dir)
    # Replace minv with nvisits
    for filename in filenames:
        new_filename = filename.replace("minv", "nvisits")
        os.rename(f"{dir}{filename}", f"{dir}{new_filename}")
        # print(f"Renamed {filename} to {new_filename}")