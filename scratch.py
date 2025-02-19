from FilePaths import *
import os

dirs = [solutions_filepath, objective_values_filepath, runtimes_filepath]

for dir in dirs:
    filenames = os.listdir(dir)
    # Replace minv with nvisits
    for filename in filenames:
        split_filename = filename.split("_")
        # print(split_filename)
        split_filename = split_filename[:2] + ["MTSP"] + split_filename[2:]
        new_filename = ""
        for i in range(len(split_filename)):
            new_filename += split_filename[i] + "_"
        new_filename = new_filename[:-1]
        # print(new_filename)
        # new_filename = filename.replace("minv", "nvisits")
        os.rename(f"{dir}{filename}", f"{dir}{new_filename}")
        # print(f"Renamed {filename} to {new_filename}")