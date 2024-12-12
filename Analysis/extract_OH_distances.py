import numpy as np
import sys

sys.path.append("../")
import os
from Simulation.helper_functions import SOLVENT_DICT as solvent_dict
from analysis import get_distance_by_name
import mdtraj
from typing import DefaultDict
import pickle


pair_dict = {0: ("O1", "H8"), 1: ("O1", "H10"), 2: ("O1", "H12")}

idx = 0
folder = "/cluster/work/igc/kpaul/projects/multisolvent_pub/Simulation/simulation/intra_molecular_hbond2/gbneck_simulations/"
files = os.listdir(folder)

results = DefaultDict(dict)
for file in files:
    if ".h5" in file:
        path = f"{folder}{file}"
        idx = int(file.split("_")[5])
        dielectric = float(file.split("_")[8])
        print(f"Working on {idx} with dielectric {dielectric}")
        try:
            traj = mdtraj.load(path)
            dis = get_distance_by_name(traj, pair_dict[idx][0], pair_dict[idx][1])
            results[idx][dielectric] = dis
        except Exception as e:
            print(e)
            continue

with open("I1I2I3_distance.pkl", "wb") as f:
    pickle.dump(results, f)
