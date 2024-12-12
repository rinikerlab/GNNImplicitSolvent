import numpy as np
import sys

sys.path.append("../")
import os
from Simulation.helper_functions import SOLVENT_DICT as solvent_dict
from analysis import get_dihedrals_by_name
import mdtraj
from typing import DefaultDict
import pickle

idx = 0
folder = "/cluster/work/igc/kpaul/projects/multisolvent_pub/Simulation/simulation/dimethoxy/gbneck_simulations/"
files = os.listdir(folder)

results = DefaultDict(dict)
for file in files:
    if ".h5" in file:
        path = f"{folder}{file}"
        idx = int(file.split("_")[3])
        dielectric = float(file.split("_")[6])
        print(f"Working on {idx} with dielectric {dielectric}")
        try:
            traj = mdtraj.load(path)
            dis = gnn_dis = get_dihedrals_by_name(traj, *("O1 C2 C3 O2".split()))
            results[idx][dielectric] = dis
        except Exception as e:
            print(e)
            continue

with open("dimethoxy_dihedrals.pkl", "wb") as f:
    pickle.dump(results, f)
