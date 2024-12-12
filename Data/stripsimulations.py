import matplotlib.pyplot as plt
import mdtraj
import os
import yaml

import argparse

parser = argparse.ArgumentParser(description='Run ML MM')
parser.add_argument('-f','--folder',type=str,help='ID of pdb, if available amber files will be used')
args = parser.parse_args()

folder = args.folder

def load_traj(traj_file):
    print("working on: ",traj_file)
    assert os.path.isfile(traj_file)
    file_core = traj_file.split('.')[0]
    if os.path.isfile(file_core + '_stripped.h5'):
        print(file_core + '_stripped.h5 exists')
        traj = mdtraj.load(file_core + '_stripped.h5')
    else:
        print("loading traj")
        try:
            traj = mdtraj.load(traj_file)
            traj = traj.atom_slice(traj.top.select('resid %i' % (traj.top.n_residues - 1)))
            traj.save(file_core + '_stripped.h5')
        except Exception as e:
            print(e)
    # return traj


files = os.listdir(folder)

for file in files:
    if ".h5" in file:
        if not "stripped" in file:
            load_traj(folder + file)
