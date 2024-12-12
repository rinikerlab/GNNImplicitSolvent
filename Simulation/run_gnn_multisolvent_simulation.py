import argparse

parser = argparse.ArgumentParser(description="Run ML MM")
# Simulation
parser.add_argument("-id", "--smiles", type=str, help="ID in file or smiles")
parser.add_argument(
    "-ns", "--ns", default=20, type=float, help="Time of simulation in ns"
)
parser.add_argument(
    "-fi", "--file", default="none", type=str, help="file to map ids to smiles"
)
parser.add_argument(
    "-nr", "--num_rep", default=20, type=int, help="replicates to simulate at once"
)
parser.add_argument(
    "-a", "--addition", default="", type=str, help="Additional information"
)
parser.add_argument(
    "-sn",
    "--solvent_names",
    default=[],
    type=str,
    nargs="+",
    help="solvent names list",
)
parser.add_argument(
    "-sny",
    "--solvent_names_yaml",
    default="solvents.yml",
    type=str,
    help="yaml file with solvent names and mapping",
)
parser.add_argument(
    "-mfile",
    "--model_file",
    default=".",
    type=str,
    help="Pytorch file with stored model",
)
args = parser.parse_args()

import sys

sys.path.append("../")
sys.path.append("../MachineLearning/")

import os
from Simulator import Simulator, Multi_simulator
from ForceField.Forcefield import (
    TIP5P_force_field,
    TIP3P_force_field,
    GB_Neck2_force_field,
    Vacuum_force_field,
    Vacuum_force_field_plus_custom,
    Vacuum_force_field,
    OpenFF_forcefield_vacuum_plus_custom,
    OpenFF_forcefield_vacuum,
)
import yaml
from openmmtorch import TorchForce
from openmm import LangevinMiddleIntegrator
from MachineLearning.GNN_Models import *
from MachineLearning.GNN_Trainer import Trainer
from openmmtorch import TorchForce
from openmm.unit import kelvin, picosecond, picoseconds, bar
import pandas as pd
from helper_functions import (
    create_gnn_sim,
    smiles_to_mol,
    calculate_DGv3,
    set_positions_for_simulation,
    run_minimisation,
)

if args.file == "none":
    smiles_run_name = args.smiles + "_in_v"
    run_name = smiles_run_name
    save_name = smiles_run_name
else:
    smiles = pd.read_table(args.file, names=["smi"]).values.T[0]
    smiles_run_name = smiles[int(args.smiles)] + "_in_v"
    run_name = args.file.split("/")[-1].split(".")[0]
    save_name = run_name + "_id_%i" % int(args.smiles)

solvent_dict = yaml.load(open(args.solvent_names_yaml), Loader=yaml.FullLoader)[
    "solvent_mapping_dict"
]

run_name += "_multi_"
run_name += str(args.num_rep) + "_"
run_name += args.addition
work_dir = "../"  # directory of the repository
n_interval = 100  # Interval for saving frames in steps
ns = args.ns  # Nanoseconds to run the simulation for
num_rep = args.num_rep

smiles = smiles_run_name.split("_")[0]
mol = smiles_to_mol(smiles)
dg_traj, dgmol = calculate_DGv3(
    mol, args.num_rep * len(args.solvent_names), return_mol=True, pruneRmsThresh=0
)

num_confs = dg_traj.n_frames
os.system("mkdir -p run_caches")
model_dict = torch.load(args.model_file)["model"]
gnn_sim = create_gnn_sim(
    smiles,
    cache="run_caches/" + run_name + ".cache",
    num_confs=num_confs,
    workdir=work_dir,
    run_name=run_name,
    save_name=save_name,
    rdkit_mol=dgmol,
    solvent=args.solvent_names,
    solvent_dict=solvent_dict,
    model_dict=model_dict,
)

gnn_sim = set_positions_for_simulation(gnn_sim, dgmol, num_confs=num_confs, iteration=0)
gnn_sim, _ = run_minimisation(gnn_sim)
n_steps = ns / 0.002 * 1000
gnn_sim.run_simulation(n_steps, n_interval, minimize=False)
gnn_sim.save_states(0)
