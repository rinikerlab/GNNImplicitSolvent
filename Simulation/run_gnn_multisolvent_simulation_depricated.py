import argparse

parser = argparse.ArgumentParser(description="Run ML MM")
# Simulation
parser.add_argument(
    "-id", "--pdb_id", type=str, help="ID of pdb, if available amber files will be used"
)
parser.add_argument(
    "-ns", "--ns", default=20, type=float, help="Time of simulation in ns"
)
parser.add_argument("-fi", "--file", type=str, help="file to map ids to smiles")
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
    "-sm",
    "--solvent_model",
    default=[0],
    type=int,
    nargs="+",
    help="solvent model (list)",
)
parser.add_argument(
    "-sd",
    "--solvent_dielectrics",
    default=[78.5],
    type=float,
    nargs="+",
    help="solvent dielectric (list)",
)
# GNN
parser.add_argument("-b", "--batchsize", default=32, type=int, help="Batchsize")
parser.add_argument("-p", "--per", default=0.8, type=float, help="fraction of training")
parser.add_argument("-f", "--fra", default=0.1, type=float, help="scaling parameter")
parser.add_argument("-r", "--random", default=161311, type=int, help="random seed")
parser.add_argument("-ra", "--radius", default=0.6, type=float, help="radius")
parser.add_argument("-l", "--lr", default=0.001, type=float, help="learning rate")
parser.add_argument(
    "-fpt",
    "--ptfile",
    default=".",
    type=str,
    help="Pytorch file with stored training data",
)
parser.add_argument("-e", "--epochs", default=30, type=int, help="epochs to train for")
parser.add_argument("-m", "--modelid", default=0, type=int, help="Model_architecture")
parser.add_argument("-n", "--name", default="", type=str, help="name of model")
parser.add_argument("-c", "--clip", default=0, type=float, help="norm clipping")
parser.add_argument(
    "-s", "--solvent_dielectric", default=78.5, type=float, help="solvent dielectric"
)
parser.add_argument(
    "-mfile",
    "--model_file",
    default=".",
    type=str,
    help="Pytorch file with stored model",
)
parser.add_argument(
    "-sf", "--scaling_factor", default=2, type=float, help="scaling factor"
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
    pdb_id = args.pdb_id + "_in_v"
    run_name = pdb_id
    save_name = pdb_id
else:
    pdbs = pd.read_table(args.file, names=["smi"]).values.T[0]
    pdb_id = pdbs[int(args.pdb_id)] + "_in_v"
    run_name = args.file.split("/")[-1].split(".")[0]
    save_name = run_name + "_id_%i" % int(args.pdb_id)


# make string of commands
model_inf_string = ""
for arg in vars(args):
    if arg in [
        "folder",
        "mapping_file",
        "ptfile",
        "npfile",
        "pdb_id",
        "ns",
        "file",
        "num_rep",
        "addition",
        "solvent_model",
        "solvent_dielectrics",
    ]:
        continue

    model_inf_string += "_%s_%s" % (arg, getattr(args, arg))

# Setup runmodel
solvent_model = []
solvent_dielectric = []

if args.solvent_names == []:
    for repetition in range(args.num_rep):
        solvent_model += args.solvent_model
        solvent_dielectric += args.solvent_dielectrics
else:
    solvent_dict = yaml.load(open(args.solvent_names_yaml), Loader=yaml.FullLoader)[
        "solvent_mapping_dict"
    ]
    for repetition in range(args.num_rep):
        for solvent in args.solvent_names:
            solvent_model.append(solvent_dict[solvent]["solvent_id"])
            solvent_dielectric.append(solvent_dict[solvent]["dielectric"])


class GNN3_multisolvent_run_multiple_e(GNN3_Multisolvent_embedding_run_multiple):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10.0,
        unique_radii=None,
        hidden=64,
        num_solvents=42,
        hidden_token=128,
        scaling_factor=2.0,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            num_reps,
            gbneck_radius,
            unique_radii,
            hidden,
            78.5,
            num_solvents=num_solvents,
            hidden_token=hidden_token,
            scaling_factor=scaling_factor,
        )

    def set_num_reps(
        self,
        num_reps=len(solvent_model),
        solvent_models=solvent_model,
        solvent_dielectric=solvent_dielectric,
    ):
        return super().set_num_reps(num_reps, solvent_models, solvent_dielectric)


class GNN3_multisolvent_e(GNN3_Multisolvent_embedding):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        gbneck_radius=10.0,
        unique_radii=None,
        hidden=64,
        num_solvents=42,
        hidden_token=128,
        scaling_factor=2.0,
    ):
        super().__init__(
            fraction=fraction,
            radius=radius,
            max_num_neighbors=max_num_neighbors,
            parameters=parameters,
            device=device,
            jittable=jittable,
            unique_radii=unique_radii,
            hidden=hidden,
            num_solvents=num_solvents,
            hidden_token=hidden_token,
            scaling_factor=scaling_factor,
        )


setup_dict_multisolv = {
    "trained_model": args.model_file,
    "model": GNN3_multisolvent_e,
    "run_model": GNN3_multisolvent_run_multiple_e,
}


run_name += "_multi_"
run_name += args.addition
work_dir = "../"  # directory of the repository
n_interval = 100  # Interval for saving frames in steps
ns = args.ns  # Nanoseconds to run the simulation for
num_rep = args.num_rep

smiles = pdb_id.split("_")[0]
mol = smiles_to_mol(smiles)
dg_traj, dgmol = calculate_DGv3(
    mol, len(solvent_model), return_mol=True, pruneRmsThresh=0
)
num_confs = dg_traj.n_frames
gnn_sim = create_gnn_sim(
    smiles,
    cache="/tmp/tmp.cache",
    num_confs=num_confs,
    setup_dict=setup_dict_multisolv,
    additional_parameters={"solvent_model": -1, "solvent_dielectric": 78.5},
    workdir=work_dir,
    run_name=run_name,
    fraction=args.fra,
    scaling_factor=args.scaling_factor,
)
gnn_sim = set_positions_for_simulation(gnn_sim, dgmol, num_confs=num_confs, iteration=0)
gnn_sim = run_minimisation(gnn_sim)
n_steps = ns / 0.002 * 1000
gnn_sim.run_simulation(n_steps, n_interval, minimize=False)
gnn_sim.save_states(0)
