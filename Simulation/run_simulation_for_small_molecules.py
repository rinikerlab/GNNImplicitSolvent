import argparse

parser = argparse.ArgumentParser(description="Run explicit solvent simulation")
parser.add_argument("-id", "--smi_id", type=str, help="ID of smiles in file")
parser.add_argument(
    "-n", "--ns", type=float, help="Time of simulation in ns", default=20
)
parser.add_argument(
    "-f", "--file", type=str, help="file to map ids to smiles", default="none"
)
parser.add_argument(
    "-s",
    "--solvent",
    type=str,
    help="Smiles of solvent to use or vac, GBNeck2 or SAGBNeck2 for vacuum, GBNeck2, or SAGBNeck2 implicit solvent respectively",
    default="O",
)
parser.add_argument("-r", "--random", type=int, help="random seed", default=0)
parser.add_argument("-i", "--interval", type=int, help="interval", default=1000)
parser.add_argument(
    "-ad", "--addition", type=str, help="Additional information", default=""
)
args = parser.parse_args()

import sys

sys.path.append("../")
from Simulator import Simulator
from ForceField.Forcefield import (
    OpenFF_forcefield,
    OpenFF_forcefield_GBNeck2,
    OpenFF_forcefield_vacuum,
    OpenFF_forcefield_SAGBNeck2,
)
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.unit import kelvin, picosecond, picoseconds, bar
import pandas as pd
from typing import DefaultDict

forcefield_dict = DefaultDict(lambda: OpenFF_forcefield)
forcefield_dict["SAGBNeck2"] = OpenFF_forcefield_SAGBNeck2
forcefield_dict["GBNeck2"] = OpenFF_forcefield_GBNeck2
forcefield_dict["vac"] = OpenFF_forcefield_vacuum

solvent_model_dict = lambda x: x if not x in ["GBNeck2", "vac", "SAGBNeck2"] else "v"

if args.file == "none":
    pdb_id = args.smi_id + "_in_" + solvent_model_dict(args.solvent)
    run_name = pdb_id
    save_name = pdb_id + args.addition
else:
    pdbs = pd.read_table(args.file, names=["smi"]).values.T[0]
    pdb_id = pdbs[int(args.smi_id)] + "_in_" + solvent_model_dict(args.solvent)
    run_name = args.file.split("/")[-1].split(".")[0]
    save_name = run_name + args.addition + "_id_%i" % int(args.smi_id)

work_dir = "../"  # directory of the repository
n_interval = args.interval  # Interval for saving frames in steps
ns = args.ns  # Nanoseconds to run the simulation for

sim = Simulator(
    work_dir=work_dir,
    pdb_id=pdb_id,
    run_name=run_name,
    save_name=save_name,
    random_number_seed=args.random,
)
sim.forcefield = forcefield_dict[args.solvent](pdb_id)
sim.integrator = LangevinMiddleIntegrator(
    300 * kelvin, 1 / picosecond, 0.002 * picoseconds
)
if solvent_model_dict(args.solvent) != "v":
    sim.barostat = MonteCarloBarostat(1 * bar, 300 * kelvin)
sim.platform = "GPU"
n_steps = ns / 0.002 * 1000
sim.run_simulation(n_steps=n_steps, minimize=True, n_interval=n_interval)
