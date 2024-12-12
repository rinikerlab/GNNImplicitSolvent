"""
This file is used to calculate the forces for training.
"""

import argparse

parser = argparse.ArgumentParser(description="Sanitise processed files")
parser.add_argument("-i", "--runid", type=int, help="ID of smiles to process")
parser.add_argument(
    "-n", "--numberofsmiles", type=int, help="Number of molecules to process"
)
parser.add_argument("-s", "--solvent", type=str, help="Solvent to use")
parser.add_argument(
    "-r", "--randomseed", type=int, help="Random seed to use", default=161311
)
parser.add_argument("-f", "--file", type=str, help="File to process")
parser.add_argument(
    "-c",
    "--cachelocation",
    type=str,
    help="Location of cache",
    default="Calculated_caches/",
)
parser.add_argument("--cacheonly", action="store_true", help="Only calculate cache")
parser.add_argument(
    "-nc", "--numcores", type=str, help="Number of cores to use", default="4"
)
parser.add_argument(
    "-nf", "--numframes", type=int, help="Number of frames to use", default=3
)
parser.add_argument(
    "--startonly", action="store_true", help="Only generate starting structures"
)
parser.add_argument(
    "-st",
    "--starttrajloc",
    type=str,
    help="Location of starting trajectory",
    default="Calculated_starting_trajectories/",
)
parser.add_argument(
    "-sl", "--saveloc", type=str, help="Location to save", default="Calculated_data/"
)
parser.add_argument(
    "--oneonly",
    action="store_true",
    help="Only calculate one frame and save only one to file",
)
args = parser.parse_args()

runid = args.runid
numberofsmiles = args.numberofsmiles
seed = args.randomseed
file_path = args.file
cacheloc = args.cachelocation
numframes = args.numframes
starttrajloc = args.starttrajloc

solvent = args.solvent if not ("TIP" in args.solvent) else "O"
water_model = "TIP3P" if not ("TIP" in args.solvent) else args.solvent


import os

os.environ["OMP_NUM_THREADS"] = args.numcores
os.environ["OPENMM_CPU_THREADS"] = args.numcores

import sys

sys.path.append("../")
from Data.Datahandler import hdf5_storage
from rdkit import Chem

import sys

sys.path.append("../")
sys.path.append("../MachineLearning")
from MachineLearning.GNN_Trainer import Trainer
from Simulation.Simulator import Simulator
from ForceField.Forcefield import OpenFF_forcefield
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.unit import kelvin, picosecond, picoseconds, bar
import numpy as np
from Data.Datahandler import hdf5_storage
from Simulation.Simulator import Explicit_solvent_simulator_force_only
import os
import time
import pandas as pd
import mdtraj


if solvent != "O":
    storage = hdf5_storage(
        args.saveloc
        + "/"
        + solvent
        + "_small_molecules_n_%i_id_%i_seed_%i.hdf5" % (numberofsmiles, runid, seed)
    )
else:
    storage = hdf5_storage(
        args.saveloc
        + "/"
        + water_model
        + "_small_molecules_n_%i_id_%i_seed_%i.hdf5" % (numberofsmiles, runid, seed)
    )

am1bcc_cache = cacheloc + "/small_molecules_n_%i_id_%i.json" % (numberofsmiles, runid)

c_file = os.environ["TMPDIR"] + "/" + file_path.split("/")[-1]
os.system("cp %s %s" % (file_path, c_file))

if file_path.endswith(".txt"):
    smiles_to_process = pd.read_csv(c_file, header=None).values.flatten()[
        numberofsmiles * runid : numberofsmiles * (runid + 1)
    ]
elif file_path.endswith(".npy"):
    smiles_to_process = np.load(c_file, allow_pickle=True)[
        numberofsmiles * runid : numberofsmiles * (runid + 1)
    ]


for m, smiles in enumerate(smiles_to_process):

    molid = str(m)
    confid = "0"

    # Check if already processed
    smie, sime, forcee, ate = storage.get_existing_entries(
        molid, smiles=smiles, confid=confid
    )
    if np.sum((smie, sime, forcee, ate)) == 4:
        continue

    if not smie:
        storage.create_smiles_entry(molid, smiles, confid)

    try:
        print("START")
        print("Working on %s" % smiles, flush=True)
        start_time = time.time()
        pdb_id = smiles + "_in_" + solvent

        # Setup Simulation
        work_dir = os.environ["TMPDIR"] + "/"  # directory of the repository
        if args.oneonly:
            n_interval = 90000
        else:
            n_interval = 10000  # Interval for saving frames in steps
        ns = 0.5  # Nanoseconds to run the simulation for

        run_name = "Small_molecules_n_%i_id_%i" % (numberofsmiles, runid)
        save_name = "Small_molecules_molid_%s_confid_%s" % (molid, confid)

        # Simulate
        if os.path.isfile(starttrajloc + run_name + "_%i.h5" % m):
            starttraj = mdtraj.load(starttrajloc + run_name + "_%i.h5" % m)
            print(
                "used precomputed starting trajectory",
                starttrajloc + run_name + "_%i.h5" % m,
                flush=True,
            )
        else:
            starttraj = None
        sim = Simulator(
            work_dir=work_dir,
            pdb_id=pdb_id,
            run_name=run_name,
            save_name=save_name,
            random_number_seed=seed,
            starting_traj=starttraj,
        )
        if args.startonly:
            sim.save_starting_trajectory_to_file(starttrajloc + run_name + "_%i.h5" % m)
            continue

        forcefieldtime = time.time()
        sim.forcefield = OpenFF_forcefield(pdb_id, water_model, cache=am1bcc_cache)
        sim.integrator = LangevinMiddleIntegrator(
            300 * kelvin, 1 / picosecond, 0.002 * picoseconds
        )
        sim.barostat = MonteCarloBarostat(1 * bar, 300 * kelvin)
        sim.platform = "GPU"

        ## If only cache is needed
        if args.cacheonly:
            continue

        print("TIFO")
        print(time.time() - forcefieldtime)
        print("END", flush=True)

        if args.oneonly:
            n_steps = 90000
        else:
            n_steps = (1 + 8 * numframes) * n_interval

        sim.run_simulation(n_steps=n_steps, n_interval=n_interval)
        sim._simulation.reporters[0].close()

        # Save Simulation
        storetime = time.time()
        sim_traj = storage.create_simulation_entry_from_files(
            molid, confid, sim.hdf5_loc, sim.log_loc
        )

        print("TISTO")
        print(time.time() - storetime)
        print("END", flush=True)

        # Delete Simulation to free up threads
        del sim

        print("TISI")
        print(time.time() - start_time)
        print("END", flush=True)

        # Extract Forces
        test = Explicit_solvent_simulator_force_only(
            work_dir=work_dir,
            name="ligandsforce",
            run_name="ligandsforce_%i" % runid,
            pdb_id=pdb_id,
            hdf5_file=None,
            boxsize=None,
            save_name=save_name,
            starting_frame_traj=sim_traj,
            pdb=None,
            cache=am1bcc_cache,
            random_number_seed=seed,
            create_data=False,
            solvent_model=water_model,
        )
        if args.oneonly:
            frames = [0]
        else:
            frames = [8 + i * 8 for i in range(numframes)]

        savedir = os.environ["TMPDIR"] + "/"
        test.read_in_frame_and_set_positions(0)
        test.constrain_solute()
        force_file, pos_file, frames_file = test.calculate_mean_force_for_pre_calc_pos(
            save_location=savedir,
            save_add="run_%i" % runid,
            n_steps=100,
            n_frames=1000,
            frames=frames,
        )
        time.sleep(1)
        # Save Forces
        storage.create_extraction_entry_from_file(
            molid, confid, force_file, pos_file, frames_file
        )
        storage.create_reprocessed_force_entry(molid, confid, np.load(force_file))

        del test

        print("TIEX")
        print(time.time() - start_time)
        print("END", flush=True)

        atom_features, uniqueRadii = Trainer.get_gbneck2_param_small_molecules_unique(
            smiles + "_in_v", work_dir=os.environ["TMPDIR"] + "/", cache=am1bcc_cache
        )
        storage.create_atom_feature_entry(molid, confid, atom_features, uniqueRadii)
        storage.create_reprocessed_atom_feature_entry(
            molid, confid, atom_features, uniqueRadii
        )
        print("atom features added", flush=True)

        print("TIME")
        print(time.time() - start_time)
        print("END", flush=True)

    except Exception as e:
        print("ERROR")
        print(e)
        print(smiles)
        print("END", flush=True)

        storage.create_error_entry(molid, confid, str(e))

        continue
