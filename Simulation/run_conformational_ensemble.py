import mdtraj
import numpy as np
import matplotlib.pyplot as plt
import nglview
import sys
import os

sys.path.append("../")
from Simulation.helper_functions import (
    get_dihedrals_by_name,
    get_cluster_asignments_ordered,
    calculate_entropy,
)
from Simulation.helper_functions import minimize_mol
from rdkit import Chem
from rdkit.Chem import AllChem
import nglview
import yaml
import argparse

parser = argparse.ArgumentParser(description="Run MB analysis")
parser.add_argument("-i", "--idx", type=int, help="Index of the molecule to analyze")
parser.add_argument(
    "-nc", "--numcores", type=str, help="Number of cores to use", default="8"
)
parser.add_argument("-mf", "--model_file", type=str, help="Model file to use")
parser.add_argument("-r", "--random_seed", type=int, help="Random seed", default=42)
parser.add_argument(
    "-c", "--cutoff", type=float, help="Cutoff for clustering", default=0.05
)
args = parser.parse_args()
idx = args.idx

os.environ["OMP_NUM_THREADS"] = args.numcores
os.environ["OPENMM_CPU_THREADS"] = args.numcores

solvent_dict = yaml.load(open("../Simulation/solvents.yml"), Loader=yaml.FullLoader)[
    "solvent_mapping_dict"
]
from copy import deepcopy


def get_mol(smiles, num_confs=1024):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if num_confs > 0:
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_confs,
            randomSeed=args.random_seed,
            useExpTorsionAnglePrefs=False,
            numThreads=int(args.numcores),
        )
    return mol


all_smiles = list(
    np.loadtxt(
        "../Simulation/simulation_smiles/conformational_ensemble_smiles.txt",
        dtype=str,
        comments=None,
    )
)
labels = ["conformational_ensemble_id_%i" % i for i in range(len(all_smiles))]

# Perform Run
os.system("mkdir -p Minimizations/conformational_ensemble/%s" % labels[idx])
os.system("mkdir -p Minimizations/conformational_ensemble/caches")
mol = get_mol(all_smiles[idx], num_confs=5120)

solvents = [
    "Chloroform",
    "tip3p",
    "DMSO",
    "Methanol",
    "vac",
    "gbneck2_Chloroform",
    "gbneck2_tip3p",
    "gbneck2_DMSO",
    "gbneck2_Methanol",
]

results = {}
for solvent in solvents:

    savename = "none" if solvent == "vac" else solvent + str(idx)
    adapted_mol, traj, energies = minimize_mol(
        deepcopy(mol),
        solvent,
        args.model_file,
        solvent_dict,
        return_traj=True,
        strides=32,
        cache="Minimizations/conformational_ensemble/caches/%s.cache" % labels[idx],
        save_name=savename,
    )

    # Make clustering and get free energies of cluster centers
    mol_p = Chem.MolFromSmiles(all_smiles[idx])
    permutations = mol_p.GetSubstructMatches(mol_p, useChirality=True, uniquify=False)

    cluster_center_traj, cluster_energies, adapted_mol = get_cluster_asignments_ordered(
        traj,
        energies,
        thresh=args.cutoff,
        energy_thresh=100,
        mol=adapted_mol,
        permutations=permutations,
    )

    entropies, free_energy = calculate_entropy(
        adapted_mol,
        solvent,
        args.model_file,
        solvent_dict,
        forcefield="openff-2.0.0",
        strides=1,
        save_name=savename,
    )

    cluster_center_traj.save_hdf5(
        "Minimizations/conformational_ensemble/%s/%s_%s_seed_%i_cluster_center.h5"
        % (labels[idx], labels[idx], solvent, args.random_seed)
    )
    np.save(
        "Minimizations/conformational_ensemble/%s/%s_%s_seed_%i_cluster_center_energies.npy"
        % (labels[idx], labels[idx], solvent, args.random_seed),
        cluster_energies,
    )
    np.save(
        "Minimizations/conformational_ensemble/%s/%s_%s_seed_%i_cluster_center_entropies.npy"
        % (labels[idx], labels[idx], solvent, args.random_seed),
        entropies,
    )
    np.save(
        "Minimizations/conformational_ensemble/%s/%s_%s_seed_%i_cluster_center_free_energy.npy"
        % (labels[idx], labels[idx], solvent, args.random_seed),
        free_energy,
    )
