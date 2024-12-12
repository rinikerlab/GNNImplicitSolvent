import mdtraj
import numpy as np
import matplotlib.pyplot as plt
import nglview
import sys
import os

sys.path.append("../")
from Simulation.helper_functions import get_dihedrals_by_name
from Simulation.helper_functions import (
    minimize_mol,
    calculate_entropy,
    get_cluster_asignments_ordered,
)
from rdkit import Chem
from rdkit.Chem import AllChem
import nglview
import yaml
import argparse

parser = argparse.ArgumentParser(description="Run MB analysis")
parser.add_argument("-i", "--idx", type=int, help="Index of the molecule to analyze")
parser.add_argument("-mf", "--model_file", type=str, help="Model file to use")
parser.add_argument("--gbneck2", action="store_true", help="Use gbneck2 model")
parser.add_argument(
    "-c", "--cutoff", type=float, help="Cutoff for clustering", default=0.05
)
args = parser.parse_args()
idx = args.idx


solvent_dict = yaml.load(open("../Simulation/solvents.yml"), Loader=yaml.FullLoader)[
    "solvent_mapping_dict"
]
from copy import deepcopy


def get_mol(smiles, num_confs=1024):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if num_confs > 0:
        AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=42, useExpTorsionAnglePrefs=False
        )
    return mol


# Get molecules
set1 = ["N", "Nc2ccccc2", "Nc2ccccc(C#N)2", "Nc2ccccc(OC)2"]
set2 = [
    "NC(=O)C",
    "NC(=O)C(F)(F)F",
    "NC(=O)C(C)(C)C",
    "NC(=O)c2ccccc2",
    "NC(=O)c2cc(OC)ccc2",
    "NC(=O)c2ccc(F)cn2",
    "NC(=O)c2ccccn2",
]
As, Bs, Cs, Ds = [], [], [], []
for X in set1:
    As.append("Fc1ccc(N(C=O)Cc2ccccc2%s)cc1" % X)
    Bs.append("Fc1ccc(N(C=O)CCc2ccccc2%s)cc1" % X)
for X in set2:
    Cs.append("Fc1ccc(N(C=O)Cc2ccccc2%s)cc1" % X)
    Ds.append("Fc1ccc(N(C=O)CCc2ccccc2%s)cc1" % X)

all_compounds = As + Bs + Cs + Ds
labels = (
    ["A%i" % (i + 1) for i in range(len(As))]
    + ["B%i" % (i + 1) for i in range(len(Bs))]
    + ["C%i" % (i + 1) for i in range(len(Cs))]
    + ["D%i" % (i + 1) for i in range(len(Ds))]
)

# Perform Run
os.system("mkdir -p Minimizations/MB_results/%s" % labels[idx])
mol = get_mol(all_compounds[idx], num_confs=5120)

solvents = [
    "Chloroform",
    "acetone",
    "acetonitrile",
    "Ethylacetate",
    "THF",
    "DCM",
    "Ethanol",
    "Methanol",
    "DMSO",
]

solvents += ["gbneck2_" + solvent for solvent in solvents]

results = {}
for solvent in solvents:
    if args.gbneck2:
        solvent = "gbneck2_" + solvent
    adapted_mol, traj, energies = minimize_mol(
        deepcopy(mol),
        solvent,
        args.model_file,
        solvent_dict,
        return_traj=True,
        strides=32,
    )
    traj.save_hdf5(
        "Minimizations/MB_results/%s/%s_%s.h5" % (labels[idx], labels[idx], solvent)
    )
    np.save(
        "Minimizations/MB_results/%s/%s_%s.npy" % (labels[idx], labels[idx], solvent),
        energies,
    )

    # Make clustering and get free energies of cluster centers
    mol_p = Chem.MolFromSmiles(all_compounds[idx])
    permutations = mol_p.GetSubstructMatches(mol_p, useChirality=True, uniquify=False)

    dhcalc = get_dihedrals_by_name(traj, *"C4 N1 C5 H3".split(" "))
    closed_open_flag = (dhcalc < -2) | (dhcalc > 2)
    cluster_center_traj, cluster_energies, adapted_mol = get_cluster_asignments_ordered(
        traj,
        energies,
        thresh=args.cutoff,
        energy_thresh=100,
        additional_requirements=closed_open_flag,
        mol=adapted_mol,
        permutations=permutations,
    )
    entropies, free_energy = calculate_entropy(
        adapted_mol,
        solvent,
        "../MachineLearning/trained_models/ProductionRun_seed_1612_49_ckpt.pt",
        solvent_dict,
        forcefield="openff-2.0.0",
        strides=1,
    )

    cluster_center_traj.save_hdf5(
        "Minimizations/MB_results/%s/%s_%s_cluster_center.h5"
        % (labels[idx], labels[idx], solvent)
    )
    np.save(
        "Minimizations/MB_results/%s/%s_%s_cluster_center_energies.npy"
        % (labels[idx], labels[idx], solvent),
        cluster_energies,
    )
    np.save(
        "Minimizations/MB_results/%s/%s_%s_cluster_center_entropies.npy"
        % (labels[idx], labels[idx], solvent),
        entropies,
    )
    np.save(
        "Minimizations/MB_results/%s/%s_%s_cluster_center_free_energy.npy"
        % (labels[idx], labels[idx], solvent),
        free_energy,
    )
