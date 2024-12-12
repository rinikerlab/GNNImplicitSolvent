import sys
from pathlib import Path

file_path = Path(__file__).parent
sys.path.append(str(file_path.parent))
sys.path.append(str(file_path.parent / "MachineLearning"))
sys.path.append(str(file_path.parent / "Simulation"))
from Simulator import Multi_simulator
from ForceField.Forcefield import (
    OpenFF_forcefield_vacuum,
    OpenFF_forcefield_vacuum_plus_custom,
    OpenFF_forcefield_GBNeck2,
)
from openmm import LangevinMiddleIntegrator
from MachineLearning.GNN_Models import (
    GNN3_scale_64,
    GNN3_scale_64_run,
    GNN3_scale_64_SA_flex,
    GNN3_scale_64_SA_flex_run,
    GNN3_Multisolvent_embedding,
    GNN3_Multisolvent_embedding_run_multiple,
    GNN3_Multisolvent_embedding_run_multiple_Delta,
    DEFAULT_UNIQUE_RADII,
)
from MachineLearning.GNN_Trainer import Trainer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mtl
import mdtraj
import numpy as np
from Analysis.analysis import load_parallel_traj
from openmm.app import HBonds
from openmm.unit import (
    Quantity,
    kilojoule,
    mole,
    picosecond,
    picoseconds,
    kelvin,
    kilojoule_per_mole,
    nanometer,
    dalton,
    meter,
)
from scipy.spatial import ConvexHull
from rdkit.Geometry import Point3D
import torch

import tempfile
import os
import warnings

import yaml

SOLVENT_DICT = yaml.load(
    open(f"{str(file_path.parent)}/Simulation/solvents.yml"), Loader=yaml.FullLoader
)["solvent_mapping_dict"]
MODEL_PATH = f"{str(file_path.parent)}/MachineLearning/trained_models/GNN.pt"
MODEL_PATH_WATER_ONLY = (
    f"{str(file_path.parent)}/MachineLearning/trained_models/GNN_WATER_ONLY.pt"
)


def get_dihedrals_by_name(traj, i, j, k, l, ins=2):

    if isinstance(traj, list):
        dis = []
        for t in traj:
            dis.append(get_dihedrals_by_name(t, i, j, k, l, ins))
        return dis

    oc = 0
    for a, atom in enumerate(traj.top.atoms):
        if str(atom)[-len(i) :] == i:
            i_id = a
            oc += 1
        if str(atom)[-len(j) :] == j:
            j_id = a
            oc += 1
        if str(atom)[-len(k) :] == k:
            k_id = a
            oc += 1
        if str(atom)[-len(l) :] == l:
            l_id = a
            oc += 1
    assert oc == 4

    return mdtraj.compute_dihedrals(traj, [[i_id, j_id, k_id, l_id]])


def get_prob(d1, d2, num=50):
    z, xedge, yedge = np.histogram2d(
        d1, d2, density=True, bins=np.linspace(-np.pi, np.pi, num=num)
    )
    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])
    z = z.T

    return x, y, z


def get_xyz(d1, d2, num_bins=50):
    z, xedge, yedge = np.histogram2d(
        d1, d2, density=False, bins=np.linspace(-np.pi, np.pi, num=num_bins)
    )
    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])
    z = z.T

    kjz = in_kjmol(z)

    return x, y, kjz


def get_xz(d1):
    z, xedge = np.histogram(d1, bins=100)
    x = 0.5 * (xedge[:-1] + xedge[1:])
    z = z.T

    return x, in_kjmol(z)


def initialize_plt(figsize=(15, 10), fontsize=22):
    # initialice Matplotlib
    _ = plt.figure()
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams.update({"font.family": "Sans"})


def plot_free_energy(
    ax,
    dis,
    nbins,
    range=[0.1, 0.7],
    color="blue",
    label="",
    ret_xz=False,
    linewidth=5,
    linestyle="solid",
):

    if not isinstance(dis, list):
        dis = [dis]

    zs = []
    for d in dis:
        z, xedge = np.histogram(d, bins=nbins, range=range)
        x = 0.5 * (xedge[:-1] + xedge[1:])
        z = z.T
        zs.append(z)

    valuey = np.array([in_kjmol(z) for z in zs])
    mean_y = np.nanmean(valuey, axis=0) - np.min(
        np.nanmean(valuey, axis=0)
    )  # static shift
    std_y = np.nanstd(valuey, axis=0)
    if ret_xz:
        return x, mean_y, std_y
    ax.plot(
        x, mean_y, linewidth=linewidth, color=color, label=label, linestyle=linestyle
    )
    ax.fill_between(x=x, y1=mean_y - std_y, y2=mean_y + std_y, color=color, alpha=0.3)


def normalize(arr, set_zero_to_max=True):
    if set_zero_to_max:
        arr[arr == 0] = np.min(arr[arr > 0])
    return arr / np.sum(arr)


def in_kjmol(z, set_zero_to_max=True):
    kT = 2.479
    if set_zero_to_max:
        z[z == 0] = np.min(z[z > 0])
    return (
        -np.log(normalize(z, set_zero_to_max))
        - np.min(-np.log(normalize(z, set_zero_to_max)))
    ) * kT


def mol_to_traj(mol):
    with tempfile.NamedTemporaryFile() as tmp:
        Chem.MolToPDBFile(mol, tmp.name)
        traj = mdtraj.load_pdb(tmp.name)
    return traj


def set_traj_positions_in_mol(mol, traj):

    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    for i in range(traj.n_frames):
        conf = mol.GetConformer(0)
        for j in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                j,
                Point3D(
                    float(traj.xyz[i][j][0] * 10),
                    float(traj.xyz[i][j][1] * 10),
                    float(traj.xyz[i][j][2] * 10),
                ),
            )
        new_mol.AddConformer(conf, assignId=True)
    return new_mol


def get_distance_by_name(traj, n1, n2, ins=2):
    if isinstance(traj, list):
        dis = []
        for t in traj:
            dis.append(get_distance_by_name(t, n1, n2, ins))
        return dis

    oc = 0
    for a, atom in enumerate(traj.top.atoms):
        if str(atom)[-len(n1) :] == n1:
            n1_id = a
            oc += 1
        if str(atom)[-len(n2) :] == n2:
            n2_id = a
            oc += 1
    assert oc == 2

    return calculate_distance(traj.xyz, (n1_id, n2_id))


def calculate_distance(xyz, fromto):
    return np.reshape(
        np.sqrt(np.sum((xyz[:, fromto[0]] - xyz[:, fromto[1]]) ** 2, axis=1)),
        (xyz.shape[0], 1),
    )


def calculate_ETKDGv3(mol, conformer_num=1000, pruneRmsThresh=0.1):
    mol = Chem.AddHs(mol)

    etkdg = rdDistGeom.ETKDGv3()
    etkdg.randomSeed = 0xA700F
    etkdg.verbose = False
    etkdg.numThreads = 8
    etkdg.useExpTorsionAnglePrefs = True
    etkdg.ETversion = 2
    etkdg.pruneRmsThresh = pruneRmsThresh
    AllChem.EmbedMultipleConfs(mol, numConfs=conformer_num, params=etkdg)

    with tempfile.NamedTemporaryFile() as tmp:
        Chem.MolToPDBFile(mol, tmp.name)
        traj = mdtraj.load_pdb(tmp.name)

    return traj


def calculate_DGv3(mol, conformer_num=1000, return_mol=False, pruneRmsThresh=0.1):
    mol = Chem.AddHs(mol)

    etkdg = rdDistGeom.ETKDGv3()
    etkdg.randomSeed = 0xA700F
    etkdg.verbose = False
    etkdg.numThreads = 8
    etkdg.useExpTorsionAnglePrefs = False
    etkdg.ETversion = 2
    # use cutoff
    etkdg.pruneRmsThresh = pruneRmsThresh
    AllChem.EmbedMultipleConfs(mol, numConfs=conformer_num, params=etkdg)

    with tempfile.NamedTemporaryFile() as tmp:
        Chem.MolToPDBFile(mol, tmp.name)
        traj = mdtraj.load_pdb(tmp.name)

    if return_mol:
        return traj, mol
    return traj


def get_cluster_asignments_ordered(
    traj,
    energies,
    thresh=0.05,
    energy_thresh=2,
    verbose=False,
    additional_requirements=None,
    permutations=None,
    mol=None,
):

    selection = traj.top.select("element != H")
    if permutations is None:
        permutations = [selection]

    ordered_traj = mdtraj.Trajectory(traj.xyz[energies.argsort()], traj.top)
    ordered_energies = energies[energies.argsort()]
    cluster_center_traj = ordered_traj[0]
    cluster_energies = [energies[energies.argsort()][0]]

    index_sort_mapping = energies.argsort()

    indices = [index_sort_mapping[0]]
    if additional_requirements is not None:
        additional_requirements_sorted = additional_requirements[energies.argsort()]
        cluster_additional_requirements = [additional_requirements_sorted[0]]

    for i in np.arange(1, ordered_traj.n_frames):
        rmsds = np.zeros((len(permutations), len(cluster_center_traj)), dtype=np.float)
        for p, permutation in enumerate(permutations):
            rmsds[p] = mdtraj.rmsd(
                cluster_center_traj,
                ordered_traj,
                i,
                atom_indices=selection,
                ref_atom_indices=permutation,
            )
        rmsd = np.min(rmsds, axis=0)
        if np.min(rmsd) < thresh:
            if np.all(
                np.abs(
                    (np.array(cluster_energies)[rmsd < thresh] - ordered_energies[i])
                )
                > energy_thresh
            ) or (
                (additional_requirements is not None)
                and (
                    additional_requirements_sorted[i]
                    not in np.array(cluster_additional_requirements)[rmsd < thresh]
                )
            ):
                cluster_center_traj += ordered_traj[i]
                indices.append(index_sort_mapping[i])
                cluster_energies.append(ordered_energies[i])
                if additional_requirements is not None:
                    cluster_additional_requirements.append(
                        additional_requirements_sorted[i]
                    )
        else:
            cluster_center_traj += ordered_traj[i]
            indices.append(index_sort_mapping[i])
            cluster_energies.append(ordered_energies[i])
            if additional_requirements is not None:
                cluster_additional_requirements.append(
                    additional_requirements_sorted[i]
                )

    if mol is not None:
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        for i in indices:
            new_mol.AddConformer(mol.GetConformer(int(i)), assignId=True)

        return cluster_center_traj, cluster_energies, new_mol

    return cluster_center_traj, cluster_energies


def calculate_MMFF(mol, conformer_num=1000, pruneRmsThresh=0.1):
    mol = Chem.AddHs(mol)

    etkdg = rdDistGeom.ETKDGv3()
    etkdg.randomSeed = 0xA700F
    etkdg.verbose = False
    etkdg.numThreads = 8
    etkdg.useExpTorsionAnglePrefs = False
    etkdg.ETversion = 2
    etkdg.pruneRmsThresh = pruneRmsThresh

    AllChem.EmbedMultipleConfs(mol, numConfs=conformer_num, params=etkdg)
    AllChem.MMFFOptimizeMoleculeConfs(mol)

    with tempfile.NamedTemporaryFile() as tmp:
        Chem.MolToPDBFile(mol, tmp.name)
        traj = mdtraj.load_pdb(tmp.name)

    return traj


def load_trajectories(ce_id, load_gnn=True):
    file_location = "/fileserver/pine/pine8/kpaul/small_molecule_pub/Simulation/simulation/conformational_ensemble_smiles/"
    if os.path.isfile(
        file_location
        + "conformational_ensemble_smiles_id_%i_openff200_tip3p_0_161311_output_stripped.h5"
        % ce_id
    ):
        stripped_tip3p_traj = mdtraj.load(
            file_location
            + "conformational_ensemble_smiles_id_%i_openff200_tip3p_0_161311_output_stripped.h5"
            % ce_id
        )
    else:
        tip3p_traj = mdtraj.load(
            file_location
            + "conformational_ensemble_smiles_id_%i_openff200_tip3p_0_161311_output.h5"
            % ce_id
        )
        stripped_tip3p_traj = tip3p_traj.atom_slice(
            tip3p_traj.top.select("resid %i" % (tip3p_traj.top.n_residues - 1))
        )
        stripped_tip3p_traj.save(
            file_location
            + "conformational_ensemble_smiles_id_%i_openff200_tip3p_0_161311_output_stripped.h5"
            % ce_id
        )

    file_location = "/fileserver/pine/pine8/kpaul/small_molecule_pub/Simulation/simulation/conformational_ensemble_smiles_pub/"
    if load_gnn:
        gnn_traj = load_parallel_traj(
            nrep=128,
            file=file_location
            + "conformational_ensemble_smiles_id_%i_openff200_vacuum_plus_GNN3_multi_128_model_64_random_2_0_2_output.h5"
            % ce_id,
        )
    else:
        gnn_traj = None
    file_location = "/fileserver/pine/pine8/kpaul/small_molecule_pub/Simulation/simulation/conformational_ensemble_smiles/"
    return stripped_tip3p_traj, gnn_traj


def load_traj(traj_file):
    file_core = traj_file.split(".")[0]
    if os.path.isfile(file_core + "_stripped.h5"):
        traj = mdtraj.load(file_core + "_stripped.h5")
    else:
        traj = mdtraj.load(traj_file)
        traj = traj.atom_slice(traj.top.select("resid %i" % (traj.top.n_residues - 1)))
        traj.save(file_core + "_stripped.h5")
    return traj


def create_vac_sim(
    smiles,
    cache=None,
    num_confs=64,
    save_name=None,
    mol=None,
    partial_charges=None,
    forcefield="openff-2.0.0",
    constraints=HBonds,
):
    pdb_id = smiles + "_in_v"
    workdir = tempfile.mkdtemp() + "/"
    if save_name is None:
        save_name = str(np.random.randint(1000000))
    vac_sim = Multi_simulator(
        work_dir=workdir,
        pdb_id=pdb_id,
        run_name=save_name,
        num_rep=num_confs,
        cache=cache,
        save_name=save_name,
        openff_forcefield=forcefield,
        constraints=constraints,
    )
    vac_sim.forcefield = OpenFF_forcefield_vacuum(
        pdb_id,
        cache=cache,
        rdkit_mol=mol,
        partial_charges=partial_charges,
        forcefield=forcefield,
        constraints=constraints,
    )
    vac_sim.integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, 0.002 * picoseconds
    )
    vac_sim._ref_system.integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, 0.002 * picoseconds
    )
    vac_sim._ref_system.forcefield = OpenFF_forcefield_vacuum(
        pdb_id,
        cache=cache,
        rdkit_mol=mol,
        forcefield=forcefield,
        constraints=constraints,
    )
    vac_sim._ref_system.platform = "GPU"
    vac_sim.setup_replicates()
    return vac_sim


def create_gbneck_sim(
    smiles,
    cache=None,
    dielectric=78.5,
    SA=None,
    rdkit_mol=None,
    partial_charges=None,
    forcefield="openff-2.0.0",
    constraints=HBonds,
):

    pdb_id = smiles + "_in_v"
    workdir = tempfile.mkdtemp() + "/"
    gbneck_sim = Multi_simulator(
        work_dir=workdir,
        pdb_id=pdb_id,
        run_name=pdb_id,
        cache=cache,
        partial_charges=partial_charges,
        openff_forcefield=forcefield,
        constraints=constraints,
        rdkit_mol=rdkit_mol,
    )
    gbneck_sim.forcefield = OpenFF_forcefield_GBNeck2(
        pdb_id,
        cache=cache,
        solvent_dielectric=dielectric,
        SA=SA,
        rdkit_mol=rdkit_mol,
        partial_charges=partial_charges,
        forcefield=forcefield,
        constraints=constraints,
    )
    gbneck_sim.integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, 0.002 * picoseconds
    )

    return gbneck_sim


def load_explicit_solvent_reference(id, solvent_smiles, path):

    if os.path.isfile(
        path
        + "conformational_ensemble_smiles_id_%i_openff200_%s_0_161311_output_stripped.h5"
        % (id, solvent_smiles)
    ):
        stripped_traj = mdtraj.load(
            path
            + "conformational_ensemble_smiles_id_%i_openff200_%s_0_161311_output_stripped.h5"
            % (id, solvent_smiles)
        )
    else:
        traj = mdtraj.load(
            path
            + "conformational_ensemble_smiles_id_%i_openff200_%s_0_161311_output.h5"
            % (id, solvent_smiles)
        )
        stripped_traj = traj.atom_slice(
            traj.top.select("resid %i" % (traj.top.n_residues - 1))
        )
        stripped_traj.save(
            path
            + "conformational_ensemble_smiles_id_%i_openff200_%s_0_161311_output_stripped.h5"
            % (id, solvent_smiles)
        )

    return stripped_traj


def plot_single_contour_f(ax, x, y, kjz, colors="Blues_r", vmax=20, levels=10):

    # set axis equal
    ax.set_aspect("equal", "box")
    # set axis range
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    # set axis label
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    # set axis ticks
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
    )
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
    )
    contour = ax.contourf(x, y, kjz, levels=levels, cmap=colors, vmax=vmax)
    return contour


def plot_contour_f(axd, x, y, kjz, colors="Blues_r", vmax=20, levels=10):
    for ax in axd.values():
        countour = plot_single_contour_f(
            ax, x, y, kjz, colors=colors, vmax=vmax, levels=levels
        )
    return countour


def plot_colorbar_legend(ax, levels, vmax):
    levels = levels
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis("off")
    bounds = [vmax / levels * i for i in range(levels + 2)]
    cmap = "Greys_r"
    norm = mtl.colors.Normalize(vmin=0, vmax=vmax)
    axins = ax.inset_axes([0.25, 0, 0.5, 0.5])
    cb1 = mtl.colorbar.ColorbarBase(
        axins,
        cmap=cmap,
        norm=norm,
        boundaries=bounds,
        ticks=bounds[::2],
        orientation="horizontal",
    )
    cb1.set_label(r"free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$")


def set_positions_for_simulation(sim, mol, num_confs=64, iteration=0, offset=0):
    """Transfer num_confs conformers from mol to sim, starting from num_confs*iteration + offset"""
    positions = []
    for i in range(
        (iteration * num_confs) + offset, ((iteration + 1) * num_confs + offset)
    ):
        pos = mol.GetConformer(i).GetPositions()
        positions.append(pos / 10)
    positions = np.array(positions)
    sim.set_positions(positions.reshape(-1, positions.shape[-1]))
    return sim


def run_minimisation(sim, tolerance=1e-4, max_iterations=0):

    status = 0
    try:
        sim._simulation.minimizeEnergy(
            tolerance=tolerance, maxIterations=max_iterations
        )
    except:
        status = 1

    return sim, status


def get_minimised_positions(sim):
    return (
        sim._simulation.context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(nanometer)
    )


def get_traj_from_positions(mol, positions, num_iters=1, num_confs=64):
    with tempfile.NamedTemporaryFile() as tmp:
        Chem.MolToPDBFile(mol, tmp.name)
        traj = mdtraj.load_pdb(tmp.name)

    traj_optimised = traj[:]
    n_atoms = mol.GetNumAtoms()
    for ni in range(num_iters):
        for i in range(num_confs):
            traj_optimised.xyz[ni * num_confs + i] = positions[ni][
                i * n_atoms : (i + 1) * n_atoms
            ]

    return traj_optimised


def set_positions_in_mol(mol, positions):
    """Use an array of shape (1, n_atoms*n_frames, 3) to update n_frames conformers."""
    n_confs = mol.GetNumConformers()
    n_atoms = mol.GetNumAtoms()
    max_n_confs = len(positions[0]) // n_atoms
    if max_n_confs < n_confs:
        diff = n_confs - max_n_confs
        print(
            f"Warning: too few positions in set_positions_in_mol; the last {diff} conformers will not be updated."
        )
    for i in range(min(n_confs, max_n_confs)):
        conf = mol.GetConformer(i)
        for j in range(n_atoms):
            conf.SetAtomPosition(j, positions[0][i * n_atoms + j] * 10)
    return mol


from rdkit.Geometry import Point3D


def traj_to_mol(traj, mol):
    n_confs = mol.GetNumConformers()
    n_atoms = mol.GetNumAtoms()
    for i in range(n_confs):
        conf = mol.GetConformer(i)
        for j in range(n_atoms):
            conf.SetAtomPosition(
                j,
                Point3D(
                    float(traj.xyz[i][j][0]),
                    float(traj.xyz[i][j][1]),
                    float(traj.xyz[i][j][2]),
                ),
            )
    return mol


from openmmtorch import TorchForce


def kjmol_to_prop(energies):
    kT = 2.479
    props = np.exp(-energies / kT) / np.sum(np.exp(-energies / kT))

    return props


def create_gnn_sim(
    smiles,
    cache=None,
    num_confs=64,
    workdir=tempfile.mkdtemp() + "/",
    run_name="",
    save_name=None,
    rdkit_mol=None,
    solvent_dict=None,
    solvent="tip3p",
    num_solvents=42,
    model_dict=None,
    solvent_model=None,
    solvent_dielectric=None,
    partial_charges=None,
    forcefield="openff-2.0.0",
    constraints=HBonds,
):

    pdb_id = smiles + "_in_v"
    gnn_sim = Multi_simulator(
        work_dir=workdir,
        pdb_id=pdb_id,
        run_name=run_name,
        num_rep=num_confs,
        cache=cache,
        save_name=save_name,
        partial_charges=partial_charges,
        rdkit_mol=rdkit_mol,
        openff_forcefield=forcefield,
        constraints=constraints,
    )

    # add to function
    if solvent_model is None:
        if isinstance(solvent, str):
            solvent_model, solvent_dielectric = [], []
            for repetition in range(num_confs):
                solvent_model.append(solvent_dict[solvent]["solvent_id"])
                solvent_dielectric.append(solvent_dict[solvent]["dielectric"])
        else:
            solvent_model, solvent_dielectric = [], []
            for repetition in range(num_confs // len(solvent)):
                for s in solvent:
                    solvent_model.append(solvent_dict[s]["solvent_id"])
                    solvent_dielectric.append(solvent_dict[s]["dielectric"])

    parameters = Trainer.get_gbneck2_param(
        pdb_id,
        tempfile.mkdtemp() + "/",
        uniqueRadii=DEFAULT_UNIQUE_RADII,
        cache=cache,
        rdkit_mol=rdkit_mol,
        partial_charges=partial_charges,
        forcefield=forcefield,
    )

    assert parameters[1] == DEFAULT_UNIQUE_RADII
    parameters = parameters[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(model_dict, str):
        model_dict = torch.load(model_dict, map_location=device)

    hidden_shape = model_dict["interaction1.message1.weight"].shape[0]

    run_model = GNN3_Multisolvent_embedding_run_multiple(
        parameters=parameters,
        device=device,
        hidden=hidden_shape,
        num_solvents=num_solvents,
        solvent_models=[solvent_model[0]],
        solvent_dielectric=[solvent_dielectric[0]],
    )
    run_model.load_state_dict(model_dict)
    run_model.eval()

    # Create Ref file
    reffile = tempfile.NamedTemporaryFile(suffix=".pt").name
    torch.jit.optimize_for_inference(torch.jit.script(run_model.eval())).save(reffile)

    # Create Run file
    runfile = tempfile.NamedTemporaryFile(suffix=".pt").name
    run_model.set_num_reps(num_confs, solvent_model, solvent_dielectric)
    torch.jit.optimize_for_inference(torch.jit.script(run_model)).save(runfile)
    torch_force = TorchForce(runfile)
    torch_force.addGlobalParameter("solvent_model", -1)
    torch_force.addGlobalParameter("solvent_dielectric", 78.5)

    gnn_sim.integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, 0.002 * picoseconds
    )
    # Build the Runsystem
    gnn_sim.forcefield = OpenFF_forcefield_vacuum_plus_custom(
        pdb_id,
        torch_force,
        "GNN3_vap",
        cache=cache,
        rdkit_mol=rdkit_mol,
        partial_charges=partial_charges,
        forcefield=forcefield,
        constraints=constraints,
    )

    # Build the Refsystem
    torch_force = TorchForce(reffile)
    torch_force.addGlobalParameter("solvent_model", -1)
    torch_force.addGlobalParameter("solvent_dielectric", 78.5)

    gnn_sim._ref_system.integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, 0.002 * picoseconds
    )
    gnn_sim._ref_system.forcefield = OpenFF_forcefield_vacuum_plus_custom(
        pdb_id,
        torch_force,
        "GNN3_vap",
        cache=cache,
        rdkit_mol=rdkit_mol,
        partial_charges=partial_charges,
        forcefield=forcefield,
        constraints=constraints,
    )

    gnn_sim._ref_system.platform = "GPU"
    gnn_sim.setup_replicates(only_check_first=True)
    return gnn_sim


def create_gnn_model(
    mol,
    model_class=GNN3_Multisolvent_embedding_run_multiple,
    cache=None,
    solvent_dict=SOLVENT_DICT,
    solvent="tip3p",
    num_solvents=42,
    model_dict=None,
    partial_charges=None,
    forcefield="openff-2.0.0",
    device="cpu",
    jit=False,
):

    if isinstance(model_dict, str):
        model_dict = torch.load(model_dict, map_location=device)["model"]

    smiles = Chem.MolToSmiles(
        mol,
        canonical=False,
        isomericSmiles=True,
        doRandom=False,
        kekuleSmiles=False,
    )
    pdb_id = smiles + "_in_v"
    num_confs = 1

    if isinstance(solvent, str):
        solvent_model, solvent_dielectric = [], []
        for repetition in range(num_confs):
            solvent_model.append(solvent_dict[solvent]["solvent_id"])
            solvent_dielectric.append(solvent_dict[solvent]["dielectric"])
    else:
        solvent_model, solvent_dielectric = [], []
        for repetition in range(num_confs // len(solvent)):
            for s in solvent:
                solvent_model.append(solvent_dict[s]["solvent_id"])
                solvent_dielectric.append(solvent_dict[s]["dielectric"])

    parameters = Trainer.get_gbneck2_param(
        pdb_id,
        tempfile.mkdtemp() + "/",
        uniqueRadii=DEFAULT_UNIQUE_RADII,
        cache=cache,
        rdkit_mol=mol,
        partial_charges=partial_charges,
        forcefield=forcefield,
    )

    assert parameters[1] == DEFAULT_UNIQUE_RADII
    parameters = parameters[0]

    if isinstance(model_dict, str):
        model_dict = torch.load(model_dict, map_location=device)

    hidden_shape = model_dict["interaction1.message1.weight"].shape[0]

    run_model = model_class(
        parameters=parameters,
        device=device,
        hidden=hidden_shape,
        num_solvents=num_solvents,
        solvent_models=[solvent_model[0]],
        solvent_dielectric=[solvent_dielectric[0]],
    )
    run_model.load_state_dict(model_dict)
    run_model.eval()

    # Create Ref file
    if jit:
        return torch.jit.optimize_for_inference(torch.jit.script(run_model.eval()))
    else:
        return run_model.eval()


def get_energies(sim, traj, additional_parameters=[]):

    energies = []

    for i in range(traj.n_frames):
        if len(additional_parameters) != 0:
            for key, val in additional_parameters[i].items():
                sim._ref_system._simulation.context.setParameter(key, val)
        sim._ref_system.set_positions(traj.xyz[i])
        try:
            energy = sim._ref_system.calculate_energy()._value
        except:
            warnings.warn(f"Energy calculation failed for frame {i}")
            energy = np.nan

        energies.append(energy)

    return np.array(energies)


def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol


def get_boltzmann_mean(energies):
    return -np.log(np.mean(np.exp(-energies)))


def minimize_mol(
    mol,
    solvent,
    model_path=MODEL_PATH,
    solvent_dict=SOLVENT_DICT,
    return_traj=False,
    tolerance=1e-4,
    max_iterations=0,
    gnn_sim=None,
    return_gnn_sim=False,
    strides=1,
    cache=tempfile.mkdtemp() + "/tmp.cache",
    save_name=None,
    partial_charges=None,
    forcefield="openff-2.0.0",
    constraints=HBonds,
):
    """Minimize each conformer of an RdKit molecule

    :param mol: RdKit moleule with at least 1 conformer
    :param solvent: str, a solvent defined in *solvent_dict*
    :param model_path: str, the filename of a model .pt file
    :param return_traj: bool, whether to return a trajectory
    :param tolerance: float, tolerance for the optimizer
    :param max_iterations: int, maximum iterations for the optimizer, 0 means no maximum
    :param gnn_sim: Optional, a Simulation.Simulator object
    :param return_gnn_sim: bool, whether to return a Simulation.Simulator object
    :param strides: int, the number groups in which the optimization should be
        performed. If 1, all conformers need to fit in GPU ram simultaneously.
    :param cache: str, cache directory
    :param save_name: Optional str
    :param partial_charges: Optional array, if None, will generate AM1/BCC charges
    :param forcefield: str, an OpenFF-compatible force field
    :param constraints: OpenMM constraints class
    """

    # Redefine cache if partial charges are used
    if partial_charges is not None:
        cache = tempfile.NamedTemporaryFile().name

    if "gbneck" in solvent:
        return minimize_mol_gbneck(
            mol,
            solvent,
            solvent_dict,
            return_traj=return_traj,
            tolerance=1e-4,
            max_iterations=0,
            gnn_sim=None,
            return_gnn_sim=return_gnn_sim,
            strides=1,
            cache=cache,
            partial_charges=partial_charges,
            forcefield=forcefield,
            constraints=constraints,
        )

    num_rep = mol.GetNumConformers()
    num_iters = strides
    num_confs = num_rep // strides

    if gnn_sim is None:
        gnn_sim = get_gnn_sim(
            mol=mol,
            solvent=solvent,
            model_path=model_path,
            solvent_dict=solvent_dict,
            cache=cache,
            save_name=save_name,
            partial_charges=partial_charges,
            forcefield=forcefield,
            constraints=constraints,
            num_confs=num_confs,
        )
    # gnn_sim = set_positions_for_simulation(
    #     gnn_sim, mol, num_confs=num_confs, iteration=0
    # )
    # gnn_sim, status = run_minimisation(
    #     gnn_sim, tolerance=tolerance, max_iterations=max_iterations
    # )
    positions = []
    # positions.append(get_minimised_positions(gnn_sim))

    for i in tqdm(range(0, num_iters)):
        gnn_sim = set_positions_for_simulation(
            gnn_sim, mol, num_confs=num_confs, iteration=i
        )
        gnn_sim, status = run_minimisation(
            gnn_sim, tolerance=tolerance, max_iterations=max_iterations
        )
        if status == 0:
            positions.append(get_minimised_positions(gnn_sim))
        else:
            print(
                f"Batch Optimization failed for iteration {i}, starting individual optimization"
            )
            single_gnn_sim = get_gnn_sim(
                mol=mol,
                solvent=solvent,
                model_path=model_path,
                solvent_dict=solvent_dict,
                cache=cache,
                save_name=save_name,
                partial_charges=partial_charges,
                forcefield=forcefield,
                constraints=constraints,
                num_confs=1,
            )
            individual_positions = []
            for j in range(i * num_confs, (i + 1) * num_confs):
                single_gnn_sim = set_positions_for_simulation(
                    single_gnn_sim, mol, num_confs=1, iteration=0, offset=j
                )
                single_gnn_sim, status = run_minimisation(
                    single_gnn_sim, tolerance=tolerance, max_iterations=max_iterations
                )
                if status == 0:
                    individual_positions.append(get_minimised_positions(single_gnn_sim))
                else:
                    print(f"Individual optimization failed for confomer {j}")
                    individual_positions.append(np.zeros((mol.GetNumAtoms(), 3)))
            positions.append(np.concatenate(individual_positions, axis=0))

            del single_gnn_sim

    n_missing = num_rep - num_iters * num_confs
    if n_missing:
        print(f"Individually optimizing the last {n_missing} conformers")
        gnn_sim_extra = get_gnn_sim(
            mol=mol,
            solvent=solvent,
            model_path=model_path,
            solvent_dict=solvent_dict,
            cache=cache,
            save_name=save_name,
            partial_charges=partial_charges,
            forcefield=forcefield,
            constraints=constraints,
            num_confs=n_missing,
        )
        gnn_sim_extra = set_positions_for_simulation(
            gnn_sim_extra,
            mol,
            num_confs=n_missing,
            iteration=0,
            offset=num_iters * num_confs,
        )
        gnn_sim_extra, status = run_minimisation(
            gnn_sim_extra, tolerance=tolerance, max_iterations=max_iterations
        )
        if status == 0:
            positions.append(get_minimised_positions(gnn_sim_extra))
        else:
            del gnn_sim_extra
            print(
                f"Batch Optimization failed for iteration {i}, starting individual optimization"
            )
            single_gnn_sim = get_gnn_sim(
                mol=mol,
                solvent=solvent,
                model_path=model_path,
                solvent_dict=solvent_dict,
                cache=cache,
                save_name=save_name,
                partial_charges=partial_charges,
                forcefield=forcefield,
                constraints=constraints,
                num_confs=1,
            )
            individual_positions = []
            for j in range(num_iters * num_confs, num_rep):
                single_gnn_sim = set_positions_for_simulation(
                    single_gnn_sim, mol, num_confs=1, iteration=0, offset=j
                )
                single_gnn_sim, status = run_minimisation(
                    single_gnn_sim, tolerance=tolerance, max_iterations=max_iterations
                )
                if status == 0:
                    individual_positions.append(get_minimised_positions(single_gnn_sim))
                else:
                    print(f"Individual optimization failed for confomer {j}")
                    individual_positions.append(np.zeros((mol.GetNumAtoms(), 3)))
            positions.append(np.concatenate(individual_positions, axis=0))

            del single_gnn_sim

    gnn_traj = get_traj_from_positions(
        mol, positions, num_iters=num_iters, num_confs=num_confs
    )
    gnn_energies = get_energies(gnn_sim, gnn_traj)
    reshaped_positions = np.concatenate(positions, axis=0)
    mol = set_positions_in_mol(mol, [reshaped_positions])

    if return_traj and return_gnn_sim:
        return mol, gnn_traj, gnn_energies, gnn_sim
    elif return_traj:
        return mol, gnn_traj, gnn_energies
    else:
        return mol, gnn_energies


def get_gnn_sim(
    mol,
    solvent,
    model_path,
    solvent_dict,
    cache=tempfile.mkdtemp() + "/tmp.cache",
    save_name=None,
    partial_charges=None,
    forcefield="openff-2.0.0",
    constraints=HBonds,
    num_confs=1,
):
    run_name = "testminimise"
    work_dir = tempfile.mkdtemp() + "/"

    smiles = Chem.MolToSmiles(
        mol,
        canonical=False,
        isomericSmiles=True,
        doRandom=False,
        kekuleSmiles=False,
    )

    if solvent == "vac":
        gnn_sim = create_vac_sim(
            smiles,
            cache,
            num_confs,
            save_name,
            mol=mol,
            partial_charges=partial_charges,
            forcefield=forcefield,
            constraints=constraints,
        )
    else:
        model_dict = torch.load(model_path)["model"]
        gnn_sim = create_gnn_sim(
            smiles,
            cache=cache,
            num_confs=num_confs,
            workdir=work_dir,
            run_name=run_name,
            save_name=save_name,
            rdkit_mol=mol,
            solvent=solvent,
            solvent_dict=solvent_dict,
            model_dict=model_dict,
            partial_charges=partial_charges,
            forcefield=forcefield,
            constraints=constraints,
        )
    return gnn_sim


def minimize_mol_gbneck(
    mol,
    solvent,
    solvent_dict,
    return_traj=False,
    tolerance=1e-4,
    max_iterations=0,
    gnn_sim=None,
    return_gnn_sim=False,
    strides=1,
    cache=tempfile.mkdtemp() + "/tmp.cache",
    partial_charges=None,
    forcefield="openff-2.0.0",
    constraints=HBonds,
):
    dielectric = solvent_dict[solvent.split("_")[1]]["dielectric"]
    SA = "ACE" if ("sagb" in solvent) else None
    print("SA model: ", SA)

    smiles = Chem.MolToSmiles(
        mol,
        canonical=False,
        isomericSmiles=True,
        doRandom=False,
        kekuleSmiles=False,
    )

    gbneck_sim = create_gbneck_sim(
        smiles,
        cache=cache,
        dielectric=dielectric,
        SA=SA,
        partial_charges=partial_charges,
        forcefield=forcefield,
        constraints=constraints,
        rdkit_mol=mol,
    )
    gbneck_sim.platform = "GPU"

    energies = []
    positions = []
    for i in range(mol.GetNumConformers()):
        set_positions_for_simulation(gbneck_sim, mol, num_confs=1, iteration=i)
        gbneck_sim._simulation.minimizeEnergy(
            tolerance=tolerance, maxIterations=max_iterations
        )
        positions.append(get_minimised_positions(gbneck_sim))
        energies.append(gbneck_sim.calculate_energy()._value)

    if return_traj:
        reshaped_positions = np.concatenate(positions, axis=0)
        mol = set_positions_in_mol(mol, [reshaped_positions])
        traj = mol_to_traj(mol)
        if return_traj and return_gnn_sim:
            return mol, traj, np.array(energies), gbneck_sim
        elif return_traj:
            return mol, traj, np.array(energies)
    else:
        return np.array(energies)


def calculate_gnn_energies(
    mol,
    solvent,
    model_path,
    solvent_dict=SOLVENT_DICT,
    return_traj=False,
    tolerance=1e-4,
    max_iterations=0,
    gnn_sim=None,
    return_gnn_sim=False,
    strides=1,
    cache=tempfile.mkdtemp() + "/tmp.cache",
    save_name=None,
    partial_charges=None,
    forcefield="openff-2.0.0",
    constraints=HBonds,
):

    num_rep = mol.GetNumConformers()
    num_iters = strides
    num_confs = num_rep
    num_confs = num_rep // strides

    if gnn_sim is None:
        solvent_model = []
        solvent_dielectric = []

        if not solvent == "vac":
            for repetition in range(num_rep):
                solvent_model.append(solvent_dict[solvent]["solvent_id"])
                solvent_dielectric.append(solvent_dict[solvent]["dielectric"])

        run_name = "testminimise"
        work_dir = tempfile.mkdtemp() + "/"  # directory of the repository
        n_interval = 100  # Interval for saving frames in steps

        smiles = Chem.MolToSmiles(
            mol,
            canonical=False,
            isomericSmiles=True,
            doRandom=False,
            kekuleSmiles=False,
        )
        if solvent == "vac":
            gnn_sim = create_vac_sim(
                smiles,
                cache=tempfile.mkdtemp() + "/tmp.cache",
                num_confs=num_confs,
                mol=mol,
            )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_dict = torch.load(model_path, map_location=device)["model"]
            gnn_sim = create_gnn_sim(
                smiles,
                cache=cache,
                num_confs=num_confs,
                workdir=work_dir,
                run_name=run_name,
                save_name=save_name,
                rdkit_mol=mol,
                solvent=solvent,
                solvent_dict=solvent_dict,
                model_dict=model_dict,
                partial_charges=partial_charges,
                forcefield=forcefield,
                constraints=constraints,
            )

        gnn_sim = set_positions_for_simulation(
            gnn_sim, mol, num_confs=num_confs, iteration=0
        )
    else:
        gnn_sim = set_positions_for_simulation(
            gnn_sim, mol, num_confs=num_confs, iteration=0
        )

    positions = []
    positions.append(get_minimised_positions(gnn_sim))

    for i in tqdm(range(1, num_iters)):
        gnn_sim = set_positions_for_simulation(
            gnn_sim, mol, num_confs=num_confs, iteration=i
        )
        positions.append(get_minimised_positions(gnn_sim))

    gnn_traj = get_traj_from_positions(
        mol, positions, num_iters=num_iters, num_confs=num_confs
    )[: num_iters * num_confs]
    gnn_energies = get_energies(gnn_sim, gnn_traj)

    return gnn_energies


def calculate_gnn_delta_energy(
    mol,
    solvent,
    model_path,
    solvent_dict,
    return_traj=False,
    tolerance=1e-4,
    max_iterations=0,
    gnn_sim=None,
    return_gnn_sim=False,
    strides=1,
):

    num_rep = mol.GetNumConformers()
    num_iters = strides
    num_confs = num_rep
    num_confs = num_rep // strides

    if gnn_sim is None:
        solvent_model = []
        solvent_dielectric = []

        if not solvent == "vac":
            for repetition in range(num_rep):
                solvent_model.append(solvent_dict[solvent]["solvent_id"])
                solvent_dielectric.append(solvent_dict[solvent]["dielectric"])

            class GNN3_multisolvent_run_multiple_e(
                GNN3_Multisolvent_embedding_run_multiple_Delta
            ):

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
                    return super().set_num_reps(
                        num_reps, solvent_models, solvent_dielectric
                    )

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
                "trained_model": model_path,
                "model": GNN3_multisolvent_e,
                "run_model": GNN3_multisolvent_run_multiple_e,
            }

        run_name = "testminimise"
        work_dir = tempfile.mkdtemp() + "/"  # directory of the repository
        n_interval = 100  # Interval for saving frames in steps

        smiles = Chem.MolToSmiles(
            mol,
            canonical=False,
            isomericSmiles=True,
            doRandom=False,
            kekuleSmiles=False,
        )
        if solvent == "vac":
            gnn_sim = create_vac_sim(
                smiles,
                cache=tempfile.mkdtemp() + "/tmp.cache",
                num_confs=num_confs,
                mol=mol,
            )
        else:
            gnn_sim = create_gnn_sim(
                smiles,
                cache=tempfile.mkdtemp() + "/tmp.cache",
                num_confs=num_confs,
                setup_dict=setup_dict_multisolv,
                additional_parameters={"solvent_model": -1, "solvent_dielectric": 78.5},
                workdir=work_dir,
                run_name=run_name,
            )
        gnn_sim = set_positions_for_simulation(
            gnn_sim, mol, num_confs=num_confs, iteration=0
        )
    else:
        gnn_sim = set_positions_for_simulation(
            gnn_sim, mol, num_confs=num_confs, iteration=0
        )

    positions = []
    positions.append(get_minimised_positions(gnn_sim))

    for i in tqdm(range(1, num_iters)):
        gnn_sim = set_positions_for_simulation(
            gnn_sim, mol, num_confs=num_confs, iteration=i
        )
        positions.append(get_minimised_positions(gnn_sim))

    gnn_traj = get_traj_from_positions(
        mol, positions, num_iters=num_iters, num_confs=num_confs
    )[: num_iters * num_confs]
    gnn_energies = get_energies(gnn_sim, gnn_traj)

    return gnn_energies


def calculate_vac_energy(mol, save_name=None):

    smiles = Chem.MolToSmiles(
        mol,
        canonical=False,
        isomericSmiles=True,
        doRandom=False,
        kekuleSmiles=False,
    )

    gbneck_sim = create_vac_sim(
        smiles,
        cache=tempfile.mkdtemp() + "/tmp.cache",
        num_confs=1,
        save_name=save_name,
        mol=mol,
    )
    energies = []
    for i in range(mol.GetNumConformers()):
        set_positions_for_simulation(gbneck_sim, mol, num_confs=1, iteration=i)
        energies.append(gbneck_sim.calculate_energy()._value)
    return np.array(energies)


def calculate_gbneck_energy(mol, solvent, solvent_dict=SOLVENT_DICT, SA=None):

    dielectric = solvent_dict[solvent]["dielectric"]

    smiles = Chem.MolToSmiles(
        mol,
        canonical=False,
        isomericSmiles=True,
        doRandom=False,
        kekuleSmiles=False,
    )

    gbneck_sim = create_gbneck_sim(
        smiles, cache=tempfile.mkdtemp() + "/tmp.cache", dielectric=dielectric, SA=SA
    )
    energies = []
    for i in range(mol.GetNumConformers()):
        set_positions_for_simulation(gbneck_sim, mol, num_confs=1, iteration=i)
        energies.append(gbneck_sim.calculate_energy()._value)
    return np.array(energies)


def build_mass_weighted_hessian(openmm_simulation, delta=0.0001):
    """
    Adapted from https://leeping.github.io/forcebalance/doc/html/api/openmmio_8py_source.html

    OpenMM single frame hessian evaluation
    Since OpenMM doesnot provide a Hessian evaluation method, we used finite difference on forces

    Parameters
    ----------
    simulation: OpenMM simulation object
        Simulation object with the context initialized and the minimized positions set

    Returns
    -------
    hessian: np.array with shape 3N x 3N, N = number of "real" atoms
        The result hessian matrix.
        The row indices are fx0, fy0, fz0, fx1, fy1, ...
        The column indices are x0, y0, z0, x1, y1, ..
        The unit is kilojoule / (nanometer^2 * mole * dalton) => 10^24 s^-2
    """

    context = openmm_simulation.context
    pos = context.getState(getPositions=True).getPositions(asNumpy=True)
    # pull real atoms and their mass
    massList = np.array(
        [
            atom.element.mass.value_in_unit(dalton)
            for atom in openmm_simulation.topology.atoms()
        ]
    )
    realAtomIdxs = [atom.index for atom in openmm_simulation.topology.atoms()]

    # initialize an empty hessian matrix
    noa = len(realAtomIdxs)
    hessian = np.empty((noa * 3, noa * 3), dtype=float)
    # finite difference step size
    diff = Quantity(delta, unit=nanometer)
    coef = 1.0 / (delta * 2)  # 1/2h
    for i, i_atom in enumerate(realAtomIdxs):
        massWeight = 1.0 / np.sqrt(massList * massList[i])
        # loop over the x, y, z coordinates
        for j in range(3):
            # plus perturbation
            pos[i_atom][j] += diff
            context.setPositions(pos)
            grad_plus = (
                context.getState(getForces=True)
                .getForces(asNumpy=True)
                .value_in_unit(kilojoule / (nanometer * mole))
            )
            grad_plus = -grad_plus[realAtomIdxs]  # gradients are negative forces
            # minus perturbation
            pos[i_atom][j] -= 2 * diff
            context.setPositions(pos)
            grad_minus = (
                context.getState(getForces=True)
                .getForces(asNumpy=True)
                .value_in_unit(kilojoule / (nanometer * mole))
            )
            grad_minus = -grad_minus[realAtomIdxs]  # gradients are negative forces
            # set the perturbation back to zero
            pos[i_atom][j] += diff
            # fill one row of the hessian matrix
            hessian[i * 3 + j] = np.ravel(
                (grad_plus - grad_minus) * coef * massWeight[:, np.newaxis]
            )
    # make hessian symmetric by averaging upper right and lower left
    hessian += hessian.T
    hessian *= 0.5
    # recover the original position
    context.setPositions(pos)
    return hessian


def normal_modes(openmm_simulation, delta=0.0001):
    """
    Adapted from https://leeping.github.io/forcebalance/doc/html/api/openmmio_8py_source.html

    OpenMM Normal Mode Analysis
    Since OpenMM doesnot provide a Hessian evaluation method, we used finite difference on forces

    Parameters
    ----------
    shot: int
        The frame number in the trajectory of this target
    optimize: bool, default True
        Optimize the geometry before evaluating the normal modes

    Returns
    -------
    freqs: np.array with shape (3N - 6) x 1, N = number of "real" atoms
        Harmonic frequencies, sorted from smallest to largest, with the 6 smallest removed, in unit cm^-1
    normal_modes: np.array with shape (3N - 6) x (3N), N = number of "real" atoms
        The normal modes corresponding to each of the frequencies, scaled by mass^-1/2.
    """

    # step 1: build a full hessian matrix
    hessian_matrix = build_mass_weighted_hessian(openmm_simulation, delta=delta)
    realAtomIdxs = [atom.index for atom in openmm_simulation.topology.atoms()]
    noa = len(realAtomIdxs)
    # step 2: diagonalize the hessian matrix
    eigvals, eigvecs = np.linalg.eigh(hessian_matrix)
    # step 3: convert eigenvalues to frequencies
    coef = 0.5 / np.pi * 33.3564095  # 10^12 Hz => cm-1
    negatives = (eigvals >= 0).astype(int) * 2 - 1  # record the negative ones
    freqs = np.sqrt(eigvals + 0j) * coef * negatives
    # step 4: convert eigenvectors to normal modes
    # re-arange to row index and shape
    normal_modes = eigvecs.T.reshape(noa * 3, noa, 3)
    # step 5: Remove mass weighting from eigenvectors
    massList = np.array(
        [
            atom.element.mass.value_in_unit(dalton)
            for atom in openmm_simulation.topology.atoms()
        ]
    )  # unit in dalton
    for i in range(normal_modes.shape[0]):
        mode = normal_modes[i]
        mode /= np.sqrt(massList[:, np.newaxis])
        mode /= np.linalg.norm(mode)
    # step 5: remove the 6 freqs with smallest abs value and corresponding normal modes
    n_remove = 5 if len(realAtomIdxs) == 2 else 6
    larger_freq_idxs = np.sort(np.argpartition(np.abs(freqs), n_remove)[n_remove:])
    # larger_freq_idxs = np.sort(np.argpartition(np.abs(freqs), n_remove))[n_remove:]
    freqs = freqs[larger_freq_idxs]
    normal_modes = normal_modes[larger_freq_idxs]
    return freqs, normal_modes


def calculate_vibrational_entropy(openmm_simulation, delta=0.0001):

    frequencies, _ = normal_modes(openmm_simulation)
    Sv = calculate_entropy_from_frequencies(frequencies)

    return Sv


def calculate_entropy(
    mol,
    solvent,
    model_path=MODEL_PATH,
    solvent_dict=SOLVENT_DICT,
    tolerance=1e-4,
    max_iterations=0,
    gnn_sim=None,
    strides=1,
    cache=tempfile.mkdtemp() + "/tmp.cache",
    save_name=None,
    partial_charges=None,
    forcefield="openff-2.0.0",
    return_free_energy=True,
    return_traj=False,
):

    mol, gnn_traj, gnn_energies, gnn_sim = minimize_mol(
        mol,
        solvent,
        model_path,
        solvent_dict,
        return_traj=True,
        tolerance=tolerance,
        max_iterations=max_iterations,
        gnn_sim=gnn_sim,
        return_gnn_sim=True,
        strides=strides,
        cache=cache,
        partial_charges=partial_charges,
        save_name=save_name,
        forcefield=forcefield,
        constraints=None,
    )

    entropies = []

    ref_sim = gnn_sim._ref_system
    for i in range(mol.GetNumConformers()):
        ref_sim.set_positions(gnn_traj.xyz[i])
        try:
            S = calculate_entropy_from_simulation(ref_sim._simulation)
        except:
            warnings.warn(f"Entropy calculation failed at conformer {i}")
            S = 0.0
        entropies.append(S)

    entropies = np.array(entropies)

    if return_free_energy:
        if return_traj:
            return gnn_traj, gnn_energies - entropies
        return entropies, gnn_energies - entropies
    else:
        return entropies


def calculate_entropy_from_simulation(
    openmm_simulation, delta=0.0001, use_minimisation_cycles=True
):
    """Calculate the entropy of a molecule using the vibrational and rotational entropy contributions.
    Args:
        openmm_simulation (openmm.Simulation): OpenMM simulation object with the context initialized and the minimized positions set
        delta (float, optional): Finite difference step size for numerical differentiation. Defaults to 0.0001.
        use_minimisation_cycles (bool, optional): Whether to perform multiple minimisation cycles as long as some frequencies are imaginary. Defaults to True.
    Returns:
        float: Total entropy in kJ/(mol·K)
    """

    frequencies, _ = normal_modes(openmm_simulation, delta=delta)
    if frequencies[0].imag != 0:
        frequencies = minimization_cycles(openmm_simulation, delta=delta)

    Sv = calculate_entropy_from_frequencies(frequencies)
    Sr = calculate_rotational_entropy(openmm_simulation)

    return Sv + Sr


def calculate_entropy_from_frequencies(frequencies):
    """Convert vibrational frequencies to vibrational entropy.
        This function uses the quasi-RRHO approach from Grimme
        Grimme, S. (2012), Supramolecular Binding Thermodynamics by Dispersion-Corrected Density Functional Theory. Chem. Eur. J., 18: 9955-9964. https://doi.org/10.1002/chem.201200497
    Args:
        frequencies (float): frequencies in cm**-1
    """

    # use higher precision for calculations
    frequencies = np.array(frequencies, dtype=np.complex256)

    frequencies = frequencies * 2.99792458e10  # Convert to Hz
    temperature = np.array(300, dtype=np.float128)  # Temperature in Kelvin
    k_B = np.array(1.380649e-23, dtype=np.float128)  # Boltzmann constant in J/K
    h = np.array(6.62607015e-34, dtype=np.float128)  # Planck's constant in J*s
    T = temperature  # Temperature in Kelvin
    N_A = np.array(6.02214076e23, dtype=np.float128)  # Avogadro's number

    # Calculate vibration entropy with harmonic oscillator approximation
    sv = -np.log(1 - np.exp(-h * frequencies / (k_B * T))) + h * frequencies / (
        k_B * T
    ) / (np.exp(h * frequencies / (k_B * T)) - 1)
    sv = k_B * sv * T * N_A / 1000  # Convert to kJ/(mol·K)

    # Calculate vibration entropy with rigid rotor approximation
    Bav = 10e-44  # kg m^2

    mu = h / (8 * np.pi**2 * frequencies)
    mup = (mu * Bav) / (mu + Bav)

    sr = 0.5 + np.log(np.sqrt((8 * np.pi**3 * mup * k_B * T) / h**2))
    sr = sr * k_B * T * N_A / 1000

    a = 4
    w_0 = np.array(100 * 2.99792458e10, dtype=np.float128)  # Convert to Hz

    w = lambda x: 1 / (1 + (w_0 / x) ** a)

    S = w(frequencies) * sv + (1 - w(frequencies)) * sr
    S = np.sum(S).real

    return S


def calculate_rotational_entropy(openmm_simulation):
    """Calculate the rotational entropy of a molecule using the moment of inertia tensor from the OpenMM simulation object.

    Args:
        openmm_simulation (openmm.Simulation): OpenMM simulation object with the context initialized and the minimized positions set

    Returns:
        float: Rotational entropy in kJ/(mol·K)
    """

    context = openmm_simulation.context
    pos = (
        context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(meter)
    )
    # pull real atoms and their mass

    dalton_to_kg = np.array(1.66054e-27, dtype=np.float128)

    massList = np.array(
        [
            atom.element.mass.value_in_unit(dalton) * dalton_to_kg
            for atom in openmm_simulation.topology.atoms()
        ]
    )
    realAtomIdxs = [atom.index for atom in openmm_simulation.topology.atoms()]

    # Create Moment of inertia tensor

    # Initialize the moment of inertia tensor
    I = np.zeros((3, 3))

    # Calculate the center of mass
    com = np.zeros(3)
    total_mass = 0

    for i, i_atom in enumerate(realAtomIdxs):
        maspos = massList[i] * pos[i_atom]
        com += maspos
        total_mass += massList[i]
    com /= total_mass

    # Calculate the moment of inertia tensor
    for i, i_atom in enumerate(realAtomIdxs):

        # Calculate the position of the atom relative to the center of mass
        r = pos[i_atom] - com

        # Calculate the moment of inertia tensor
        I[0, 0] += massList[i] * (r[1] ** 2 + r[2] ** 2)
        I[1, 1] += massList[i] * (r[0] ** 2 + r[2] ** 2)
        I[2, 2] += massList[i] * (r[0] ** 2 + r[1] ** 2)
        I[0, 1] -= massList[i] * r[0] * r[1]
        I[1, 2] -= massList[i] * r[1] * r[2]
        I[0, 2] -= massList[i] * r[0] * r[2]

    I[1, 0] = I[0, 1]
    I[2, 1] = I[1, 2]
    I[2, 0] = I[0, 2]

    # Calculate the eigenvalues of the moment of inertia tensor
    eigvals = np.linalg.eigvals(I)

    # Constants
    k_B = np.array(1.380649e-23, dtype=np.float128)  # Boltzmann constant in J/K
    h = np.array(6.62607015e-34, dtype=np.float128)  # Planck's constant in J*s
    N_A = np.array(6.02214076e23, dtype=np.float128)  # Avogadro's number
    T = np.array(300, dtype=np.float128)  # Temperature in Kelvin
    rho = 1

    # Calculate the rotational entropy
    theta = h**2 / (8 * np.pi**2 * eigvals * k_B)
    Zrot = np.sqrt(np.pi) / rho * np.sqrt(T**3 / (theta[0] * theta[1] * theta[2]))

    Srot = (
        k_B * T * (1.5 + np.log(Zrot)) * N_A / 1000
    )  # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Statistical_Thermodynamics_(Jeschke)/04%3A_Entropy/4.02%3A_The_Relation_of_State_Functions_to_the_Partition_Function

    return Srot


def calculate_translational_entropy(openmm_simulation):
    """Calculate the translational entropy of a molecule using the moment of inertia tensor from the OpenMM simulation object."""

    dalton_to_kg = np.array(1.66054e-27, dtype=np.float128)
    massList = np.array(
        [
            atom.element.mass.value_in_unit(dalton) * dalton_to_kg
            for atom in openmm_simulation.topology.atoms()
        ]
    )

    total_mass = np.sum(massList)
    V = 1.0
    translational_entropy = (
        2 * np.pi * total_mass * 1.380649e-23 * 300 / (6.62607015e-34) ** 2
    ) ** (3 / 2) * V

    return translational_entropy


def generate_spherical_points(center, radius, num_points):
    """
    Generate points around a given center in a spherical distribution with a specified radius.

    Parameters:
    center (tuple): The center of the sphere (x, y, z).
    radius (float): The radius of the sphere.
    num_points (int): The number of points to generate.

    Returns:
    np.ndarray: Array of points in Cartesian coordinates.
    """
    points = []
    indices = np.arange(0, num_points, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    points = np.vstack((x, y, z)).T
    return points


def calculate_translational_entropy(openmm_simulation):
    """Calculate the translational entropy of a molecule using the moment of inertia tensor from the OpenMM simulation object."""

    pos = (
        openmm_simulation.context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(meter)
    )

    # add spheres aroung each point
    for i in range(pos.shape[0]):
        pos = np.vstack([pos, generate_spherical_points(pos[i], 1.25e-10, 100)])

    # Calculate the molecular volume
    hull = ConvexHull(pos)
    V = np.array(hull.volume, dtype=np.float128)

    dalton_to_kg = np.array(1.66054e-27, dtype=np.float128)
    massList = np.array(
        [
            atom.element.mass.value_in_unit(dalton) * dalton_to_kg
            for atom in openmm_simulation.topology.atoms()
        ]
    )

    k_B = np.array(1.380649e-23, dtype=np.float128)  # Boltzmann constant in J/K
    h = np.array(6.62607015e-34, dtype=np.float128)  # Planck's constant in J*s
    N_A = np.array(6.02214076e23, dtype=np.float128)  # Avogadro's number
    T = np.array(300, dtype=np.float128)  # Temperature in Kelvin
    rho = 1

    total_mass = np.sum(massList)
    translational_zustandssumme = (2 * np.pi * total_mass * k_B * T / h**2) ** (
        3 / 2
    ) * V

    translational_entropy = (
        k_B
        * T
        * N_A
        / 1000
        * (np.log((2 * np.pi * total_mass * k_B * T / h**2) ** (3 / 2) * V) + 2.5)
    )

    return translational_entropy


def minimization_cycles(simulation, delta=0.0001):

    pre_min_forces = (
        simulation.context.getState(getForces=True)
        .getForces(asNumpy=True)
        .value_in_unit(kilojoule_per_mole / nanometer)
    )
    old_mean_force = np.mean(np.linalg.norm(pre_min_forces, axis=1))
    min_thresh = 0.1

    for i in tqdm(range(100)):
        simulation.minimizeEnergy(tolerance=1e-5)
        forces = (
            simulation.context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(kilojoule_per_mole / nanometer)
        )
        mean_force = np.mean(np.linalg.norm(forces, axis=1))

        # only calculate normal modes if the mean force changes drastically
        if np.mean(np.abs(mean_force - old_mean_force)) < min_thresh:
            continue
        frequencies, _ = normal_modes(simulation, delta=delta)
        old_mean_force = mean_force

        if frequencies[0].imag == 0:
            return frequencies

    warnings.warn("No real frequencies found after 100 minimisation cycles.")
    return frequencies


class GNN3_scale_64_run_CHCl3(GNN3_scale_64_run):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=4.81,
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
            solvent_dielectric,
        )


class GNN3_scale_64_run_DMSO(GNN3_scale_64_run):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=46.7,
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
            solvent_dielectric,
        )


class GNN3_scale_64_run_CO(GNN3_scale_64_run):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=32.7,
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
            solvent_dielectric,
        )


setup_dict_tip3p = {
    "trained_model": "/fileserver/pine/pine8/kpaul/small_molecule_pub/MachineLearning/trained_models/GNN3_pub__batchsize_32_per_0.95_fra_0.1_random_3_radius_0.6_lr_0.0005_epochs_30_modelid_64_name__clip_1.0model.model",
    "model": GNN3_scale_64,
    "run_model": GNN3_scale_64_run,
}

setup_dict_CHCl3 = {
    "trained_model": "../MachineLearning/trained_models/GNN3_pub__batchsize_32_per_0.95_fra_0.1_random_1_radius_0.6_lr_0.0005_epochs_5_modelid_64_name__clip_1.0_solvent_dielectric_4.81model.model",
    "model": GNN3_scale_64,
    "run_model": GNN3_scale_64_run_CHCl3,
}


class GNN3_scale_64_run_CHCl3_sep_energy(GNN3_scale_64_run):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=4.81,
        print_separate_energies=True,
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
            solvent_dielectric,
            print_separate_energies=print_separate_energies,
        )


setup_dict_CHCl3_sep_energy = {
    "trained_model": "../MachineLearning/trained_models/GNN3_pub__batchsize_32_per_0.95_fra_0.1_random_1_radius_0.6_lr_0.0005_epochs_5_modelid_64_name__clip_1.0_solvent_dielectric_4.81model.model",
    "model": GNN3_scale_64,
    "run_model": GNN3_scale_64_run_CHCl3_sep_energy,
}


class GNN3_scale_64_run_CHCl3_flex_sasa(GNN3_scale_64_SA_flex_run):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=100,
        gbneck_radius=10,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=4.81,
        print_separate_energies=False,
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
            solvent_dielectric,
            print_separate_energies,
        )


setup_dict_CHCl3_flex_sasa = {
    "trained_model": "/home/kpaul/small_molecule_multisolvent/MachineLearning/trained_models/GNN3_pub__batchsize_64_per_0.95_fra_0.1_random_2_radius_0.6_lr_0.001_epochs_30_modelid_0_name_chloroform_flex_clip_1.0_solvent_dielectric_4.81_verbose_False_limit_0_database_solvent_tip3pmodel.model",
    "model": GNN3_scale_64_SA_flex,
    "run_model": GNN3_scale_64_run_CHCl3_flex_sasa,
}


class GNN3_scale_64_run_tip3p_flex_sasa(GNN3_scale_64_SA_flex_run):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=100,
        gbneck_radius=10,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
        print_separate_energies=False,
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
            solvent_dielectric,
            print_separate_energies,
        )


setup_dict_tip3p_flex_sasa = {
    "trained_model": "/home/kpaul/small_molecule_multisolvent/MachineLearning/trained_models/GNN3_pub__batchsize_64_per_0.95_fra_0.1_random_3_radius_0.6_lr_0.001_epochs_30_modelid_0_name_methanol_flex_clip_1.0_solvent_dielectric_78.5_verbose_False_limit_0_database_solvent_tip3pmodel.model",
    "model": GNN3_scale_64_SA_flex,
    "run_model": GNN3_scale_64_run_tip3p_flex_sasa,
}

setup_dict_DMSO = {
    "trained_model": "../MachineLearning/trained_models/GNN3_pub__batchsize_32_per_0.95_fra_0.1_random_3_radius_0.6_lr_0.0005_epochs_30_modelid_64_name__clip_1.0_solvent_dielectric_46.7_verbose_Falsemodel.model",
    "model": GNN3_scale_64,
    "run_model": GNN3_scale_64_run_DMSO,
}

setup_dict_CO = {
    "trained_model": "../MachineLearning/trained_models/GNN3_pub__batchsize_32_per_0.95_fra_0.1_random_1_radius_0.6_lr_0.0005_epochs_30_modelid_64_name__clip_1.0_solvent_dielectric_32.7_verbose_Falsemodel.model",
    "model": GNN3_scale_64,
    "run_model": GNN3_scale_64_run_CO,
}
