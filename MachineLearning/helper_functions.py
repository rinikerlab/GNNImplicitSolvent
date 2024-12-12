import torch
from rdkit import Chem
import numpy as np

# Hardcoded dictionary of solvents for fast execution on HPC
SOLVENT_DICT = {
    "tip3p": {
        "SMILES": "O",
        "MW": 18,
        "density": 1.0,
        "dielectric": 78.5,
        "solvent_id": 0,
        "nice_name": "TIP3P",
    },
    "Chloroform": {
        "SMILES": "ClC(Cl)Cl",
        "MW": 119.38,
        "density": 1.49,
        "dielectric": 4.81,
        "solvent_id": 1,
        "nice_name": "Chloroform",
    },
    "Methanol": {
        "SMILES": "CO",
        "MW": 32.04,
        "density": 0.792,
        "dielectric": 33.0,
        "solvent_id": 2,
        "nice_name": "Methanol",
    },
    "DMSO": {
        "SMILES": "CS(=O)C",
        "MW": 78.13,
        "density": 1.1,
        "dielectric": 47.24,
        "solvent_id": 3,
        "nice_name": "DMSO",
    },
    "DMPU": {
        "SMILES": "O=C1N(C)CCCN1C",
        "MW": 128.175,
        "density": 1.06,
        "dielectric": 36.12,
        "solvent_id": 4,
        "nice_name": "DMPU",
    },
    "Diethylether": {
        "SMILES": "CCOCC",
        "MW": 74.123,
        "density": 0.7134,
        "dielectric": 4.27,
        "solvent_id": 5,
        "nice_name": "Diethyl Ether",
    },
    "Ethanol": {
        "SMILES": "OCC",
        "MW": 46.069,
        "density": 0.78945,
        "dielectric": 25.3,
        "solvent_id": 6,
        "nice_name": "Ethanol",
    },
    "DMF": {
        "SMILES": "CN(C)C=O",
        "MW": 73.095,
        "density": 0.948,
        "dielectric": 38.25,
        "solvent_id": 7,
        "nice_name": "DMF",
    },
    "DCM": {
        "SMILES": "ClCCl",
        "MW": 84.93,
        "density": 1.3266,
        "dielectric": 8.93,
        "solvent_id": 8,
        "nice_name": "DCM",
    },
    "Toluol": {
        "SMILES": "Cc1ccccc1",
        "MW": 92.141,
        "density": 0.8623,
        "dielectric": 2.38,
        "solvent_id": 9,
        "nice_name": "Toluene",
    },
    "Benzol": {
        "SMILES": "c1ccccc1",
        "MW": 78.114,
        "density": 0.8765,
        "dielectric": 2.28,
        "solvent_id": 10,
        "nice_name": "Benzene",
    },
    "Hexan": {
        "SMILES": "CCCCCC",
        "MW": 86.178,
        "density": 0.6606,
        "dielectric": 1.88,
        "solvent_id": 11,
        "nice_name": "Hexane",
    },
    "acetonitrile": {
        "SMILES": "CC#N",
        "MW": 41.053,
        "density": 0.786,
        "dielectric": 36.64,
        "solvent_id": 12,
        "nice_name": "Acetonitrile",
    },
    "acetone": {
        "SMILES": "CC(=O)C",
        "MW": 58.08,
        "density": 0.7845,
        "dielectric": 21.01,
        "solvent_id": 13,
        "nice_name": "Acetone",
    },
    "aceticacid": {
        "SMILES": "CC(O)=O",
        "MW": 60.052,
        "density": 1.049,
        "dielectric": 6.2,
        "solvent_id": 14,
        "nice_name": "Acetic Acid",
    },
    "14dioxane": {
        "SMILES": "O1CCOCC1",
        "MW": 88.106,
        "density": 1.033,
        "dielectric": 2.22,
        "solvent_id": 15,
        "nice_name": "1,4-Dioxane",
    },
    "nitrobenzol": {
        "SMILES": "c1ccc(cc1)[N+](=O)[O-]",
        "MW": 123.11,
        "density": 1.199,
        "dielectric": 35.6,
        "solvent_id": 16,
        "nice_name": "Nitrobenzene",
    },
    "HMPA": {
        "SMILES": "O=P(N(C)C)(N(C)C)N(C)C",
        "MW": 179.2,
        "density": 1.03,
        "dielectric": 29.6,
        "solvent_id": 17,
        "nice_name": "HMPA",
    },
    "MTBE": {
        "SMILES": "O(C(C)(C)C)C",
        "MW": 88.15,
        "density": 0.7404,
        "dielectric": 4.5,
        "solvent_id": 18,
        "nice_name": "MTBE",
    },
    "IPA": {
        "SMILES": "CC(O)C",
        "MW": 60.096,
        "density": 0.786,
        "dielectric": 20.18,
        "solvent_id": 19,
        "nice_name": "IPA",
    },
    "Hexafluorobenzene": {
        "SMILES": "Fc1c(F)c(F)c(F)c(F)c1F",
        "MW": 186.056,
        "density": 2.03,
        "dielectric": 2.03,
        "solvent_id": 20,
        "nice_name": "Hexafluorobenzene",
    },
    "pyridine": {
        "SMILES": "c1ccncc1",
        "MW": 79.102,
        "density": 0.9819,
        "dielectric": 13.26,
        "solvent_id": 21,
        "nice_name": "Pyridine",
    },
    "THF": {
        "SMILES": "C1CCOC1",
        "MW": 72.107,
        "density": 0.8876,
        "dielectric": 7.52,
        "solvent_id": 22,
        "nice_name": "THF",
    },
    "Ethylacetate": {
        "SMILES": "O=C(OCC)C",
        "MW": 88.106,
        "density": 0.902,
        "dielectric": 6.2,
        "solvent_id": 23,
        "nice_name": "Ethyl Acetate",
    },
    "Sulfolane": {
        "SMILES": "C1CCS(=O)(=O)C1",
        "MW": 120.17,
        "density": 1.261,
        "dielectric": 43.3,
        "solvent_id": 24,
        "nice_name": "Sulfolane",
    },
    "nitromethane": {
        "SMILES": "C[N+](=O)[O-]",
        "MW": 61.04,
        "density": 1.1371,
        "dielectric": 37.27,
        "solvent_id": 25,
        "nice_name": "Nitromethane",
    },
    "Butylformate": {
        "SMILES": "CC(C)(C)OC=O",
        "MW": 102.133,
        "density": 0.872,
        "dielectric": 6.1,
        "solvent_id": 26,
        "nice_name": "Butyl Formate",
    },
    "NMP": {
        "SMILES": "CN1CCCC1=O",
        "MW": 99.133,
        "density": 1.028,
        "dielectric": 32.55,
        "solvent_id": 27,
        "nice_name": "NMP",
    },
    "Octanol": {
        "SMILES": "CCCCCCCCO",
        "MW": 130.231,
        "density": 0.83,
        "dielectric": 10.3,
        "solvent_id": 28,
        "nice_name": "Octanol",
    },
    "cyclohexane": {
        "SMILES": "C1CCCCC1",
        "MW": 84.162,
        "density": 0.7739,
        "dielectric": 2.024,
        "solvent_id": 29,
        "nice_name": "Cyclohexane",
    },
    "glycerin": {
        "SMILES": "OCC(O)CO",
        "MW": 92.094,
        "density": 1.261,
        "dielectric": 46.53,
        "solvent_id": 30,
        "nice_name": "Glycerin",
    },
    "carbontetrachloride": {
        "SMILES": "ClC(Cl)(Cl)Cl",
        "MW": 153.81,
        "density": 1.5867,
        "dielectric": 2.24,
        "solvent_id": 31,
        "nice_name": "Carbon Tetrachloride",
    },
    "DME": {
        "SMILES": "COCCOC",
        "MW": 90.122,
        "density": 0.8683,
        "dielectric": 7.3,
        "solvent_id": 32,
        "nice_name": "DME",
    },
    "2Nitropropane": {
        "SMILES": "CC(C)[N+](=O)[O-]",
        "MW": 89.094,
        "density": 0.9821,
        "dielectric": 26.74,
        "solvent_id": 33,
        "nice_name": "2-Nitropropane",
    },
    "Trifluorotoluene": {
        "SMILES": "C1=CC=C(C=C1)C(F)(F)F",
        "MW": 146.11,
        "density": 1.19,
        "dielectric": 9.22,
        "solvent_id": 34,
        "nice_name": "Trifluorotoluene",
    },
    "hexafluroacetone": {
        "SMILES": "FC(F)(F)C(=O)C(F)(F)F",
        "MW": 166.02,
        "density": 1.32,
        "dielectric": 2.104,
        "solvent_id": 35,
        "nice_name": "Hexafluoroacetone",
    },
    "Propionitrile": {
        "SMILES": "CCC#N",
        "MW": 55.08,
        "density": 0.772,
        "dielectric": 29.7,
        "solvent_id": 36,
        "nice_name": "Propionitrile",
    },
    "Benzonitrile": {
        "SMILES": "N#Cc1ccccc1",
        "MW": 103.12,
        "density": 1.0,
        "dielectric": 25.9,
        "solvent_id": 37,
        "nice_name": "Benzonitrile",
    },
    "oxylol": {
        "SMILES": "CC1=C(C)C=CC=C1",
        "MW": 106.168,
        "density": 0.88,
        "dielectric": 2.56,
        "solvent_id": 38,
        "nice_name": "Oxylol",
    },
}

# Hardcoded function from trainer for fast execution on HPC
from Simulation.Simulator import Simulator
from ForceField.Forcefield import OpenFF_forcefield_GBNeck2, GBSAGBn2Force
import tempfile

DEFAULT_UNIQUE_RADII = [0.14, 0.117, 0.155, 0.15, 0.21, 0.185, 0.18, 0.17, 0.12, 0.13]


def get_gbneck2_param_small_molecules(
    pid,
    work_dir,
    cache=None,
    rdkit_mol=None,
    uniqueRadii=None,
    partial_charges=None,
    forcefield="openff-2.0.0",
):

    sim = Simulator(
        work_dir=work_dir, pdb_id=pid, run_name="getparam", rdkit_mol=rdkit_mol
    )
    sim.forcefield = OpenFF_forcefield_GBNeck2(
        pid,
        cache=cache,
        rdkit_mol=rdkit_mol,
        partial_charges=partial_charges,
        forcefield=forcefield,
    )
    topology = sim._datahandler.topology
    charges = np.array(
        [
            sim._system.getForces()[0].getParticleParameters(i)[0]._value
            for i in range(topology._numAtoms)
        ]
    )
    force = GBSAGBn2Force(cutoff=None, SA=None, soluteDielectric=1)
    gbn2_parameters = np.empty((topology.getNumAtoms(), 7))
    gbn2_parameters[:, 0] = charges  # Charges
    gbn2_parameters[:, 1:6] = force.getStandardParameters(topology)
    radii = gbn2_parameters[:, 1]
    if uniqueRadii is None:
        uniqueRadii = list(sorted(set(radii)))
    radiusToIndex = {r: i for i, r in enumerate(uniqueRadii)}
    gbn2_parameters[:, 6] = [
        radiusToIndex[float("%.4f" % (r))] for r in gbn2_parameters[:, 1]
    ]
    offset = 0.0195141
    gbn2_parameters[:, 1] = gbn2_parameters[:, 1] - offset
    gbn2_parameters[:, 2] = gbn2_parameters[:, 2] * gbn2_parameters[:, 1]

    return gbn2_parameters, uniqueRadii


def create_gnn_model(
    mol,
    model_class=None,
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        parameters = get_gbneck2_param_small_molecules(
            pdb_id,
            tmp_dir + "/",
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
