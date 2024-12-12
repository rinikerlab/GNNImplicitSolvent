# GNNImplicitSolvent

## Publications

[3] ChemRxiv. 2024; DOI: [https://doi.org/10.26434/chemrxiv-2024-1hb0b](https://doi.org/10.26434/chemrxiv-2024-1hb0b)

[2] Chem. Sci., 2024, DOI: [https://doi.org/10.1039/D4SC02432J](https://doi.org/10.1039/D4SC02432J)

[1] J. Chem. Phys. 158, 204101 (2023), DOI: [https://doi.org/10.1063/5.0147027](https://doi.org/10.1063/5.0147027)

## Abstract
[3] Understanding and manipulating the conformational behavior of a molecule in different solvent environments is of great interest in the fields of drug discovery and organic synthesis. Molecular dynamics (MD) simulations with solvent molecules explicitly present are the gold standard to compute such conformational ensembles (within the accuracy of the underlying force field), complementing experimental findings and supporting their interpretation. However, conventional methods often face challenges related to computational cost (explicit solvent) or accuracy (implicit solvent). Here, we showcase how our graph neural network (GNN)-based implicit solvent (GNNIS) approach can be used to rapidly compute small molecule conformational ensembles in 39 common organic solvents with high accuracy compared to explicit-solvent simulations. We validate this approach using nuclear magnetic resonance (NMR) measurements, thus identifying the conformers contributing most to the experimental observable. The method allows the time required to accurately predict conformational ensembles to be reduced from days to minutes while achieving results within one kBT of the experimental values.

[2] The dynamical behavior of small molecules in their environment can be studied with classical molecular dynamics (MD) simulations to gain deeper insight on an atomic level and thus complement and rationalize the interpretation of experimental findings. Such approaches are of great value in various areas of research, e.g., in the development of new therapeutics. The accurate description of solvation effects in such simulations is thereby key and has in consequence been an active field of research since the introduction of MD. So far, the most accurate approaches involve computationally expensive explicit solvent simulations, while widely applied models using an implicit solvent description suffer from reduced accuracy. Recently, machine learning (ML) approaches that provide a probabilistic representation of solvation effects have been proposed as potential alternatives. However, the associated computational costs and minimal or lack of transferability render them unusable in practice. Here, we report the first example of a transferable ML-based implicit solvent model trained on a diverse set of 3 000 000 molecular structures that can be applied to organic small molecules for simulations in water. Extensive testing against reference calculations demonstrated that the model delivers on par accuracy with explicit solvent simulations while providing an up to 18-fold increase in sampling rate.

[1] Molecular dynamics (MD) simulations enable the study of the motion of small and large (bio)molecules and the estimation of their conformational ensembles. The description of the environment (solvent) has thereby a large impact. Implicit solvent representations are efficient but in many cases not accurate enough (especially for polar solvents such as water). More accurate but also computationally more expensive is the explicit treatment of the solvent molecules. Recently, machine learning (ML) has been proposed to bridge the gap and simulate in an implicit manner explicit solvation effects. However, the current approaches rely on prior knowledge of the entire conformational space, limiting their application in practice. Here, we introduce a graph neural network (GNN) based implicit solvent that is capable of describing explicit solvent effects for peptides with different composition than contained in the training set. 


## Data

[3] Data is available at doi: [10.3929/ethz-b-000710355](https://doi.org/10.3929/ethz-b-000710355) 

[2] Data is available at doi: [10.3929/ethz-b-000667722](https://doi.org/10.3929/ethz-b-000667722) 

[1] Data is available at doi: [10.3929/ethz-b-000599309](https://doi.org/10.3929/ethz-b-000599309) 

## Installation

First clone this repository to your work station and install the environmentusing conda or mamba:

```bash
mamba env create -f environment.yml
```

Activate the environment:

```bash
conda activate GNNImplicitSolvent
```

Install the package using the provided setup.py file:

```bash
pip install .
```

## USAGE

All data used in this study can be reproduced and the corresponding workflows are described in the following sections.

The key application of the package is the minimization of compounds. A minimal example is given below:

```python
from GNNImplicitSolvent import minimize_mol, calculate_entropy
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles('COCCO')
mol = Chem.AddHs(mol)
AllChem.EmbedMultipleConfs(mol, numConfs=128)

minimized_mol, energies = minimize_mol(mol,"DMSO")
entropies, free_energies = calculate_entropy(minimized_mol,"DMSO")
```

In addition an example workflow is provided in the [ExampleConformationEnsemble.ipynb](Analysis/ExampleConformationEnsemble.ipynb) notebook.

## Reproducibility

This section is intended to provide a step-by-step guide to reproduce the results of the paper.

### Data Set Generation

The training set was generated using the following procedure:

1. All smiles for which data was extracted for the implicit solvent water model (as stored in the lwreg database) were written into the canonical_smiles.npy file using the [create_smiles_data.ipynb](Simulation/create_smiles_data.ipynb) notebook.

2. The forces for all molecules were then extracted using the [run_training_set_generation.py](Simulation/run_training_set_generation.py) script which was submitted using the [submit_data_generation.sh](Simulation/submit_data_generation.sh) bash script [decommenting the lines for each solvent]. Note that not all simulations converged and only the molecules for which the forces could successfully be extracted were used for further processing.

3. The forces were than put into the database using the [run_deposit_database.py](Simulation/run_deposit_database.py) script which was submitted using the [submit_deposit.sh](Simulation/submit_deposit_database.sh) bash script.

4. All molecules for which forces were extracted but that were used for the external test set or prospective simulations were marked accordingly in the database using the [mark_test_set.ipynb](MachineLearning/mark_test_set.ipynb) notebook.

5. The test set was subsequently analyzed and steps 2-4 repeated until every solvent had at least one entry per molecule in the database. This procedure is performed using the [get_solvent_smiles_with_not_all_test.ipynb](MachineLearning/get_solvent_smiles_with_not_all_test.ipynb) jupyter notebook and the produces bash scripts.

### Machine Learning

The train and test dataset was collected using the [collect_complete_dataset.py](MachineLearning/collect_complete_dataset.py) script.

The training of the Machine Learning model is performed using the [train_model_multisolvent.py](MachineLearning/train_model_multisolvent.py) python script.
The training can be repeated by running the [submit_training.sh](MachineLearning/submit_training.sh) script which uses the [train_config.yml](MachineLearning/train_config.yml) configuration file.

### Prospective Simulations

The prospective simulations were performed using the [run_gnn_multisolvent_simulation.py](Simulation/run_gnn_multisolvent_simulation.py) and [run_simulation_for_small_molecules.py](Simulation/run_simulation_for_small_molecules.py)script for the GNN and explicit simulations, respectively. The were submitted using the [submit_production_simulations.sh](Simulation/submit_production_simulations.sh) bash script. Failed simulations were repeated using [submit_missing_simulations.sh](Simulation/submit_missing_simulations.sh). For the comparison GBNeck2 simulations the simulations were submitted using teh [submit_gbneck2_reference_simulations.sh](Simulation/submit_gbneck2_reference_simulations.sh) script. Again failed simulations were repeated using the [submit_missing_gbneck2_simulations.sh](Simulation/submit_missing_gbneck2_simulations.sh) script.


### Minimization

The minimizations of the two small compounds are performed directly in the analyis notebook [Jcoupling.ipynb](Analysis/Jcoupling.ipynb). The minimizations for the simulated ensembles and experimental ensembles were performed using the [run_conformational_ensemble.py](Simulation/run_conformational_ensemble.py), [run_platinum_analysis.py](Simulation/run_platinum_analysis.py), and the (run_MB_analysis.py)[Simulation/run_MB_analysis.py] scripts. The minimizations were submitted using the [submit_minimisations.sh](Simulation/submit_minimisations.sh) script.


#### Analysis

The analysis of the Machine Learning models is performed using the [test_model.ipynb](MachineLearning/test_model.ipynb) notebook.

The prospective simulations are analysed using the Jupyter Notebook [Intra_molecular_HBond.ipynb](Analysis/Intra_molecular_HBond.ipynb).

The Simulated Ensembles were analyzed using the [Analyse_Simulated_Ensembles.ipynb](Analysis/Analyse_Simulated_Ensembles.ipynb) notebook. The Experimental Ensembles were analyzed using the [Analyse_Experimental_Ensembles.ipynb](Analysis/Analyse_Experimental_Ensembles.ipynb) notebook and the [Jcoupling.ipynb](Analysis/Jcoupling.ipynb) notebook.