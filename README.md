# GNNImplicitSolvent

## Publications

[2] TBD, [https://chemrxiv.org](https://chemrxiv.org)

[1] J. Chem. Phys. 158, 204101 (2023), DOI: [https://doi.org/10.1063/5.0147027](https://doi.org/10.1063/5.0147027)

## Abstract
[2] The dynamical behavior of small molecules in a given environment can be studied with molecular dynamics (MD) simulations to assist the interpretation of experimental findings, e.g. in the development of new therapeutics. The description of accurate solvation effects for these simulations is thereby key and has been an active field of research since the introduction of MD. Until today the most accurate approach involves computationally expensive explicit solvent simulations. Recently, machine learning (ML) approaches have been proposed as an alternative by providing an accurate probabilistic representation of solvation effects. However, the associated computational costs and low-to-no transferability used to render them unusable in practice. Here, we report the first instance of a transferable ML-based implicit solvent model trained on a diverse set of 3'000'000 molecular structures that can be applied to organic small molecules for simulations in water. Extensive testing against reference calculations demonstrates that the model is capable of providing on par accuracy with explicit solvent simulations while providing an 18-fold increase in sampling rate.

[1] Molecular dynamics (MD) simulations enable the study of the motion of small and large (bio)molecules and the estimation of their conformational ensembles. The description of the environment (solvent) has thereby a large impact. Implicit solvent representations are efficient but in many cases not accurate enough (especially for polar solvents such as water). More accurate but also computationally more expensive is the explicit treatment of the solvent molecules. Recently, machine learning (ML) has been proposed to bridge the gap and simulate in an implicit manner explicit solvation effects. However, the current approaches rely on prior knowledge of the entire conformational space, limiting their application in practice. Here, we introduce a graph neural network (GNN) based implicit solvent that is capable of describing explicit solvent effects for peptides with different composition than contained in the training set. 


## Data

[2] Data is available at doi: [10.3929/ethz-b-000667722](https://doi.org/10.3929/ethz-b-000667722) 

[1] Data is available at doi: [10.3929/ethz-b-000599309](https://doi.org/10.3929/ethz-b-000599309) 

## Installation

For the exact environment composition used in publication [2] the [environment.yml](environment.yml) is provided. It can be installed with conda or mamba.

```bash
mamba env create -n GNNimplicit -f environment.yml
```

In case this fails (e.g., if a different operating system is used) or other packages are required the authors recommend to install pytorch and pytorch geometric first following the installation recomondations of [pytorch](https://pytorch.org/get-started/locally/) and [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). In addition the libraries torch-cluster, torch-sparse, and torch-scatter should be installed (pip is recommended). 

Additional packagaes listed in the [environment.yml](environment.yml) can be installed using conda or mamba.

## USAGE

### Simulations

To run an implicit solvent simulation using the GNN as a solvent the Multi_simulator class can be used. An example is shown in file [run_gnn_simulation.py](Simulation/run_gnn_simulation.py). The necessary input is only the molecules SMILES (-s), the desired simulation time in ns (-ns), the number of parallel simulations (-nr), the random seed of the run (-r), and the path to the trained GNN (-m):

```bash
python run_gnn_simulation.py -s CCCO -ns 0.1 -nr 20 -r 1234 -m ../MachineLearning/trained_models/GNN.model
```

One pretrained GNN is provided in this repository [GNN.model](MachineLearning/trained_models/GNN.model).

To run simulations using explicit solvent or GBNeck2 simulations as performed in publication [2] the [run_simulation_for_small_molecules.py](Simulation/run_simulation_for_small_molecules.py) script can be used following the instructions as described in the file.

```bash
python run_simulation_for_small_molecules.py --help
```

### Force Extraction

The complete dataset used in study [2] can be downloaded from the ETH Research Collection. In order to extract forces for new compounds or conformations the [run_data_generation.py](Simulation/run_data_generation.py) script can be used. By default the script stores all results in an hdf5 file. For easy access of a small number of results the user can also run the script with the ```--return_numpy_files``` flag, which will print the location of the stored positions and forces as numpy arrays.

### Machine Learning

The training and test set used in this study [2] can be downloaded from the ETH Research Collection. A new model can be trained using the [train_model.py](MachineLearning/train_model.py) and tested using the [test_model.py](MachineLearning/test_model.py) scripts. To train a model with the same architecture as used in this study one could use the following command:

```bash
python train_model.py -b 32 -p 0.95 -f 0.1 -r 161311 -ra 0.6 -l 0.0005 -fpt datasets/train_set.pt -e 30 -m 64 -c 1.0
```

### Analysis

Two examples for the analysis of the provided model and data are provided.
For analysing the performance of a trained GNN the [ForcePrediction.ipynb](Analysis/ForcePrediction.ipynb) notebook can be used. For analysing trajectories the [AnalysisExample.ipynb](Analysis/AnalysisExample.ipynb) can be used.