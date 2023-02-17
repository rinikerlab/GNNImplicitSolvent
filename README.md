# implicitML


## Abstract
Molecular dynamics (MD) simulations enable the study of the motion of small and large (bio)molecules and the estimation of their conformational ensembles. The description of the environment (solvent) has thereby a large impact. Implicit solvent representations are efficient but in many cases not accurate enough (especially for polar solvents such as water). More accurate but also computationally more expensive is the explicit treatment of the solvent molecules. Recently, machine learning (ML) has been proposed to bridge the gap and simulate in an implicit manner explicit solvation effects. However, the current approaches rely on prior knowledge of the entire conformational space, limiting their application in practice. Here, we introduce a graph neural network (GNN) based implicit solvent that is capable of describing explicit solvent effects for peptides with different composition than contained in the training set. 

## Data

Data is available at doi: [10.3929/ethz-b-000599309](https://doi.org/10.3929/ethz-b-000599309) 

## USAGE

### Simulations

Simulations can be performed using the Simulator class defined in  [Simulator.py](Simulation/Simulator.py). An example how to use the class can be found in the [run_simulations.ipynb](Simulation/run_simulations.ipynb) Jupyter notebook.

### Force Extraction

For calculating explicit solvent forces the Explicit_water_simulator class defined in [Simulator.py](Simulation/Simulator.py) can be used. An example how to use the class can be found in the [CalculateForces.ipynb](Simulation/CalculateForces.ipynb) Jupyter notebook.

### Machine Learning

Models are defined in the [GNN_Models.py](MachineLearning/GNN_Models.py). For training the GNN models the Trainer class defined in [Trainer.py](MachineLearning/GNN_Trainer.py) can be used. An example how to use the class can be found in the [Train_models.ipynb](MachineLearning/Train_models.ipynb) Jupyter notebook.

### Analysis

For analysing the simulated trajectories the PeptideAnalyzer class defined in [analysis.py](Analysis/analysis.py) can be used. An example how to use the class can be found in the [Analysis_example.ipynb](Analysis/Analysis_example.ipynb) Jupyter notebook.