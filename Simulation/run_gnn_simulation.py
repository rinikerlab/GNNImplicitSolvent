import argparse

parser = argparse.ArgumentParser(description='Run ML MM')
# Simulation
parser.add_argument('-s','--smiles',type=str,help='SMILES of molecule')
parser.add_argument('-ns','--ns',default=20,type=float,help='Time of simulation in ns')
parser.add_argument('-nr','--num_rep',default=20,type=int,help='replicates to simulate at once')
parser.add_argument('-r','--random',default=10,type=int,help='random seed')
# GNN
parser.add_argument('-m','--model_path',type=str,help='Path to trained model')

args = parser.parse_args()

import sys
sys.path.append('../')
sys.path.append('../MachineLearning/')

import os
from Simulator import Simulator, Multi_simulator
from ForceField.Forcefield import TIP5P_force_field, TIP3P_force_field, GB_Neck2_force_field, Vacuum_force_field, Vacuum_force_field_plus_custom, Vacuum_force_field, OpenFF_forcefield_vacuum_plus_custom, OpenFF_forcefield_vacuum
from openmmtorch import TorchForce
from openmm import LangevinMiddleIntegrator
from MachineLearning.GNN_Models import *
from MachineLearning.GNN_Trainer import Trainer
from openmmtorch import TorchForce
from openmm.unit import kelvin, picosecond, picoseconds, bar
import pandas as pd

run_name = 'GNN3'

work_dir = "../" # directory of the repository
n_interval = 100 # Interval for saving frames in steps
ns = args.ns # Nanoseconds to run the simulation for

num_rep = args.num_rep
smiles = args.smiles + "_in_v"
save_name = 'GNN3_multi'

msim = Multi_simulator(work_dir=work_dir,pdb_id=smiles,num_rep=num_rep,run_name=run_name,save_name=save_name)
trainer = Trainer(verbose=True,name='GNN3',path='../MachineLearning/trained_models',force_mode=True,enable_tmp_dir=False,random_state=10)

radius = 0.6
fraction = 0.1
random_seed = '10'
n_steps = ns / 0.002 * 1000

model = GNN3_scale_64
run_model = GNN3_scale_64_run
runfile, reffile = msim.generate_model_pt_file(trainer,work_dir,smiles,args.model_path,radius,fraction,model,run_model,random_seed,device='cuda')

torch_force = TorchForce(runfile)
msim.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
msim.forcefield = OpenFF_forcefield_vacuum_plus_custom(smiles,torch_force,'GNN3_multi_' + str(msim._num_rep) + '_random_' + str(args.random))

torch_force = TorchForce(reffile)
msim._ref_system.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
msim._ref_system.forcefield = OpenFF_forcefield_vacuum_plus_custom(smiles,torch_force,'GNN3_multi_' + str(msim._num_rep) + '_random_' + str(args.random))

msim.setup_replicates()
msim.set_random_positions_for_each_replicate()
msim.run_simulation( n_steps ,n_interval,minimize=False)
msim.save_states(0)
