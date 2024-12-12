import argparse

parser = argparse.ArgumentParser(description='Run ML MM')
# Simulation
parser.add_argument('-id','--pdb_id',type=str,help='ID of pdb, if available amber files will be used')
parser.add_argument('-ns','--ns',default=20,type=float,help='Time of simulation in ns')
parser.add_argument('-fi','--file',type=str,help='file to map ids to smiles')
parser.add_argument('-nr','--num_rep',default=20,type=int,help='replicates to simulate at once')
parser.add_argument('-a','--addition',default='',type=str,help='Additional information')
# GNN
parser.add_argument('-b','--batchsize',default=32,type=int,help='Batchsize')
parser.add_argument('-p','--per',default=0.8,type=float,help='fraction of training')
parser.add_argument('-f','--fra',default=0.1,type=float,help='scaling parameter')
parser.add_argument('-r','--random',default=161311,type=int,help='random seed')
parser.add_argument('-ra','--radius',default=0.6,type=float,help='radius')
parser.add_argument('-l','--lr',default=0.001,type=float,help='learning rate')
parser.add_argument('-fpt','--ptfile',default='.',type=str,help='Pytorch file with stored training data')
parser.add_argument('-e','--epochs',default=30,type=int,help='epochs to train for')
parser.add_argument('-m','--modelid',default=0,type=int,help='Model_architecture')
parser.add_argument('-n','--name',default='',type=str,help='name of model')
parser.add_argument('-c','--clip',default=0,type=float,help='norm clipping')

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

if args.file == 'none':
    pdb_id = args.pdb_id + '_in_v'
    run_name = pdb_id
    save_name = pdb_id
else:
    pdbs = pd.read_table(args.file,names=['smi']).values.T[0]
    pdb_id = pdbs[int(args.pdb_id)] + '_in_v'
    run_name = args.file.split('/')[-1].split('.')[0]
    save_name = run_name + '_id_%i' % int(args.pdb_id)


# make string of commands
model_inf_string = ''
for arg in vars(args):
    if arg in ['folder','mapping_file','ptfile','npfile','pdb_id','ns','file','num_rep','addition']:
        continue

    model_inf_string += '_%s_%s' % (arg, getattr(args, arg))

model_class_dict = {128 : GNN3_scale_128,
                    96 : GNN3_scale_96,
                    64 : GNN3_scale_64,
                    48 : GNN3_scale_48,
                    32 : GNN3_scale_32}

run_model_class_dict = {128 : GNN3_scale_128_run,
                    96 : GNN3_scale_96_run,
                    64 : GNN3_scale_64_run,
                    48 : GNN3_scale_48_run,
                    32 : GNN3_scale_32_run}

run_name += '_pub'
run_name += args.addition

work_dir = "../" # directory of the repository
n_interval = 100 # Interval for saving frames in steps
ns = args.ns # Nanoseconds to run the simulation for

num_rep = args.num_rep

msim = Multi_simulator(work_dir=work_dir,pdb_id=pdb_id,num_rep=num_rep,run_name=run_name,save_name=save_name,random_number_seed=args.random)
trainer = Trainer(verbose=True,name='GNN3_pub_' + model_inf_string,path='../MachineLearning/trained_models',force_mode=True,enable_tmp_dir=False,random_state=10)

radius = 0.6
fraction = 0.1
random_seed = '10'
n_steps = ns / 0.002 * 1000

model = model_class_dict[args.modelid]
run_model = run_model_class_dict[args.modelid]
runfile, reffile = msim.generate_model_pt_file(trainer,work_dir,pdb_id,trainer.model_path,radius,fraction,model,run_model,random_seed,device='cuda')

torch_force = TorchForce(runfile)
msim.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
msim.forcefield = OpenFF_forcefield_vacuum_plus_custom(pdb_id,torch_force,'GNN3_multi_' + str(msim._num_rep) + '_model_' + str(args.modelid) + '_random_' + str(args.random))

torch_force = TorchForce(reffile)
msim._ref_system.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
msim._ref_system.forcefield = OpenFF_forcefield_vacuum_plus_custom(pdb_id,torch_force,'GNN3_multi_' + str(msim._num_rep)+ '_model_' + str(args.modelid) + '_random_' + str(args.random))

msim.setup_replicates()
msim.set_random_positions_for_each_replicate()
msim.run_simulation( n_steps ,n_interval,minimize=False)
msim.save_states(0)
