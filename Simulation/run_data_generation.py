'''
This file is an example on how to extract the data from the external test set.
The calculation of the forces for training is done in analogy to this file.
'''

import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENMM_CPU_THREADS'] = '8'

import sys
sys.path.append('../')
from Data.Datahandler import hdf5_storage
from rdkit import Chem

import sys
sys.path.append('../')
sys.path.append('../MachineLearning')
from MachineLearning.GNN_Trainer import Trainer
from Simulation.Simulator import Simulator
from ForceField.Forcefield import OpenFF_forcefield
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.unit import kelvin, picosecond, picoseconds, bar
import numpy as np
from Data.Datahandler import hdf5_storage
from Simulation.Simulator import Explicit_solvent_simulator_force_only
import argparse
import os
import time

def handler(signum, frame):
    print("TIME")
    raise Exception("end of time")

def get_max_dif(arr):
    return np.max(arr) - np.min(arr)

parser = argparse.ArgumentParser(description='Run Force Extraction')
parser.add_argument('-i','--runid',type=int,help='ID of SDF file to use')
parser.add_argument('-s','--sdf',type=str,help='SDF file to use')
parser.add_argument('-c','--cache',type=str,help='Cache file to use')
parser.add_argument('-h5','--hdf5',type=str,help='HDF5 file to use')
parser.add_argument('--return_numpy_files',action='store_true',help='Return numpy files')
args = parser.parse_args()
runid = args.runid

file = args.sdf + '%i.sdf' % runid
cfile = os.environ['TMPDIR']+ args.sdf.split('/')[-1] +'%i.sdf' % runid

os.system("cp %s %s" % (file,cfile))

storage = hdf5_storage(args.hdf5 + '%i.h5' % runid )
supl = Chem.SDMolSupplier(cfile,removeHs=False)

rand = args.runid
am1bcc_cache = args.cache + '%i.json' % runid

for m,mol in enumerate(supl):

    print('START')
    start_time = time.time()

    sdf_exist, sim_exist, extracted_exist, atomfeatures_exist = storage.get_existing_entries(str(m),mol)

    if sdf_exist and sim_exist and extracted_exist and atomfeatures_exist:
        continue

    print(sdf_exist, sim_exist, extracted_exist, atomfeatures_exist,flush=True)
    try:

        conf = mol.GetConformer()
        xyz = conf.GetPositions()

        # GET 1 nm padding
        boxlength = np.max([get_max_dif(xyz[:,i]) for i in range(3)])/10 + 2
        print('Boxsize %.3f nm' % boxlength)

        if not sdf_exist:
            molid, confid = storage.create_sdf_entry(str(m),mol) # add to loop
        else:
            mcdict = storage.get_molids_and_confids()
            molid = str(m)
            confid = mcdict[molid][0]

        pdb_id = storage.get_smiles(molid) + '_in_O'
        pdb = storage.get_pdb_string(molid,confid)

        print('TISD')
        print(time.time() - start_time)
        print('END',flush=True)

        sim_traj = None
        if not sim_exist:
            # Setup Simulation
            work_dir = os.environ['TMPDIR']+'/' # directory of the repository
            n_interval = 10000 # Interval for saving frames in steps
            ns = 0.5 # Nanoseconds to run the simulation for

            run_name = 'Small_molecules_%i' % runid
            save_name = 'Small_molecules_molid_%s_confid_%s' % (molid,confid)

            # Simulate
            sim = Simulator(work_dir=work_dir,pdb_id=pdb_id,run_name=run_name,save_name=save_name,boxlength=boxlength)
            forcefieldtime = time.time()
            sim._datahandler.ready = False
            sim.forcefield = OpenFF_forcefield(pdb_id,'TIP3P',cache=am1bcc_cache)
            sim._platform = "GPU"
            sim.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
            sim._datahandler.ready = True
            sim.barostat = MonteCarloBarostat(1*bar,300*kelvin)

            print('TIFO')
            print(time.time() - forcefieldtime)
            print('END',flush=True)

            n_steps = ns / 0.002 * 1000
            sim.run_simulation(n_steps=n_steps,minimize=False,n_interval=n_interval)
            sim._simulation.reporters[0].close()

            # Save Simulation
            storetime = time.time()
            sim_traj = storage.create_simulation_entry_from_files(molid,confid,sim.hdf5_loc, sim.log_loc)
            
            print('TISTO')
            print(time.time() - storetime)
            print('END',flush=True)

            # Delete Simulation to free up threads
            del sim


        print('TISI')
        print(time.time() - start_time)
        print('END',flush=True)

        if not extracted_exist:
            work_dir = os.environ['TMPDIR']+'/' # directory of the repository
            run_name = 'Small_molecules_%i' % runid
            save_name = 'Small_molecules_molid_%s_confid_%s' % (molid,confid)
            # Extract Forces
            if sim_traj is None:
                traj = storage.get_trajectory(molid,confid)
            else:
                traj = sim_traj
            
            test = Explicit_solvent_simulator_force_only(work_dir=work_dir,name="ligandsforce",run_name='ligandsforce_%i' % runid,
                                                        pdb_id=pdb_id,hdf5_file=None,boxsize=boxlength,save_name=save_name,
                                                        starting_frame_traj=traj,pdb=None,cache=am1bcc_cache)
            frames = [8 + i*8 for i in range(3)]
            savedir = os.environ['TMPDIR']+'/' 
            test.read_in_frame_and_set_positions(0)
            test.constrain_solute()
            force_file, pos_file, frames_file = test.calculate_mean_force_for_pre_calc_pos(save_location=savedir,save_add='run_%i' % runid, n_steps=100, n_frames=1000,frames = frames)
            if args.return_numpy_files:
                print('Force file:',force_file)
                print('Pos file:',pos_file)

            time.sleep(1)
            # Save Forces
            storage.create_extraction_entry_from_file(molid,confid,force_file, pos_file, frames_file)
            storage.create_reprocessed_force_entry(molid,confid,np.load(force_file))
       
        print('TIEX')
        print(time.time() - start_time)
        print('END',flush=True)

        if not atomfeatures_exist:
            work_dir = os.environ['TMPDIR']+'/' # directory of the repository
            run_name = 'Small_molecules_%i' % runid
            save_name = 'Small_molecules_molid_%s_confid_%s' % (molid,confid)
            smiles = storage.get_smiles(molid)
            atom_features, uniqueRadii = Trainer.get_gbneck2_param_small_molecules_unique(storage.get_smiles(molid) + '_in_v',work_dir=os.environ['TMPDIR']+'/',cache=am1bcc_cache)
            storage.create_atom_feature_entry(molid,confid,atom_features,uniqueRadii)
            storage.create_reprocessed_atom_feature_entry(molid,confid,atom_features,uniqueRadii)
            print("atom features added",flush=True)

        print('TIME')
        print(time.time() - start_time)
        print('END',flush=True)

    except Exception as e:
        print("failed ad iteration: %i" % m)
        print(e,flush=True)
