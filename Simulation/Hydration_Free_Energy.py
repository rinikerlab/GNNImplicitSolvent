import sys
sys.path.append('../')
sys.path.append('../MachineLearning/')
from MachineLearning.GNN_Trainer import Trainer
from openmmtorch import TorchForce
from Simulator import Multi_simulator, Simulator
from ForceField.Forcefield import OpenFF_forcefield_vacuum, OpenFF_forcefield_vacuum_plus_custom, OpenFF_forcefield_GBNeck2
import mdtraj
import numpy as np
from openmm import LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, picoseconds
from MachineLearning.GNN_Models import GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr, GNN3_all_swish_GBNeck_trainable_dif_graphs_corr_run_multiple
import tqdm
from pymbar import MBAR
from pymbar import timeseries 


class Hydration_Free_Energy_calculator:

    def __init__(self,pdb_id,num_rep,vac_traj_file,gnn_traj_file,trained_model_file,workdir='.',gnn_model=None,gnn_run_model=None,cache=None):

        self._pdb_id = pdb_id
        self._num_rep = num_rep
        self._vac_traj_file = vac_traj_file
        self._gnn_traj_file = gnn_traj_file
        self._trained_model_file = trained_model_file
        self._workdir = workdir
        self._cache = cache
        self._gnn_model = gnn_model
        self._gnn_run_model = gnn_run_model

    def load_trajectories(self):
        vac_traj = mdtraj.load(self._vac_traj_file)
        self._individual_vac_traj = [vac_traj.atom_slice(vac_traj.top.select('chainid %i' % i)) for i in range(self._num_rep)]

        gnn_traj = mdtraj.load(self._gnn_traj_file)
        self._individual_gnn_traj = [gnn_traj.atom_slice(gnn_traj.top.select('chainid %i' % i)) for i in range(self._num_rep)]

    def create_simulators(self):

        self._vac_sim = Simulator(work_dir=self._workdir,pdb_id=self._pdb_id,run_name=self._pdb_id)
        self._vac_sim.forcefield = OpenFF_forcefield_vacuum(self._pdb_id,cache=self._cache)
        self._vac_sim.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        self._vac_sim.platform = "GPU"


        if not self._gnn_model is None:
            self._gnn_sim = Multi_simulator(work_dir=self._workdir,pdb_id=self._pdb_id,run_name='CalculateHydrationFreeEnergy',num_rep=1,cache=self._cache)
            trainer = Trainer(verbose=True,name='CalculateHydrationFreeEnergy',force_mode=True,enable_tmp_dir=False,random_state=10)
            trained_model = self._trained_model_file
            radius = 0.6
            fraction = 0.1
            random_seed = '10'
            model = self._gnn_model
            run_model = self._gnn_run_model
            runfile, reffile = self._gnn_sim.generate_model_pt_file(trainer,self._workdir,self._pdb_id,trained_model,radius,fraction,model,run_model,random_seed)

            torch_force = TorchForce(runfile)
            self._gnn_sim.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
            self._gnn_sim.forcefield = OpenFF_forcefield_vacuum_plus_custom(self._pdb_id,torch_force,'GNN3_vap',cache=self._cache)
            self._gnn_sim.platform = "GPU"
    
    def evaluate_hamiltonians(self,num_pos):
        
        self._vac_pos_vac_energies = np.empty(num_pos*self._num_rep)
        self._vac_pos_gnn_energies = np.empty(num_pos*self._num_rep)

        for j, vac_traj in enumerate(self._individual_vac_traj):

            frac = len(vac_traj)//num_pos
            assert not (len(vac_traj) % num_pos)

            vac_pos = vac_traj.xyz[::frac]

            for i in range(len(vac_pos)):
                pos = vac_pos[i]
                self._vac_sim.set_positions(pos)
                self._vac_pos_vac_energies[j*len(vac_pos) + i] = self._vac_sim.calculate_energy()._value
                if not self._gnn_model is None:
                    self._gnn_sim.set_positions(pos)
                    self._vac_pos_gnn_energies[j*len(vac_pos) + i] = self._gnn_sim.calculate_energy()._value

        self._gnn_pos_vac_energies = np.empty(num_pos*self._num_rep)
        self._gnn_pos_gnn_energies = np.empty(num_pos*self._num_rep)

        for j, gnn_traj in enumerate(self._individual_gnn_traj):

            frac = len(gnn_traj)//num_pos
            assert not (len(gnn_traj) % num_pos)

            gnn_pos = gnn_traj.xyz[::frac]

            for i in range(len(gnn_pos)):
                self._vac_sim.set_positions(gnn_pos[i])
                self._gnn_pos_vac_energies[j*len(gnn_pos) + i] = self._vac_sim.calculate_energy()._value
                self._gnn_sim.set_positions(gnn_pos[i])
                self._gnn_pos_gnn_energies[j*len(gnn_pos) + i] = self._gnn_sim.calculate_energy()._value

    def add_GBNeck2_model(self,SA=None):

        self._gb_sim = Simulator(work_dir=self._workdir,pdb_id=self._pdb_id,run_name=self._pdb_id)
        self._gb_sim.forcefield = OpenFF_forcefield_GBNeck2(self._pdb_id,SA=SA,cache=self._cache)
        self._gb_sim.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        self._gb_sim.platform = "GPU"
    
    def evaluate_GBNeck2_hamiltonian(self,num_pos):

        self._vac_pos_gb_energies = np.empty(num_pos*self._num_rep)

        for j, vac_traj in enumerate(self._individual_vac_traj):

            frac = len(vac_traj)//num_pos
            assert not (len(vac_traj) % num_pos)

            vac_pos = vac_traj.xyz[::frac]

            for i in range(len(vac_pos)):
                pos = vac_pos[i]
                self._gb_sim.set_positions(pos)
                self._vac_pos_gb_energies[j*len(vac_pos) + i] = self._gb_sim.calculate_energy()._value

        self._gnn_pos_gb_energies = np.empty(num_pos*self._num_rep)

        for j, gnn_traj in enumerate(self._individual_gnn_traj):

            frac = len(gnn_traj)//num_pos
            assert not (len(gnn_traj) % num_pos)

            gnn_pos = gnn_traj.xyz[::frac]

            for i in range(len(gnn_pos)):
                self._gb_sim.set_positions(gnn_pos[i])
                self._gnn_pos_gb_energies[j*len(gnn_pos) + i] = self._gb_sim.calculate_energy()._value

    def calculate_hydration_free_energies_GBNeck2(self):

        vac_pos_energies = np.array([self._vac_pos_vac_energies,self._vac_pos_gb_energies])
        gnn_pos_energies = np.array([self._gnn_pos_vac_energies,self._gnn_pos_gb_energies])
        sub_vac_pos_energies = vac_pos_energies
        sub_gnn_pos_energies = gnn_pos_energies

        N_k = np.array([len(sub_vac_pos_energies[0]),len(sub_gnn_pos_energies[0])])
        U_kn = np.concatenate((sub_vac_pos_energies,sub_gnn_pos_energies),axis=1)

        kT = 2.479
        u_kn = U_kn / kT

        mbar = MBAR(u_kn=u_kn,N_k=N_k)
        results = mbar.getFreeEnergyDifferences()

        # return self._vac_pos_vac_energies,self._vac_pos_gnn_energies, self._gnn_pos_vac_energies,self._gnn_pos_gnn_energies
        # return N_k, U_kn
        return results[0] * kT


    def calculate_hydration_free_energies(self):
        print("Warning should use subsample")

        vac_pos_energies = np.array([self._vac_pos_vac_energies,self._vac_pos_gnn_energies])
        gnn_pos_energies = np.array([self._gnn_pos_vac_energies,self._gnn_pos_gnn_energies])
        sub_vac_pos_energies = vac_pos_energies
        sub_gnn_pos_energies = gnn_pos_energies

        N_k = np.array([len(sub_vac_pos_energies[0]),len(sub_gnn_pos_energies[0])])
        U_kn = np.concatenate((sub_vac_pos_energies,sub_gnn_pos_energies),axis=1)

        kT = 2.479
        u_kn = U_kn / kT

        mbar = MBAR(u_kn=u_kn,N_k=N_k)
        results = mbar.getFreeEnergyDifferences()

        # return self._vac_pos_vac_energies,self._vac_pos_gnn_energies, self._gnn_pos_vac_energies,self._gnn_pos_gnn_energies
        # return N_k, U_kn
        return results[0] * kT

    def calculate_sasa(self):

        gamma = 0.00542 # kcal/(mol A^2)
        beta = 0.92 # kcal / mol

        sasa = mdtraj.shrake_rupley(self._individual_vac_traj[0][::100])
        total_sasa = sasa.sum(axis=1)
        Gnp = gamma * np.mean(total_sasa)*100 + beta

        return Gnp

def get_sub(data):
    sub = data[timeseries.subsampleCorrelatedData(data)] 
    return sub     


