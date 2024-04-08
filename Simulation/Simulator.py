import copy
import os
import sys
import time
import warnings

import mdtraj
import numpy as np
import pandas as pd
import tqdm

from Data.Datahandler import DataHandler
from ForceField.Forcefield import _generic_force_field, TIP5P_force_field, Vacuum_force_field, OpenFF_forcefield, OpenFF_forcefield_vacuum
from openmm.app import Simulation, Modeller, PDBReporter, StateDataReporter, CheckpointReporter
from mdtraj.reporters import HDF5Reporter
from openmm import Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.unit import nanometer, picoseconds, nanoseconds, elementary_charge, Quantity, kelvin, picosecond, \
    nanometers, kilojoules_per_mole, norm, bar
from openmm.app import element
from openmm.vec3 import Vec3

from openmm import HarmonicBondForce, NonbondedForce, Context
from copy import deepcopy

from time import sleep
#from MachineLearning.GNN_Trainer import Trainer
from MachineLearning.GNN_Graph import get_Graph_for_one_frame
from MachineLearning.GNN_Models import *
import torch

from openmm.app import ForceField, PME, HBonds
from sklearn.metrics import mean_squared_error
from openmm.unit import kelvin, picosecond, picoseconds

from openmm import NonbondedForce, CustomNonbondedForce, CustomBondForce

from rdkit import Chem
from rdkit.Chem import AllChem

class Simulator:
    '''
    Class for running simulations.
    '''

    def __init__(self, work_dir: str, name: str = "", pdb_id: str = "", forcefield: _generic_force_field =
    _generic_force_field(), integrator=None, platform: str = "CPU", cutoff=1 * nanometer, run_name="",barostat=None,save_name=None,solute_pdb=None,boxlength=None,padding=2,create_data=True,random_number_seed=0):
        '''
        Initialice the Simulator Class
        :param work_dir: directory to save files in
        :param name: Name of the simulator (for string name)
        :param pdb_id: ID of protein to simulate
        :param forcefield: ForceField to use
        :param integrator: Integrator to use
        :param platform: Platform to use
        :param cutoff: Cutoff to use NOTE: Only applies for non "implicit" FFs
        :param run_name: Name of the run (creates an folder in which all resutls are then saved)
        '''
        self._work_folder = work_dir + "Simulation/simulation/" + run_name + "/"
        if save_name is None:
            self._save_name = pdb_id
        else:
            self._save_name = save_name

        if not os.path.isdir(self._work_folder):
            os.makedirs(self._work_folder)

        self._iteration = 0
        self._name = name
        self._pdb_id = pdb_id
        self._cutoff = cutoff
        if create_data:
            self._datahandler = DataHandler(work_dir=work_dir, pdb_id=self._pdb_id,solute_pdb=solute_pdb,boxlength=boxlength,padding=padding,random_number_seed=random_number_seed)
        else:
            self._datahandler = DataHandler(work_dir=work_dir,random_number_seed=random_number_seed)
        self._forcefield = forcefield
        self._integrator = integrator
        self._barostat = barostat
        self._random_number_seed = random_number_seed

        self.update_all_properties()
        self.platform = platform

    def clean_file(self):
        '''
        Clean File
        :return:
        '''
        self._datahandler.clean_file()

    def make_terminal(self):
        '''
        Add Terminals to pdb (Note: renaming residues only)
        :return:
        '''

        self._datahandler.make_terminal()

    def create_system_with_padding(self, padding=2.0):
        """
        Create a system with padding.

        Args:
            padding (float): The amount of padding to add around the system.

        Returns:
            None
        """
        self._datahandler.create_box_with_padding(padding)

    def change_box_size(self, boxlength = 4):

        self._datahandler.change_box_size(boxlength)

    def add_box(self, padding: float = 1.0):
        '''
        Creates a Box of solvent molecules as defined in the forcefield
        :param padding: Padding for calculating the box size.
        :return:
        '''
        modeller = Modeller(self._datahandler.topology, self._datahandler.positions)
        modeller.addSolvent(self._forcefield.forcefield, padding=padding * nanometer,
                            model=self._forcefield.water_model)
        self._datahandler.topology = modeller.topology
        self._datahandler.positions = modeller.positions
        self._datahandler.rebase_pdb()

    def add_virtual_sites(self):
        
        modeller = Modeller(self._datahandler.topology, self._datahandler.positions)
        modeller.addExtraParticles(self._forcefield.forcefield)
        self._datahandler.topology = modeller.topology
        self._datahandler.positions = modeller.positions

    def create_system(self):
        '''
        Use Forcefield to create system
        :return:
        '''
        assert self._datahandler.ready and self._forcefield.ready
        self._system = self._forcefield.create_system(topology=self._datahandler.topology,
                                                      nonbondedCutoff=self._cutoff)
        if not self._barostat is None:
            self._system.addForce(deepcopy(self._barostat))

    def update_all_properties(self):
        '''
        Update all properties to match new settings. Automatically creates system and simulation if requirements are met.
        :return:
        '''
        if self._datahandler.ready and self._forcefield.ready:
            try:
                self.add_virtual_sites()
            except:
                'print no virtual sites added'
            self.create_system()
            if self._integrator:
                self.create_simulation()
            else:
                self._simulation = None
        else:
            self._system = None
            self._simulation = None

    def create_simulation(self, positions=None):
        '''
        Create Simulation
        :return:
        '''
        assert self._datahandler.ready and self._forcefield.ready and self._integrator

        # Reset integrator for new simulation
        integrator = copy.deepcopy(self._integrator)
        del self._integrator
        self._integrator = integrator

        if self._platform == "GPU":
            platform = Platform.getPlatformByName('CUDA')
            platformProperties = {'Precision': 'mixed', 'CudaPrecision': 'mixed'}
            self._simulation = Simulation(self._datahandler.topology, self._system, self._integrator, platform,
                                          platformProperties)
        elif self._platform == "OpenCL":
            platform = Platform.getPlatformByName('OpenCL')
            platformProperties = {'Precision': 'mixed'}
            self._simulation = Simulation(self._datahandler.topology, self._system, self._integrator, platform,
                                          platformProperties)
        else:
            platform = Platform.getPlatformByName('CPU')
            self._simulation = Simulation(self._datahandler.topology, self._system, self._integrator, platform)

        # check if positions need to be initialized
        if positions is None:
            original_pos = self._simulation.context.getState(getPositions=True).getPositions()
            if original_pos._value[0] == original_pos._value[1]:
                self._simulation.context.setPositions(self._datahandler.positions)
                #print("set initial positions of PDB")
        else:
            self._simulation.context.setPositions(positions)

    def adapt_values(self, Data):
        '''
        Adapt the values of a forcefield object, should be used with a custom force object.
        :param Data: np.array((number_atoms,3))
        :return:
        '''
        # Check whether single or double force
        if len(self._simulation.context.getSystem().getForces()) == 6:
            force = self._simulation.context.getSystem().getForces()[5]
            for i in range(len(Data)):
                if self._forcefield.name == "GBSA_ACE":
                    # index, charge, Born radius, Atomic radius, scale factor
                    force.setParticleParameters(i, [Data[i, 0], Data[i, 1] * 0.1, Data[i, 2] * 0.1, Data[i, 3]])
                elif self._forcefield.name == "GBSA_ACE_born_scale":
                    # index, charge, B scale, Atomic radius, scale factor
                    force.setParticleParameters(i, [Data[i, 0], Data[i, 1], Data[i, 2] * 0.1, Data[i, 3]])
                elif self._forcefield.name == "GBSA_ACE_I_scale" or self._forcefield.name == "GBSA_ACE_I_scale_no_SASA":
                    # index, charge, I scale, Atomic radius, scale factor
                    force.setParticleParameters(i, [Data[i, 0] * elementary_charge, Data[i, 1],
                                                    Data[i, 2] * 0.1 * nanometer, Data[i, 3]])
                elif self._forcefield.name == "GBSA_OBC_ACE":
                    force.setParticleParameters(i, [Data[i, 0], Data[i, 2] * 0.1, Data[i, 3]])
                else:
                    force.setParticleParameters(i, [Data[i, 0], Data[i, 1] * 0.1, Data[i, 2] * 0.1])

            force.updateParametersInContext(self._simulation.context)

        else:
            single_force = self._simulation.context.getSystem().getForces()[5]
            pair_force = self._simulation.context.getSystem().getForces()[6]

            for i in range(len(Data)):
                single_force.setParticleParameters(i, i, (Data[i, 0], Data[i, 1] * 0.1))
                pair_force.setParticleParameters(i, (Data[i, 0], Data[i, 1] * 0.1, Data[i, 2] * 0.1))

            single_force.updateParametersInContext(self._simulation.context)
            pair_force.updateParametersInContext(self._simulation.context)

    def run_simulation(self, n_steps: int = 10000, n_interval: int = 1000, minimize: bool = True, workfolder=None):
        '''
        Run Simulation
        :param n_steps: Steps to run
        :param n_interval: Interval for saving the .h5 file
        :param minimize: If true a minimization is performed.
        :param workfolder: The folder intermediate files should be stored in. On LSF the $TMPDIR can be used to utilize local scratch
        :return:
        '''

        if workfolder is None:
            workfolder = self._work_folder
        elif workfolder == "TMPDIR":
            workfolder = os.environ['TMPDIR'] + "/"

        if (minimize and self._iteration == 0):
            self._simulation.minimizeEnergy()
        if len(self._simulation.reporters) == 0:
            self._simulation.reporters.append(
                HDF5Reporter(workfolder + self._save_name + '_' + str(self._forcefield) + '_' + str(self._iteration) + '_' + str(self._random_number_seed) + '_output.h5', n_interval))
        if len(self._simulation.reporters) == 1:
            stdout = open(workfolder + self._save_name + '_' + str(self._forcefield) + '_' + str(self._iteration)+ '_' + str(self._random_number_seed) + '_log.txt', "w")
            self._simulation.reporters.append(
                StateDataReporter(stdout, reportInterval=n_interval, step=True, speed=True, potentialEnergy=True,
                                  temperature=True))
        self._simulation.step(n_steps)

    @property
    def hdf5_loc(self):
        return self._work_folder + self._save_name + '_' + str(self._forcefield) + '_' + str(self._iteration) + '_' + str(self._random_number_seed) + '_output.h5'
    
    @property
    def log_loc(self):
        return self._work_folder + self._save_name + '_' + str(self._forcefield) + '_' + str(self._iteration) + '_' + str(self._random_number_seed) + '_log.txt'

    def get_radii_data(self, workfolder=None, bluues_executable="", initial=False, msms_executable="",
                       include_scale=False, calculate_born_scale=False, calculate_I_scale=True):
        '''
        Get Born radii
        Warning radii variable can also hold born or I scale
        :param workfolder: The folder intermediate files should be stored in. On LSF the $TMPDIR can be used to utilize local scratch
        :param bluues_executable: The executable for the Bluues Program used for calculating the Radii
        :param initial: Flag whether the initial Born radii should be found or the ones for the last frame
        :return: Data array
        '''

        if not workfolder:
            workfolder = self._work_folder
        elif workfolder == "TMPDIR":
            # add unique workfolder to avoid overlapping data
            workfolder = os.environ['TMPDIR'] + "/" + self._pdb_id + '_' + str(self._forcefield) + "/"

            # Create Folders if necessary
            if not os.path.isdir(workfolder):
                os.makedirs(workfolder)

        if workfolder == "TMPDIR" and not initial:
            pdbin = workfolder + "initial.pdb"
        else:
            pdbin = self._datahandler.file_path

        if initial:
            infile = self._datahandler.file_path
        else:
            infile = workfolder + self._pdb_id + '_' + str(self._forcefield) + '_output.pdb'

        pd = PDB2PQR()

        pqr_out = workfolder + self._pdb_id + '_' + str(self._forcefield) + ".pqr"

        if include_scale:
            radii, charges, radius, scale = pd.pdb2pqr(infile=infile, outfile=pqr_out, pdbfile=pdbin,
                                                       workdir=workfolder,
                                                       n_steps=1,
                                                       save_radii=False, return_charges=True, return_radius=True,
                                                       bluues_executable=bluues_executable,
                                                       msms_executable=msms_executable,
                                                       calculate_scale=True, calculate_born_scale=calculate_born_scale,
                                                       calculate_I_scale=calculate_I_scale)
            radii_and_charge_data = np.empty((len(radii), 4), dtype=np.float32)
        else:
            radii, charges, radius = pd.pdb2pqr(infile=infile, outfile=pqr_out, pdbfile=pdbin, workdir=workfolder,
                                                n_steps=1,
                                                save_radii=False, return_charges=True, return_radius=True,
                                                bluues_executable=bluues_executable, msms_executable=msms_executable)
            radii_and_charge_data = np.empty((len(radii), 3), dtype=np.float32)
        radii_and_charge_data[:, 0] = charges
        radii_and_charge_data[:, 1] = radii
        radii_and_charge_data[:, 2] = radius
        if include_scale:
            radii_and_charge_data[:, 3] = scale
        return radii_and_charge_data

    def get_force_field_parameters(self, force_name="GBSAOBCForce"):
        '''
        Return the parameters of a force
        :param force_name: Name of the force which parameters should be returned
        :return: np.array of (number particles,number parameters)
        '''

        force_dict = {}

        forces = self._system.getForces()
        # Get all forces by name
        for force in forces:
            force_dict[str(force).split(".")[2].split(";")[0]] = force

        # Get desired force
        force = force_dict[force_name]
        num_particles = force.getNumParticles()
        num_parameters = len(force.getParticleParameters(0))
        parameters = np.empty((num_particles, num_parameters))

        for i in range(num_particles):
            parameters[i, :] = [convert_unit_to_float(j) for j in force.getParticleParameters(i)]

        return parameters

    def initiate_BornCalculator(self):
        self._born_calculator = Born_Calculator()

    def add_ml_model(self, file):
        '''
        Load Model from file
        :param file:
        :return:
        '''
        sys.path.append('/home/kpaul/implicitml/MachineLearning')
        self._ml_model = torch.load(file, map_location=torch.device('cpu'))

    def get_implict_info(self):
        pdbin = self._datahandler.original_file_path
        self._born_calculator.get_calculation_input(pdbin, pdbin)
        self._born_calculator.generate_input_for_Born_calc()
        data = self._born_calculator.data.values
        implicit_data = np.empty((data.shape[0], 4))
        implicit_data[:, 0] = data[:, 10]
        implicit_data[:, 1] = 1
        implicit_data[:, 2] = data[:, 11]
        implicit_data[:, 3] = data[:, 12]
        return implicit_data

    def get_protein_info(self):
        data = self._datahandler.get_protein_info()
        print(data.head())
        exit()

    def run_hybrid_simulation(self, n_steps=10000, update_radii_every=1000, n_interval=1000, bluues_executable="",
                              workfolder=None, msms_executable="", saveout_individual_pdbs=False,
                              calculate_born_scale=False, calculate_I_scale=False):
        '''
        Run a hybrid simulation where the radii are recalculated using the Bluues program.
        :param n_steps:
        :param update_radii_every:
        :param n_interval:
        :param bluues_executable:
        :param workfolder:
        :return:
        '''
        if not workfolder:
            workfolder = self._work_folder
        elif workfolder == "TMPDIR":
            workfolder = os.environ['TMPDIR'] + "/"

            # if TMPDIR exists move executable for faster access
            command = "cp " + bluues_executable + " " + workfolder
            os.system(command)
            bluues_executable = workfolder + bluues_executable.split("/")[-1]
            assert os.path.isfile(bluues_executable)

            command = "cp " + msms_executable + " " + workfolder
            os.system(command)
            msms_executable = workfolder + msms_executable.split("/")[-1]
            assert os.path.isfile(msms_executable)

            command = "cp " + self._datahandler.file_path + " " + workfolder + "initial.pdb"
            os.system(command)

        radii_and_charge_data = self._forcefield.Data
        self.run_simulation(n_steps=update_radii_every, n_interval=n_interval, workfolder=workfolder)

        Steps = int(n_steps / update_radii_every - 1)
        Radii = np.empty((Steps + 1, len(radii_and_charge_data[:, 1])))
        Radii[0, :] = radii_and_charge_data[:, 1]
        exceptions = 0
        for i in range(Steps):
            try:
                radii_and_charge_data = self.get_radii_data(bluues_executable=bluues_executable, workfolder=workfolder,
                                                            msms_executable=msms_executable,
                                                            include_scale=self._forcefield.scale_needed,
                                                            calculate_born_scale=calculate_born_scale,
                                                            calculate_I_scale=calculate_I_scale)
                exceptions = 0
            except:
                warnings.warn("Failed wait and try again: " + str(i))
                # if it does not work wait 0.05 seconds and try again
                for i in range(10):
                    sleep(0.02)
                    try:
                        radii_and_charge_data = self.get_radii_data(bluues_executable=bluues_executable,
                                                                    workfolder=workfolder,
                                                                    msms_executable=msms_executable,
                                                                    include_scale=self._forcefield.scale_needed,
                                                                    calculate_born_scale=calculate_born_scale,
                                                                    calculate_I_scale=calculate_I_scale)
                        exceptions = 0
                    except:
                        exceptions += 1
                        pass
                # if there is a problem skip and try again (until maximum amount (10) of exceptions in a row is reached:
                if exceptions != 0:
                    warnings.warn("Continued without update Step: " + str(i))
                    if exceptions == 100:
                        exit("Too many exceptions, calculation of radii failed")
            Radii[i + 1, :] = radii_and_charge_data[:, 1]
            self.adapt_values(radii_and_charge_data)
            self.run_simulation(n_steps=update_radii_every, n_interval=n_interval, minimize=False,
                                workfolder=workfolder)

            if saveout_individual_pdbs:
                copy_command = "cp "
                copy_command += workfolder + self._pdb_id + '_' + str(self._forcefield) + '_output.pdb '
                copy_command += workfolder + self._pdb_id + '_' + str(self._forcefield) + '_output' + str(i) + '.pdb'
                os.system(copy_command)

        np.savetxt(self._work_folder + "/" + self._pdb_id + "_radii_" + str(Steps) + ".csv", X=Radii, delimiter=",")

    def run_scaled_ml_simulation(self, n_steps=10000, update_radii_every=100, n_interval=1000,
                                 workfolder=None):
        '''
        Run a simulation using ML predicted Born Radii
        :param n_steps:
        :param update_radii_every:
        :param n_interval:
        :param workfolder:
        :return:
        '''
        if not workfolder:
            workfolder = self._work_folder
        elif workfolder == "TMPDIR":
            workfolder = os.environ['TMPDIR'] + "/"

        radii_and_charge_data = self._forcefield.Data

        # Iterate over update steps
        Steps = int(n_steps / update_radii_every - 1)
        for i in range(Steps):
            # Get radii from ML Model

            pos = self._simulation.context.getState(getPositions=True).getPositions()
            pos = np.array(pos)
            pos = np.array(list(map(get_values, pos)))

            # Build Graph input
            # start = time.time()
            graph = get_Graph_for_one_frame(pos, radii_and_charge_data[:, [0, 2]])
            graph.to('cpu')  # run on CPU to allow OpenMM to be on the GPU (for cluster)
            inverse_radii = np.array(self._ml_model(graph).tolist()).reshape((pos.shape[0]))
            radii = 1 / inverse_radii / 10  # in nm
            # print('Evaluating ML: %f.3' % (time.time()-start))

            # Get scaling factor
            # start = time.time()
            self._born_calculator.adapt_coordinates(pos)
            scale = self._born_calculator.calculate_I_scale(radii)
            radii_and_charge_data[:, 1] = scale
            radii_and_charge_data[:, 2] = radii_and_charge_data[:, 2]

            # print('Evaluating BR: %f.3' % (time.time() - start))
            # Update data
            self.adapt_values(radii_and_charge_data)

            # forces = self._simulation.context.getState(getForces=True).getForces()
            # x_forces = np.array([forces._value[i][0] for i in range(len(pos))])
            # y_forces = np.array([forces._value[i][1] for i in range(len(pos))])
            # z_forces = np.array([forces._value[i][2] for i in range(len(pos))])
            # print(np.array((x_forces,y_forces,z_forces)))
            # exit()
            # forces = self._sself._explicit_simulationimulation.context.getSystem().getForces()
            # forces[5].setForceGroup(31)
            # forces = self._simulation.context.getState(getForces=True,groups=31).getForces()
            # print(forces[0])
            # print(pos[0])#print(forces[5].getForces())
            # exit()
            # Run steps
            # start = time.time()
            self.run_simulation(n_steps=update_radii_every, n_interval=n_interval, workfolder=workfolder,
                                minimize=False)
            # print('Evaluating MM: %f.3' % (time.time() - start))

    def calculate_forces(self, adapt_values=False, use_NN=False):

        if use_NN:
            radii_and_charge_data = self._forcefield.Data

            pos = self._simulation.context.getState(getPositions=True).getPositions()
            pos = np.array(pos)
            pos = np.array(list(map(get_values, pos)))

            # Build Graph input
            graph = get_Graph_for_one_frame(pos, radii_and_charge_data[:, [0, 2]])
            graph.to('cpu')  # run on CPU to allow OpenMM to be on the GPU (for cluster)
            inverse_radii = np.array(self._ml_model(graph).tolist()).reshape((pos.shape[0]))
            radii = 1 / inverse_radii / 10  # in nm
            # print('Evaluating ML: %f.3' % (time.time()-start))

            self._born_calculator.adapt_coordinates(pos)
            scale = self._born_calculator.calculate_I_scale(radii)
            radii_and_charge_data[:, 1] = scale
            if adapt_values:
                self.adapt_values(radii_and_charge_data)

        forces = self._simulation.context.getState(getForces=True).getForces()
        x_forces = np.array([force[0] for force in forces._value])
        y_forces = np.array([force[1] for force in forces._value])
        z_forces = np.array([force[2] for force in forces._value])
        forces = np.array((x_forces, y_forces, z_forces))

        return forces

    def calculate_energy(self):

        energy = self._simulation.context.getState(getEnergy=True).getPotentialEnergy()

        return energy

    def calculate_dif_of_force_group(self,f1,f2):
        force1 = self._simulation.context.getState(getForces=True, groups={f1}).getForces()
        force2 = self._simulation.context.getState(getForces=True, groups={f2}).getForces() / (1-1/1.0001) * (1-1/78.5)

        x_forces = np.array([force[0] for force in force1._value])
        y_forces = np.array([force[1] for force in force1._value])
        z_forces = np.array([force[2] for force in force1._value])
        f1_forces = np.array((x_forces, y_forces, z_forces))

        x_forces = np.array([force[0] for force in force2._value])
        y_forces = np.array([force[1] for force in force2._value])
        z_forces = np.array([force[2] for force in force2._value])
        f2_forces = np.array((x_forces, y_forces, z_forces))

        return np.sqrt(mean_squared_error(f1_forces,f2_forces))

    def set_positions(self, positions):

        pos_q = to_Vec3_quant(positions)
        self._simulation.context.setPositions(pos_q)

    def __str__(self):
        return self._name

    def save_states(self,iteration=0):

        self._simulation.saveState(self._work_folder + self._pdb_id + '_' + str(self._forcefield) + '_' + str(iteration) + '.xml')
        self._iteration = iteration

    def load_states(self,iteration=0):
        self._iteration = iteration
        if iteration != 0:
            self._simulation.loadState(self._work_folder + self._pdb_id + '_' + str(self._forcefield) + '_' + str(iteration) + '.xml')
        else:
            print('state were randomly initialized')

    @property
    def pos(self):
        return self._simulation.context.getState(getPositions=True).getPositions()

    @property
    def forcefield(self):
        return self._forcefield

    @forcefield.setter
    def forcefield(self, forcefield):
        self._forcefield = forcefield
        self.update_all_properties()

    @property
    def pdb_id(self):
        return self._pdb_id

    @pdb_id.setter
    def pdb_id(self, pdb_id: str = ""):
        self._pdb_id = pdb_id
        self._datahandler.pdb_id = pdb_id
        self.update_all_properties()

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, integrator):
        integrator.setRandomNumberSeed(self._random_number_seed)
        self._integrator = integrator
        self.update_all_properties()

    @property
    def barostat(self):
        return self._barostat

    @barostat.setter
    def barostat(self, barostat):
        barostat.setRandomNumberSeed(self._random_number_seed)
        self._barostat = barostat
        self.update_all_properties()

    @property
    def platform(self):
        return self._platform

    @platform.setter
    def platform(self, platform: str = "CPU"):

        # Check for default
        if platform == "CPU":
            self._platform = platform
        else:
            # ensure that simulation is already ready
            if not self._simulation:
                warnings.warn("Simulation not setup yet; defaulting to Platform: CPU")
                self._platform = "CPU"
            else:
                # Test whether Platform is available
                self._platform = platform
                original_pos = self._simulation.context.getState(getPositions=True).getPositions()
                if original_pos[0] == original_pos[1]:
                    original_pos = None
                try:
                    self.create_simulation(original_pos)
                    self._simulation.context.getState(getForces=True).getForces()
                    print("Platform: ", platform, " ready")
                except:
                    warnings.warn("Platform not available, defaulting to CPU")
                    self._platform = "CPU"
                    self.create_simulation(original_pos)


class Explicit_water_simulator(Simulator):
    '''
    Class to simulate explicit water around a fixed solute
    '''

    def __init__(self, work_dir: str, name: str = "", pdb_id: str = "", forcefield: _generic_force_field =
    _generic_force_field(), integrator=None, platform: str = "CPU", cutoff=1 * nanometer, run_name="",
                 hdf5_file: str = None, barostat = None, starting_frame_traj = None):
        # Set variables
        self._hdf5_file = hdf5_file

        # Initialize Simulator
        super().__init__(work_dir, name, pdb_id, forcefield, integrator, platform, cutoff, run_name)

        # Build Waterbox
        # self.clean_file()
        self._num_atoms = self._datahandler.topology.getNumAtoms()
        self._datahandler._ready_for_usage = False
        self.forcefield = TIP5P_force_field()
        self.add_box()
        self._datahandler._ready_for_usage = True
        self.integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)

        # create vacuum simulation
        self._vacuum_sim = Simulator(work_dir, name, pdb_id, forcefield, integrator, platform, cutoff, run_name)
        self._vacuum_sim.forcefield = Vacuum_force_field()
        self._vacuum_sim.integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
        self._vacuum_sim.platform = 'GPU'

        # create water only simulation
        modeller = Modeller(self._datahandler.topology,self._datahandler.positions)
        for r in self._datahandler.topology.chains():
            modeller.delete([r])
            break
        top = modeller.topology

        forcefield = ForceField('tip5p.xml')
        system = forcefield.createSystem(top, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer,constraints=HBonds)
        integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
        platform = Platform.getPlatformByName('CUDA')
        platformProperties = {'Precision': 'mixed', 'CudaPrecision': 'mixed'}
        self._explicit_simulation = Simulation(topology=top,system=system,integrator=integrator,platform=platform,platformProperties=platformProperties)
        self._starting_frame_traj = starting_frame_traj

    def read_in_frame_and_set_positions(self, frame_id=0):
        if self._starting_frame_traj is None:
            self._starting_frame_traj = mdtraj.load(self._hdf5_file)
        # Load frame
        self._start_frame = self._starting_frame_traj[frame_id]

        # Set positions
        uv = self._start_frame.unitcell_vectors[0]
        a = Vec3(uv[0][0],uv[0][1],uv[0][2])
        b = Vec3(uv[1][0],uv[1][1],uv[1][2])
        c = Vec3(uv[2][0],uv[2][1],uv[2][2])
        self._simulation.context.setPeriodicBoxVectors(a,b,c)
        self.set_positions(self._start_frame.xyz[0])

    def constrain_solute(self):
        '''
        Constrain solute with: Set mass to 0, which will fix the position
        :return:
        '''

        # get all solute atoms
        atomsToRestrain = [i for i in range(self._num_atoms)]
        for i in atomsToRestrain:
            self._system.setParticleMass(i, 0)

        self.create_simulation(self._start_frame.xyz[0])

    def calculate_explicit_forces(self, n_steps, n_frames):

        forces = np.empty((n_frames, self._num_atoms, 3))
        energies = np.empty((n_frames))

        for i in tqdm.tqdm(range(n_frames)):
            self._simulation.step(n_steps)
            solvent_energy = self.calculate_solvent_energy()._value
            energies[i] = self.calculate_energy()._value - solvent_energy
            forces[i] = self.calculate_forces().T[:self._num_atoms, :]

        return energies, forces

    def add_reporter(self,interval=100):
        self._simulation.reporters.append(
                HDF5Reporter(self._work_folder + self._pdb_id + '_' + str(self._forcefield) + '_output.h5', interval))

    def calculate_vacuum_forces(self):

        self._vacuum_sim.create_simulation(self._start_frame.xyz[0, :self._num_atoms])
        energy = self._vacuum_sim.calculate_energy()._value
        forces = self._vacuum_sim.calculate_forces().T

        return energy, forces

    def calculate_solvent_energy(self):

        self._explicit_simulation.context.setPositions(self.pos[self._num_atoms:])
        solvent_energy = self._explicit_simulation.context.getState(getEnergy=True).getPotentialEnergy()
        return solvent_energy

    def calculate_mean_force_of_frame(self, n_steps, n_frames):

        # Get Energies and Forces
        energies, forces = self.calculate_explicit_forces(n_steps, n_frames)

        # Calculate averages
        mean_energy = np.mean(energies)
        mean_forces = np.mean(forces, axis=0)

        # calculate vacuum
        vacuum_energy, vacuum_forces = self.calculate_vacuum_forces()

        solvent_energy = mean_energy - vacuum_energy
        solvent_forces = mean_forces - vacuum_forces

        return solvent_energy, solvent_forces

    def calculate_mean_force_for_pre_calc_pos(self,save_location: str,save_add: str = '', frames: List = [], n_steps :int = 10, n_frames : int = 100):
        '''
        Calculate the mean force of the solvent for predifined positions
        :param hdf5_file: hdf5 file of the original explicit simulation
        :return:
        '''

        force_data = np.zeros((len(frames), self._num_atoms, 4))
        pos_data = np.zeros((len(frames), self._num_atoms, 3))

        for i in tqdm.tqdm(range(len(frames))):
            # Set position
            self.read_in_frame_and_set_positions(frames[i])

            # # Constrain Solute (should allready be constrained!!!)
            # self.constrain_solute()

            # Calculate Forces
            solvent_energy, solvent_forces = self.calculate_mean_force_of_frame(n_steps, n_frames)
            force_data[i, 0, 0] = solvent_energy
            force_data[i, :, 1:] = solvent_forces
            pos_data[i] = self._start_frame.xyz[0][:self._num_atoms,:]

        np.save(save_location + "/" + self._pdb_id + '_' + save_add + "_force_out.txt", force_data)
        np.save(save_location + "/" + self._pdb_id + '_' + save_add + "_pos_out.txt", pos_data)
        np.save(save_location + "/" + self._pdb_id + '_' + save_add + "_frames_out.txt", np.array(frames))


class Explicit_water_simulator_force_only(Explicit_water_simulator):

    def calculate_explicit_forces(self, n_steps, n_frames):

        forces = np.empty((n_frames, self._num_atoms, 3))

        for i in range(n_frames):
            self._simulation.step(n_steps)
            forces[i] = self.calculate_forces().T[:self._num_atoms, :]
            # print(self._simulation.context.getState(getPositions=True).getPeriodicBoxVectors(True))

        return 0, forces

    def calculate_vacuum_forces(self):

        self._vacuum_sim.set_positions(self._start_frame.xyz[0, :self._num_atoms])
        # self._vacuum_sim.create_simulation(self._start_frame.xyz[0, :self._num_atoms])
        forces = self._vacuum_sim.calculate_forces().T

        return 0, forces

    def calculate_mean_force_of_frame(self, n_steps, n_frames):

        # Get Energies and Forces
        _, forces = self.calculate_explicit_forces(n_steps, n_frames)

        # Calculate averages
        mean_forces = np.mean(forces, axis=0)

        # calculate vacuum
        _, vacuum_forces = self.calculate_vacuum_forces()

        solvent_forces = mean_forces - vacuum_forces

        return 0, solvent_forces

    def remove_excessive_waters(self,padding=1.25):
        '''
        This function removes all waters for which the oxygen is more than padding away from the closest atom in the system while remaining rectangular shape.
        This needs to be relaxed by energy minimization to avoid clashes based on the new periodic box vector (Hs are not checked)
        '''
        
        modeller = Modeller(self._datahandler.topology, self._simulation.context.getState(getPositions=True).getPositions())
        delete = []
        pos = self._simulation.context.getState(getPositions = True).getPositions(asNumpy=True)
        xmin = np.min(pos[:self._num_atoms,0]) - padding*nanometers
        xmax = np.max(pos[:self._num_atoms,0]) + padding*nanometers
        ymin = np.min(pos[:self._num_atoms,1]) - padding*nanometers
        ymax = np.max(pos[:self._num_atoms,1]) + padding*nanometers
        zmin = np.min(pos[:self._num_atoms,2]) - padding*nanometers
        zmax = np.max(pos[:self._num_atoms,2]) + padding*nanometers
        nres = 0

        for residue in modeller.topology.residues():
            # print(residue.name)
            if residue.name == 'HOH':
                nres += 1
                oxygen = [atom for atom in residue.atoms() if atom.element == element.oxygen][0]
                if (modeller.positions[oxygen.index][0] < xmin) or (modeller.positions[oxygen.index][0] > xmax) or (modeller.positions[oxygen.index][1] < ymin) or (modeller.positions[oxygen.index][1] > ymax) or (modeller.positions[oxygen.index][2] < zmin) or (modeller.positions[oxygen.index][2] > zmax) :
                    delete.append(residue)
        print('%i waters removed from box with %i initial waters' % (len(delete),nres))
        
        # system = self._forcefield.create_system(topology=modeller.topology,nonbondedCutoff=self._cutoff)
        from openmm.app import Topology
        # modeller.delete(delete)
        newTopology = Topology()
        vector = (to_Vec3([convert_unit_to_float(xmax-xmin) + 0.15,0,0]),to_Vec3([0,convert_unit_to_float(ymax-ymin)+ 0.15,0]),to_Vec3([0,0,convert_unit_to_float(zmax-zmin)+ 0.15]))
        if len(delete) > 0:
            newTopology.setPeriodicBoxVectors(vector)
        else:
            newTopology.setPeriodicBoxVectors(self._datahandler.topology.getPeriodicBoxVectors())
        newAtoms = {}
        newPositions = []
        for chain in self._datahandler.topology.chains():
            newChain = newTopology.addChain(chain.id)
            for residue in chain.residues():
                if residue not in delete:
                    newResidue = newTopology.addResidue(residue.name, newChain, residue.id, residue.insertionCode)
                    for atom in residue.atoms():
                        newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id)
                        string = atom.name + atom.id + str(atom.residue)
                        newAtoms[string] = newAtom
                        pos = deepcopy(modeller.positions[atom.index])
                        # shift to new zero
                        pos.x -= convert_unit_to_float(xmin)
                        pos.y -= convert_unit_to_float(ymin)
                        pos.z -= convert_unit_to_float(zmin)
                        newPositions.append(deepcopy(pos))
        for bond in self._datahandler.topology.bonds():
            string = bond[0].name + bond[0].id + str(bond[0].residue)
            string1 = bond[1].name + bond[1].id + str(bond[1].residue)
            if string in newAtoms and string1 in newAtoms:
                newTopology.addBond(newAtoms[string], newAtoms[string1])
        self._datahandler.topology = newTopology
        self._datahandler.positions = newPositions # to angstrom
        # print(self._datahandler.positions)
        self.create_system()
        # constrain again
        atomsToRestrain = [i for i in range(self._num_atoms)]
        for i in atomsToRestrain:
            self._system.setParticleMass(i, 0)

        self.create_simulation(newPositions)
        # self._simulation.minimizeEnergy()

    def calculate_mean_force_for_pre_calc_pos(self,save_location: str,save_add: str = '', frames: List = [], n_steps :int = 10, n_frames : int = 100,padding=1.25):
        '''
        Calculate the mean force of the solvent for predifined positions
        :param hdf5_file: hdf5 file of the original explicit simulation
        :return:
        '''

        force_data = np.zeros((len(frames), self._num_atoms, 4))
        pos_data = np.zeros((len(frames), self._num_atoms, 3))

        base_datahandler = deepcopy(self._datahandler)

        for i in tqdm.tqdm(range(len(frames))):
            # Set position
            self._datahandler = deepcopy(base_datahandler)
            self.create_system()
            self.create_simulation()
            self.read_in_frame_and_set_positions(frames[i])
            self.remove_excessive_waters(padding=padding)

            try:
                # Calculate Forces
                solvent_energy, solvent_forces = self.calculate_mean_force_of_frame(n_steps, n_frames)
                force_data[i, 0, 0] = solvent_energy
                force_data[i, :, 1:] = solvent_forces
                pos_data[i] = self._start_frame.xyz[0][:self._num_atoms,:]
            except:
                print('failed at it %i' % i)

        np.save(save_location + "/" + self._pdb_id + '_' + save_add + "_force_out.txt", force_data)
        np.save(save_location + "/" + self._pdb_id + '_' + save_add + "_pos_out.txt", pos_data)
        np.save(save_location + "/" + self._pdb_id + '_' + save_add + "_frames_out.txt", np.array(frames))

class Explicit_solvent_simulator_force_only(Explicit_water_simulator):

    def __init__(self, work_dir: str, name: str = "", pdb_id: str = "", forcefield: _generic_force_field = _generic_force_field(), integrator=None, platform: str = "CPU", cutoff=1 * nanometer, run_name="", hdf5_file: str = None, barostat=None,boxsize=4,save_name=None,starting_frame_traj=None,pdb=None,cache=None,create_data=True):
        self._hdf5_file = hdf5_file
        Simulator.__init__(self,work_dir,name,pdb_id,forcefield,integrator,platform,cutoff,run_name,barostat,save_name,boxlength=boxsize,create_data=create_data)
        if not create_data:
            self._datahandler._traj = starting_frame_traj[0]
            self._datahandler._is_openmm = False
        self._datahandler._solute_pdb = pdb
        *_, last = self._datahandler.topology.residues() # Get last element
        self._num_atoms = len(list(last.atoms()))
        self._datahandler._ready_for_usage = False
        self.forcefield = OpenFF_forcefield(self._pdb_id,cache=cache)
        self.integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
        self._platform = 'GPU'
        self._datahandler._ready_for_usage = True
        self.barostat = MonteCarloBarostat(1*bar, 300*kelvin)

        # create vacuum simulation
        self._vacuum_sim = Simulator(work_dir, name, pdb_id.split('_in_')[0] + '_in_v', forcefield, integrator, platform, cutoff, run_name)
        self._vacuum_sim._datahandler._solute_pdb = pdb
        self._vacuum_sim._datahandler._ready_for_usage = False
        self._vacuum_sim.forcefield = OpenFF_forcefield_vacuum(self._pdb_id,cache=cache)
        self._vacuum_sim._datahandler._ready_for_usage = True
        self._vacuum_sim._platform = 'GPU'
        self._vacuum_sim.integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
        self._starting_frame_traj = starting_frame_traj

    def calculate_vacuum_forces(self):

        self._vacuum_sim.create_simulation(self._start_frame.xyz[0, len(self._start_frame.xyz[0])-self._num_atoms:])
        forces = self._vacuum_sim.calculate_forces().T

        return 0, forces

    def calculate_mean_force_for_pre_calc_pos(self,save_location: str,save_add: str = '', frames: List = [], n_steps :int = 10, n_frames : int = 100):
        '''
        Calculate the mean force of the solvent for predifined positions
        :param hdf5_file: hdf5 file of the original explicit simulation
        :return:
        '''
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        force_data = np.zeros((len(frames), self._num_atoms, 4))
        pos_data = np.zeros((len(frames), self._num_atoms, 3))

        for i in tqdm.tqdm(range(len(frames))):
            try:
                # Set position
                self.read_in_frame_and_set_positions(frames[i])
                # Calculate Forces
                solvent_energy, solvent_forces = self.calculate_mean_force_of_frame(n_steps, n_frames)
                force_data[i, 0, 0] = solvent_energy
                force_data[i, :, 1:] = solvent_forces
                pos_data[i] = self._start_frame.xyz[0][len(self._start_frame.xyz[0])-self._num_atoms:,:]
            except:
                print('failed at %i' % i)

        np.save(save_location + "/" + self._save_name + '_' + save_add + "_force_out", force_data)
        np.save(save_location + "/" + self._save_name + '_' + save_add + "_pos_out", pos_data)
        np.save(save_location + "/" + self._save_name + '_' + save_add + "_frames_out", np.array(frames))

        pre = save_location + "/" + self._save_name + '_' + save_add

        return pre + "_force_out.npy", pre + "_pos_out.npy", pre + "_frames_out.npy"

    def constrain_solute(self):
        '''
        Constrain solute with: Set mass to 0, which will fix the position
        :return:
        '''

        # get all solute atoms
        atomsToRestrain = [i + (len(self._start_frame.xyz[0])-self._num_atoms) for i in range(self._num_atoms)]
        for i in atomsToRestrain:
            self._system.setParticleMass(i, 0)

        self.create_simulation(self._start_frame.xyz[0])

    def calculate_explicit_forces(self, n_steps, n_frames):

        forces = np.empty((n_frames, self._num_atoms, 3))

        for i in range(n_frames):
            self._simulation.step(n_steps)
            # self.run_simulation(10,1,minimize=False)
            forces[i] = self.calculate_forces().T[len(self._start_frame.xyz[0])-self._num_atoms:, :]

        return 0, forces

    def calculate_mean_force_of_frame(self, n_steps, n_frames):

        # Get Energies and Forces
        _, forces = self.calculate_explicit_forces(n_steps, n_frames)

        # Calculate averages
        mean_forces = np.mean(forces, axis=0)

        # calculate vacuum
        _, vacuum_forces = self.calculate_vacuum_forces()

        solvent_forces = mean_forces - vacuum_forces

        return 0, solvent_forces

class Multi_simulator(Simulator):

    def __init__(self, work_dir: str, name: str = "", pdb_id: str = "", forcefield: _generic_force_field = _generic_force_field(), integrator=None, platform: str = "CPU", cutoff=1 * nanometer, run_name="", barostat=None,num_rep=1,save_name=None,cache=None,random_number_seed=0):
        super().__init__(work_dir, name, pdb_id, forcefield, integrator, platform, cutoff, run_name, barostat,save_name,random_number_seed=random_number_seed)
        self._num_rep = num_rep
        self._work_dir =work_dir
        self._replicates_exist = False
        self._cache = cache
        self.create_ref_system(run_name)

    def create_ref_system(self,run_name):
        self._ref_system = Simulator(work_dir=self._work_dir,pdb_id=self._pdb_id,run_name=run_name + "ref")
        self._ref_system.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        if '_in_' in self._pdb_id:
            self._ref_system.forcefield = OpenFF_forcefield_vacuum(self._pdb_id,cache=self._cache)
        else:
            self._ref_system.forcefield = Vacuum_force_field()
    
    def setup_replicates(self):
        
        self.minimize_vacuum()
        # test whether forces are equal
        self._ref_system.create_system()
        self._ref_system.create_simulation(self._datahandler.positions)
        self._ref_system.platform = "GPU"
        ref_single_forces = self._ref_system.calculate_forces()

        # Create copies of the system
        m = Modeller(self._datahandler.topology,self._datahandler.positions)
        for i in range(self._num_rep - 1):
            m.add(self._datahandler.topology,self._datahandler.positions * nanometer)

        # Rebase and generate simulation
        self._datahandler.topology = m.topology
        self._datahandler.positions = m.positions
        self.create_system()
        for force in self._system.getForces():
            if isinstance(force,NonbondedForce):
                nbf = force

        numparticles = self._system.getNumParticles() // self._num_rep

        for custom_force in [Custom_electrostatic, Custom_lennard_jones, Custom_exception_force_with_scale]:
            cf = custom_force()
            cf.get_particles_from_existing_nonbonded_force(nbf)

            if isinstance(cf,Custom_electrostatic) or isinstance(cf,Custom_lennard_jones):
                for multi in range(self._num_rep):
                    set1 = [i for i in range(multi*numparticles,(1+multi)*numparticles)]
                    cf._force.addInteractionGroup(set1,set1)

            self._system.addForce(cf._force)

        for i, m in enumerate(self._system.getForces()):
            if isinstance(m, NonbondedForce):
                self._system.removeForce(i)
                break
        
        self.create_simulation()
        self.platform = "GPU"

        # get new forces and check
        new_forces = self.calculate_forces()
        for t in range(self._num_rep):
            diff = np.abs(ref_single_forces - new_forces[:,t*numparticles:(t+1)*numparticles])
            reldiff = np.abs(diff / ref_single_forces)
            wheres = np.argwhere(reldiff > 0.01)
            for w in wheres:
                if np.abs(diff[w[0],w[1]]) > 5:
                    print(np.abs(diff[w[0],w[1]]))
                    print(diff)
                    assert False, 'Forces are not equal'

        print('All parallel systems have the same forces as the reference System')
        self._replicates_exist = True

    def set_random_positions_for_each_replicate(self):

        smiles = self._pdb_id.split('_in_')[0]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=self._num_rep, randomSeed=self._random_number_seed)
        res = AllChem.MMFFOptimizeMoleculeConfs(mol)

        positions = []
        for i in range(self._num_rep):
            pos = mol.GetConformer(i).GetPositions()
            positions.append(pos/10)

        positions = np.array(positions)
        self.set_positions(positions.reshape(-1,positions.shape[-1]))


    def run_ref_simulation(self,n_interval,n_steps,minimize=False):
        self._ref_system.run_simulation(n_interval=n_interval,n_steps=n_steps,minimize=minimize)

    def minimize_vacuum(self):
        
        vac_sim = Simulator(work_dir=self._work_dir,pdb_id=self._pdb_id,run_name='vac_ML',random_number_seed=self._random_number_seed)
        # Create copies of the system
        vac_sim.forcefield = OpenFF_forcefield_vacuum(self._pdb_id,cache=self._cache)
        vac_sim.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        vac_sim.platform = "GPU"
        vac_sim._simulation.minimizeEnergy()
        # Set minimized pos
        pos = np.array([get_values(pos) for pos in vac_sim.pos])
        if self._replicates_exist:
            newpos = np.concatenate(tuple([pos for i in range(self._num_rep)]))
        else:
            newpos = pos
        self.set_positions(newpos)
        self._ref_system.set_positions(pos)

    def generate_model_pt_file(self,trainer,work_dir,pdb_id,trained_model,radius,fraction,model,run_model,random_seed,gbneck_radius=10.0,device='cuda'):
        trainer.explicit = True
        gbneck_parameters, _ = trainer.get_gbneck2_param(pdb_id,work_dir,cache=self._cache)
        model = model(radius=radius,max_num_neighbors=10000,parameters=gbneck_parameters,device=device,fraction=fraction)
        trainer.model = model
        trainer.load_model_dict_for_finetuning(trained_model)
        savedir = 'run_models/'
        savename =  self._save_name + '_r_' + random_seed + '_multi_' + str(self._num_rep) + '_gr_' + str(gbneck_radius) + '_run.pt'
        
        model_state_dict = trainer._model.state_dict()
        
        gnn_run = run_model(radius=radius,max_num_neighbors=1000,parameters=gbneck_parameters,device=device,fraction=fraction,jittable=True,num_reps=self._num_rep,gbneck_radius=gbneck_radius)
        gnn_run.load_state_dict(model_state_dict, strict=False)
        gnn_run.to(device)
        torch.jit.optimize_for_inference(torch.jit.script(gnn_run.eval())).save(savedir + savename)
        time.sleep(2)
        gnn_run = run_model(radius=radius,max_num_neighbors=1000,parameters=gbneck_parameters,device=device,fraction=fraction,jittable=True,num_reps=1,gbneck_radius=gbneck_radius)
        
        gnn_run.load_state_dict(model_state_dict, strict=False)
        gnn_run.to(device)

        refsavename = self._save_name + '_r_' + random_seed + '_multi_' + str(1) + '_gr_' + str(gbneck_radius) + '_run.pt'
        torch.jit.optimize_for_inference(torch.jit.script(gnn_run.eval())).save(savedir + refsavename)
        time.sleep(2)

        return savedir + savename, savedir + refsavename


def to_Vec3(xyz):
    return Vec3(x=xyz[0], y=xyz[1], z=xyz[2])


def to_Vec3_quant(pos):
    vectors = []
    for p in pos:
        vectors += [to_Vec3(p)]
    return Quantity(vectors, nanometer)


def get_values(value):
    return [convert_unit_to_float(value[0]), convert_unit_to_float(value[1]), convert_unit_to_float(value[2])]


def convert_unit_to_float(unit):
    try:
        fl = unit._value
    except:
        fl = unit

    return fl


class _custom_force:
    def __init__(self):
        self._force = None

    @property
    def openmm_force(self):
        return self._force

class Custom_electrostatic(_custom_force):

    def __init__(self):
        super().__init__()
        # Setup Energy Term
        energy_term = 'charge1 * charge2 * fourpieps / r;'

        # Create Force
        force = CustomNonbondedForce(energy_term)

        # Add Global parameters
        force.addGlobalParameter('fourpieps',138.935456)

        # Add per particle Parameters
        force.addPerParticleParameter('charge')
        force.setNonbondedMethod(CustomNonbondedForce.NoCutoff)

        self._force = force

    def add_particles(self,charges):

        # Go through charges and set the parameters
        for charge in charges:
            self._force.addParticle([charge])

    def add_exceptions(self,exceptions):

        # Go through charges and set the parameters
        for exception in exceptions:
            self._force.addExclusion(exception[0],exception[1])

    def get_particles_from_existing_nonbonded_force(self,force):

        # Get charges
        for i in range(force.getNumParticles()):
            charge, _, _ = force.getParticleParameters(i)
            self._force.addParticle([charge])

        # Get exclusions
        for i in range(force.getNumExceptions()):
            k, j,_,_,_ = force.getExceptionParameters(i)
            self._force.addExclusion(k,j)

class Custom_exception_force_with_scale(_custom_force):
    '''
    Force to compensate for exceptions in Custom_lennard_jones and Custom_electrostatic
    '''

    def __init__(self):
        super().__init__()

        energy_expression = '4 * epsilon * (sigmar6^2 - sigmar6) + chargeprod * fourpieps * (1/r);'
        energy_expression += 'sigmar6 = (sigma/r)^6;'

        force = CustomBondForce(energy_expression)

        force.addGlobalParameter('fourpieps',138.935456)

        force.addPerBondParameter('epsilon')
        force.addPerBondParameter('sigma')
        force.addPerBondParameter('chargeprod')

        self._force = force

    def get_particles_from_existing_nonbonded_force(self,force):

        for i in range(force.getNumExceptions()):
            k, j, chargeprod, sigma, epsilon = force.getExceptionParameters(i)
            sigma = sigma
            self._force.addBond(k, j,[epsilon,sigma,chargeprod])

class Custom_lennard_jones(_custom_force):

    def __init__(self):
        super().__init__()
        # Get energy term
        energy_term = '4 * epsilon * ((sigmacom/r)^12 - (sigmacom/r)^6);'
        energy_term += 'sigmacom = 0.5 * (sigma1+sigma2);'
        energy_term += 'epsilon = sqrt(epsilon1*epsilon2);'

        # Create Force
        force = CustomNonbondedForce(energy_term)

        # Add particle parameters
        force.addPerParticleParameter('epsilon')
        force.addPerParticleParameter('sigma')
        force.setNonbondedMethod(CustomNonbondedForce.NoCutoff)

        force.setUseLongRangeCorrection(False)

        self._force = force

    def get_particles_from_existing_nonbonded_force(self,force):

        # Get charges
        for i in range(force.getNumParticles()):
            _, sigma, epsilon = force.getParticleParameters(i)
            sigma = sigma
            self._force.addParticle([epsilon,sigma])

        # Get exclusions
        for i in range(force.getNumExceptions()):
            k, j,_,_,_ = force.getExceptionParameters(i)
            self._force.addExclusion(k,j)
