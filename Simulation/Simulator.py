import copy
import os
import sys
import time
import warnings

import mdtraj
import numpy as np
import pandas as pd
import tqdm

import sys
sys.path.append('..')

from Data.Datahandler import DataHandler
from ForceField.Forcefield import _generic_force_field, TIP5P_force_field, Vacuum_force_field
from openmm.app import Simulation, Modeller, StateDataReporter
from mdtraj.reporters import HDF5Reporter
from openmm import Platform, LangevinMiddleIntegrator
from openmm.unit import nanometer, picoseconds, nanoseconds, elementary_charge, Quantity, kelvin, picosecond, \
    nanometers, kilojoules_per_mole, norm, bar
from openmm.vec3 import Vec3

# from openmm import HarmonicBondForce, NonbondedForce, Context
# from copy import deepcopy

from time import sleep
#from MachineLearning.GNN_Trainer import Trainer
from MachineLearning.GNN_Graph import get_Graph_for_one_frame
from MachineLearning.GNN_Models import *
import torch

from openmm.app import ForceField, PME, HBonds
from sklearn.metrics import mean_squared_error

class Simulator:
    '''
    Class for running simulations.
    '''

    def __init__(self, work_dir: str, name: str = "", pdb_id: str = "", forcefield: _generic_force_field =
    _generic_force_field(), integrator=None, platform: str = "CPU", cutoff=1 * nanometer, run_name="",barostat=None):
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

        if not os.path.isdir(self._work_folder):
            os.makedirs(self._work_folder)

        self._iteration = 0
        self._name = name
        self._pdb_id = pdb_id
        self._cutoff = cutoff
        self._datahandler = DataHandler(work_dir=work_dir, pdb_id=self._pdb_id)
        self._forcefield = forcefield
        self._integrator = integrator
        self._barostat = barostat

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

    def create_system(self):
        '''
        Use Forcefield to create system
        :return:
        '''
        assert self._datahandler.ready and self._forcefield.ready
        self._system = self._forcefield.create_system(topology=self._datahandler.topology,
                                                      nonbondedCutoff=self._cutoff)
        if not self._barostat is None:
            self._system.addForce(self._barostat)

    def update_all_properties(self):
        '''
        Update all properties to match new settings. Automatically creates system and simulation if requirements are met.
        :return:
        '''
        if self._datahandler.ready and self._forcefield.ready:
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

        if not workfolder:
            workfolder = self._work_folder
        elif workfolder == "TMPDIR":
            workfolder = os.environ['TMPDIR'] + "/"

        if (minimize and self._iteration == 0):
            self._simulation.minimizeEnergy()
        if len(self._simulation.reporters) == 0:
            self._simulation.reporters.append(
                HDF5Reporter(self._work_folder + self._pdb_id + '_' + str(self._forcefield) + '_' + str(self._iteration) + '_output.h5', n_interval))
        if len(self._simulation.reporters) == 1:
            stdout = open(self._work_folder + self._pdb_id + '_' + str(self._forcefield) + str(self._iteration) + '_log.txt', "w")
            self._simulation.reporters.append(
                StateDataReporter(stdout, reportInterval=n_interval, step=True, speed=True, potentialEnergy=True,
                                  temperature=True))
        # if len(self._simulation.reporters) == 2:
        #     self._simulation.reporters.append(PDBReporter(
        #         workfolder + self._pdb_id + '_' + str(self._forcefield) + '_output.pdb', reportInterval=n_steps))
        # else:
        #     # use fast memory for pdb
        #     self._simulation.reporters[2] = PDBReporter(
        #         workfolder + self._pdb_id + '_' + str(self._forcefield) + '_output.pdb', reportInterval=n_steps)
        self._simulation.step(n_steps)

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
        self._integrator = integrator
        self.update_all_properties()

    @property
    def barostat(self):
        return self._barostat

    @barostat.setter
    def barostat(self, barostat):
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
                 hdf5_file: str = None, barostat = None):
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

    def read_in_frame_and_set_positions(self, frame_id=0):
        # Load frame
        self._start_frame = mdtraj.load_frame(self._hdf5_file, frame_id)

        # Set positions
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

        for i in range(n_frames):
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
