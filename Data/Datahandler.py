import pandas as pd
import os
import sys
from openmm.app import PDBFile, AmberPrmtopFile, AmberInpcrdFile
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from openmm import unit
import os
import openmoltools as op

from openmm.unit import nanometer, picoseconds, nanoseconds, elementary_charge, Quantity, kelvin, picosecond, \
    nanometers, kilojoules_per_mole, norm, bar
from openmm.vec3 import Vec3

import mdtraj

SPECIAL_SOLVENTS = ['TIP5P','TIP3P','TIP4P']


class DataHandler:
    '''
    Class to handle all data files
    '''

    def __init__(self,work_dir : str = "", pdb_id : str = "",solute_pdb = None,boxlength=None,padding=2,random_number_seed=0):
        '''
        Initialize Data Handler
        :param work_dir: Work Directory
        :param pdb_id: Id of Protein to work with
        '''

        self._work_folder = work_dir + "Data/data/"
        if not os.path.isdir(self._work_folder):
            os.makedirs(self._work_folder)
        self._pdb_id = pdb_id
        self._prmtop = None
        self._inpcrd = None
        self._is_openmm = True
        self._solute_starting_coordinates = None
        self._solute_pdb = solute_pdb
        self._random_number_seed = random_number_seed
        if pdb_id != "":
            if os.path.isfile(self._work_folder + self._pdb_id + ".parm7"):
                self.from_amber(self._work_folder + self._pdb_id + ".parm7",self._work_folder + self._pdb_id + ".crd")
                self._ready_for_usage = True
            elif '_in_v' in pdb_id:
                self._traj = self.create_vacuum_traj()
                self._ready_for_usage = True
                self._is_openmm = False
            elif '_in_' in pdb_id:
                if boxlength is not None:
                    self._traj = self.create_traj(boxlength=boxlength)
                else:
                    self._traj = self.create_traj(padding,use_as_padding=True)
                self._ready_for_usage = True
                self._is_openmm = False
            else:
                self._file_path = self.get_data()
                self._pdb = PDBFile(self._file_path)
                self._ready_for_usage = True
        else:
            self._file_path = ""
            self._ready_for_usage = False

    def change_box_size(self, boxlength = 4):

        self._traj = self.create_traj(boxlength)
        self._ready_for_usage = True

    def create_box_with_padding(self,padding=2):
        self._traj = self.create_traj(padding,use_as_padding=True)
        self._ready_for_usage = True

    def create_vacuum_traj(self):

        solute_smiles = self._pdb_id.split('_in_')[0]
        soluten = str(np.random.randint(100000000)) + '.pdb'
        if self._solute_pdb is None:
            solute = Chem.MolFromSmiles(solute_smiles)
            solute = Chem.AddHs(solute)
            AllChem.EmbedMolecule(solute,randomSeed=self._random_number_seed)
            AllChem.UFFOptimizeMolecule(solute)
            # Chem.rdmolfiles.MolToPDBFile(solute, self._work_folder + self._pdb_id + 'solute.pdb')
            Chem.rdmolfiles.MolToPDBFile(solute, self._work_folder + soluten)
        else:
            print('predefined pdb used')
            # with open(self._work_folder + self._pdb_id + 'solute.pdb', 'w') as f:
            #     f.write(self._solute_pdb)
            with open(self._work_folder + soluten, 'w') as f:
                f.write(self._solute_pdb)

        traj = mdtraj.load(self._work_folder + soluten)
        
        return traj

    @staticmethod
    def get_max_dif(arr):
        return np.max(arr) - np.min(arr)

    def create_traj(self, boxlength = 4, use_random=True,use_as_padding=False):

        self._boxlength = boxlength

        solute_smiles = self._pdb_id.split('_in_')[0]
        solvent_smiles = self._pdb_id.split('_in_')[1]

        solute = Chem.MolFromSmiles(solute_smiles)
        solute = Chem.AddHs(solute)
        AllChem.EmbedMolecule(solute,randomSeed=self._random_number_seed)

        
        if self._solute_starting_coordinates is None:
            AllChem.UFFOptimizeMolecule(solute)
        else:
            print('using predefined pos')
            from rdkit.Geometry import Point3D
            conf = solute.GetConformer()
            for i in range(solute.GetNumAtoms()):
                x,y,z = self._solute_starting_coordinates[i]
                conf.SetAtomPosition(i,Point3D(x,y,z))

        if use_as_padding:
            conf = solute.GetConformer()
            xyz = conf.GetPositions()

            # GET 1 nm padding
            self._boxlength = np.max([self.get_max_dif(xyz[:,i]) for i in range(3)])/10 + boxlength

        solvent = Chem.MolFromSmiles(solvent_smiles)
        solvent = Chem.AddHs(solvent)
        AllChem.EmbedMolecule(solvent,randomSeed=self._random_number_seed)
        AllChem.UFFOptimizeMolecule(solvent)

        if use_random:
            solventn = str(np.random.randint(100000000)) + '.pdb'
            soluten = str(np.random.randint(100000000)) + '.pdb'
            Chem.rdmolfiles.MolToPDBFile(solvent, self._work_folder + solventn)
            self._solventn_file = self._work_folder + solventn

            if self._solute_pdb is None:
                Chem.rdmolfiles.MolToPDBFile(solute, self._work_folder + soluten)
            else:
                with open(self._work_folder + soluten, 'w') as f:
                    f.write(self._solute_pdb)
            self._soluten_file = self._work_folder + soluten

        else:
            Chem.rdmolfiles.MolToPDBFile(solvent, self._work_folder + self._pdb_id + 'solvent.pdb')

            if self._solute_pdb is None:
                Chem.rdmolfiles.MolToPDBFile(solute, self._work_folder + self._pdb_id + 'solute.pdb')
            else:
                with open(self._work_folder + self._pdb_id + 'solute.pdb', 'w') as f:
                    f.write(self._solute_pdb)

        chcl3_density, chcl3_gmol = self.get_solvent_data(solvent_smiles)
        box_length = self._boxlength  * (unit.nano*unit.meter)
        print('using box size: ',box_length)
        num_molecules = np.ceil(chcl3_density * box_length**3 / chcl3_gmol * unit.AVOGADRO_CONSTANT_NA)

        traj = self.create_box(num_molecules)

        self._is_openmm = False

        return traj

    def create_box(self,num_molecules,use_random=True):

        # pdblist = [self._work_folder + self._pdb_id + 'solvent.pdb',self._work_folder + self._pdb_id + 'solute.pdb']
        # try:
        #     traj = op.packmol.pack_box(pdblist,[int(num_molecules),1],2,self._boxlength * 10)
        #     assert not isinstance(traj,str)
        # except:
        if use_random:
            pdblist = [self._solventn_file ,self._soluten_file]
            traj = op.packmol.pack_box(pdblist,[int(num_molecules),1],2,self._boxlength * 10)
            return traj
        else:
            solventn = str(np.random.randint(100000000)) + '.pdb'
            solventcopy = ''
            solutecopy = ''
            for s in self._work_folder + self._pdb_id + 'solvent.pdb':
                if s in ['=','(',')']:
                    solventcopy += "\\" + s
                else:
                    solventcopy += s
            for s in self._work_folder + self._pdb_id + 'solute.pdb':
                if s in ['=','(',')']:
                    solutecopy += "\\" + s
                else:
                    solutecopy += s

            os.system('cp %s %s' % (solventcopy,self._work_folder + solventn ))
            soluten = str(np.random.randint(100000000)) + '.pdb'
            os.system('cp %s %s' % (solutecopy,self._work_folder + soluten ))

            pdblist = [self._work_folder + solventn ,self._work_folder + soluten]
            traj = op.packmol.pack_box(pdblist,[int(num_molecules),1],2,self._boxlength * 10)
            return traj

    def get_solvent_data(self,smiles):
        solvent_dict = {'ClC(Cl)Cl' : (1.49 * unit.gram / (unit.centi*unit.meter)**3,119.38 * unit.gram / unit.mole),
                        'O' : (0.997 * unit.gram / (unit.centi*unit.meter)**3,18 * unit.gram / unit.mole)}
        return solvent_dict[smiles]


    def get_data(self):
        '''
        Get Data from PDB Database
        :return:
        '''
        assert(self._pdb_id != "")
        data_in_path = 'https://files.rcsb.org/download/' + self._pdb_id + ".pdb"

        # Get file
        file_path = self._work_folder + self._pdb_id + ".pdb"
        if os.path.isfile(file_path):
            file_path = file_path
        else:
            file_path = download(url=data_in_path, out=self._work_folder)
        return file_path

    def _update_all_file_paths(self):
        self._file_path = self.get_data()
        self._ready_for_usage = True
        self._pdb = PDBFile(self._file_path)

    def rebase_pdb(self):
        self._file_path = self._work_folder + self._pdb_id + "_ada.pdb"
        stdout = open(self._file_path, 'w')
        PDBFile.writeFile(topology=self.topology,positions=self.positions,file=stdout)
        stdout.close()

    def clean_file(self):
        fixer = PDBFixer(self._file_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        self._pdb.topology = fixer.topology
        self._pdb.positions = fixer.positions
        self.rebase_pdb()

    def get_protein_info(self):
        '''
        Read in protein info
        :return: pandas dataframe
        '''

        return pd.read_csv(self._work_folder + self._pdb_id + "_Info.csv")

    def change_position(self):
        pass
    
    def from_amber(self,prmtop_file,inpcrs_file):
        print('From Amber')
        self._prmtop = AmberPrmtopFile(prmtop_file)
        self._inpcrd = AmberInpcrdFile(inpcrs_file)

    @property
    def topology(self):
        if not self._prmtop is None:
            return self._prmtop.topology
        elif not self._traj is None:
            if self._is_openmm:
                top = self._traj.topology
            else:
                top = self._traj.topology.to_openmm()
            if self._traj.unitcell_vectors is None:
                return top
            length = self._traj.unitcell_vectors[0,0,0]
            dim1 = Quantity(Vec3(x=length,y=0,z=0), nanometer)
            dim2 = Quantity(Vec3(x=0,y=length,z=0), nanometer)
            dim3 = Quantity(Vec3(x=0,y=0,z=length), nanometer)
            top.setPeriodicBoxVectors([dim1,dim2,dim3])

            return top
        else:
            return self._pdb.topology
    
    @topology.setter
    def topology(self,topology):
        if not self._prmtop is None:
            self._prmtop.topology = topology
        elif not self._traj is None:
            self._is_openmm = True
            self._traj.topology = topology
        else:
            self._pdb.topology = topology

    @property
    def positions(self):
        if not self._inpcrd is None:
            return self._inpcrd.positions
        elif not self._traj is None:
            return self._traj.xyz[0]
        else:
            return self._pdb.positions

    @positions.setter
    def positions(self,positions):
        if not self._inpcrd is None:
            self._inpcrd.positions = positions
        elif not self._traj is None:
            self._traj.xyz = positions
        else:
            self._pdb.positions = positions

    @property
    def ready(self):
        return self._ready_for_usage

    @ready.setter
    def ready(self,ready):
        self._ready_for_usage = ready


    @property
    def pdb(self)->PDBFile:
        return self._pdb

    @pdb.setter
    def pdb(self,pdb):
        self._pdb = pdb
        self.rebase_pdb()

    @property
    def pdb_id(self)->str:
        return self._pdb_id

    @pdb_id.setter
    def pdb_id(self,pdb_id:str):
        self._pdb_id = pdb_id
        self._update_all_file_paths()

    @property
    def work_folder(self) -> str:
        return self._work_folder

    @work_folder.setter
    def work_folder(self, work_folder:str):
        self._work_folder = work_folder
        self._update_all_file_paths()

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def original_file_path(self) -> str:
        return self._work_folder + self._pdb_id + '.pdb'
    
import mdtraj
from rdkit import Chem
import numpy as np
import sys
import pandas as pd
import h5py
import datetime

class hdf5_storage:

    def __init__(self, filename:str,) -> None:
        self._filename = filename

        # Add timestamp if file does not exist
        if not os.path.isfile(self._filename):
            with h5py.File(self._filename, "w") as f:
                self.add_meta_data(f)
    
    def store_topology(self,topology,topology_data):
        table, bonds = topology.to_dataframe()
        topology_data.create_dataset('bonds',data=bonds)
        for c in table.columns:
            entry = table[c].values
            if entry.dtype == np.int64:
                asciiList = entry
            else:
                asciiList = [str(n).encode("utf-8", "ignore") for n in entry]
            topology_data.create_dataset(c,data=asciiList)

    def create_trajectory(self,simulation_dataset):
        top = self.create_topology_from_dataset(simulation_dataset['topology'])
        xyz = simulation_dataset['traj_xyz'][:]
        unitcell_lengths = simulation_dataset['unitcell_lengths'][:]
        unitcell_angles = simulation_dataset['unitcell_angles'][:]
        return mdtraj.Trajectory(xyz,top,unitcell_lengths=unitcell_lengths,unitcell_angles=unitcell_angles)

    def create_topology_from_dataset(self,topology_dataset):
        df = pd.DataFrame()
        for c in topology_dataset.keys():

            entry = topology_dataset[c]
            if c == 'bonds':
                bonds = entry[:]
            else:
                df[c] = entry[:]
                if entry.dtype != np.int64:
                    encoding = 'utf-8'
                    df[c] = df[c].apply(lambda x : x.decode(encoding))
        return mdtraj.Topology.from_dataframe(df,bonds)

    def create_sdf_entry(self,mol_id,mol):
        
        conf = mol.GetConformer()
        xyz = conf.GetPositions()
        try:
            chemblID = mol.GetPropsAsDict()['CHEMBL_ID']
        except:
            chemblID = '00000000'
        try:
            confID = mol.GetPropsAsDict()["CONF_ID"]
        except:
            confID = 'conf_03'
        smiles = Chem.MolToSmiles(mol)

        with h5py.File(self._filename, "a") as f:

            # Create entry for Compound
            if ("smileIDs" in f.keys()) and (mol_id in f["smileIDs"].keys()):
                smileid = f["smileIDs/" + mol_id]
                assert smileid.attrs["chemblID"] == chemblID
                assert smileid.attrs["smiles"] == smiles
            else:            
                smileid = f.create_group("smileIDs/" + mol_id)
                smileid.attrs.create("chemblID",chemblID)
                smileid.attrs.create("smiles",smiles)

            # Create entry for Starting conformation
            confid = f.create_group("smileIDs/" + mol_id + '/' + confID)
            confid.attrs.create('starting_conformation',xyz)
            confid.attrs.create('pdb',Chem.MolToPDBBlock(mol))
            confid.attrs.create("confID",confID)
            self.add_meta_data(confid)
        
        return mol_id, confID
    
    def add_meta_data(self,dataset):        
        dataset.attrs.create('timestamp',str(datetime.datetime.now()))
        git_hash = os.popen("git log --pretty=format:'%h' -n 1").read()
        dataset.attrs.create("githash",git_hash)
    
    def update_meta_data(self,dataset):        
        dataset.attrs['timestamp'] = str(datetime.datetime.now())
        git_hash = os.popen("git log --pretty=format:'%h' -n 1").read()
        dataset.attrs["githash"] = git_hash

    def create_simulation_entry_from_files(self,mol_id,confID,hdf5_file,log_file):
        traj = mdtraj.load(hdf5_file)
        logdata = pd.read_csv(log_file)
        self.create_simulation_entry(mol_id,confID,traj,logdata)
        return traj

    def create_simulation_entry(self,mol_id,confID,traj,logdata):

        with h5py.File(self._filename, "a") as f:
            # Create entry for Simulation data (log and traj)
            simulation = f.create_group("smileIDs/" + mol_id + '/' + confID + "/simulation")
            simulation.create_dataset("log",data=logdata.values)
            simulation.create_dataset("traj_xyz",data=traj.xyz)
            simulation.create_dataset("unitcell_lengths",data=traj.unitcell_lengths)
            simulation.create_dataset("unitcell_angles",data=traj.unitcell_angles)
            topology_data = simulation.create_group("topology")
            self.store_topology(traj.top,topology_data)
            self.add_meta_data(simulation)

    def create_extraction_entry(self,mol_id,confID,forces,pos,frames):
        with h5py.File(self._filename, "a") as f:
            # Create entry for extracted Forces
            extracted_forces = f.create_group("smileIDs/" + mol_id + '/' + confID +  "/extracted_forces")
            extracted_forces.create_dataset('forces',data=forces)
            extracted_forces.create_dataset('positions',data=pos)
            extracted_forces.create_dataset('frames',data=frames)
            self.add_meta_data(extracted_forces)

    def get_meta_data_for_forces(self,mol_id,confID):

        with h5py.File(self._filename, "r") as f:
            extracted_forces = f["smileIDs/" + mol_id + '/' + confID +  "/extracted_forces"]
            return extracted_forces.attrs['timestamp'],extracted_forces.attrs['githash']

    def update_extraction_entry(self,mol_id,confID,forces,pos,frames):
        with h5py.File(self._filename, "r+") as f:
            # Create entry for extracted Forces
            f["smileIDs/" + mol_id + '/' + confID +  "/extracted_forces/forces"][...] = forces
            f["smileIDs/" + mol_id + '/' + confID +  "/extracted_forces/positions"][...] = pos
            f["smileIDs/" + mol_id + '/' + confID +  "/extracted_forces/frames"][...] = frames
            self.update_meta_data(f["smileIDs/" + mol_id + '/' + confID +  "/extracted_forces/"])

    def create_reprocessed_force_entry(self,mol_id,confID,forces):
        with h5py.File(self._filename, "a") as f:
            extracted_forces = f["smileIDs/" + mol_id + '/' + confID +  "/extracted_forces"]
            extracted_forces.create_dataset('reprocessed_forces',data=forces)

    def create_reextracted_force_entry(self,mol_id,confID,forces):
        with h5py.File(self._filename, "a") as f:
            extracted_forces = f["smileIDs/" + mol_id + '/' + confID +  "/extracted_forces"]
            extracted_forces.create_dataset('reextracted_forces',data=forces)

    def create_extraction_entry_from_file(self,molid,confid,force_file,pos_file,frame_file):
        forces = np.load(force_file)
        pos = np.load(pos_file)
        frames = np.load(frame_file)
        self.create_extraction_entry(molid,confid,forces,pos,frames)

    def update_extraction_etry_from_file(self,molid,confid,force_file,pos_file,frame_file):
        forces = np.load(force_file)
        pos = np.load(pos_file)
        frames = np.load(frame_file)
        self.update_extraction_entry(molid,confid,forces,pos,frames)

    def get_starting_coordinates(self,mol_id,confID):
        print('WARNING NO GUARANTY')
        with h5py.File(self._filename, "r") as f:
            print(f["smileIDs/" + mol_id + '/'].keys())
            # Create entry for extracted Forces
            return f["smileIDs/" + mol_id + '/' + confID].attrs['starting_conformation']
    
    def get_pdb_string(self,mol_id,confID):
        with h5py.File(self._filename, "r") as f:
            # Create entry for extracted Forces
            return f["smileIDs/" + mol_id + '/' + confID].attrs['pdb']

    def get_smiles(self,mol_id):
        with h5py.File(self._filename, "r") as f:
            # Create entry for extracted Forces
            return f["smileIDs/" + mol_id].attrs['smiles']

    def get_trajectory(self,mol_id,conf_id):
        with h5py.File(self._filename, "r") as f:
            return self.create_trajectory(f["smileIDs/" + mol_id + '/' + conf_id + "/simulation"])
        
    def get_extraction(self,mol_id,conf_id):
        with h5py.File(self._filename, "r") as f:
            if "reprocessed_forces" not in f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].keys():
                forces = f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/forces"][:]
            else:
                forces = f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/reprocessed_forces"][:]
            pos = f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/positions"][:]
            frames = f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/frames"][:]
        return forces, pos, frames

    def get_reextraction(self,mol_id,conf_id,get_reprocessed=False):
        with h5py.File(self._filename, "r") as f:
            if ("reextracted_forces" not in f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].keys()) and get_reprocessed:
                forces = f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/reprocessed_forces"][:]
            else:
                forces = f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/reextracted_forces"][:]
            pos = f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/positions"][:]
            frames = f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/frames"][:]
        return forces, pos, frames

    def get_molids(self):
        with h5py.File(self._filename, "r") as f:
            keys = f["smileIDs/"].keys()
        return keys

    def get_molids_and_confids(self):
        mol_id_confid_dict = {}
        with h5py.File(self._filename, "r") as f:
            keys = f["smileIDs/"].keys()
            for key in keys:
                try:
                    mol_id_confid_dict[key] = list(f["smileIDs/%s" % key].keys())
                except Exception as e:
                    print(e)
                    pass
        
        return mol_id_confid_dict

    def get_log_data(self,mol_id,conf_id):
        with h5py.File(self._filename, "r") as f:
            return f["smileIDs/" + mol_id + '/' + conf_id + "/simulation/log"][:]

    def create_atom_feature_entry(self,mol_id,conf_id,atom_features,unique_radii):

        with h5py.File(self._filename, "a") as f:
            # Create entry for extracted Forces
            f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].create_dataset('atom_features',data=atom_features)
            f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].create_dataset('unique_radii',data=unique_radii)

    def create_vector_norm_entry(self,mol_id,conf_id,vector_norm,vector_norm_std):

        with h5py.File(self._filename, "a") as f:
            # Create entry for extracted Forces
            f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].create_dataset('vector_norm',data=vector_norm)
            f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].create_dataset('vector_norm_std',data=vector_norm_std)
    
    def get_vector_norm(self,mol_id,conf_id):

        with h5py.File(self._filename, "r") as f:
            if "vector_norm" in f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].keys():
                return f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/vector_norm"][:], f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/vector_norm_std"][:]
            else:
                return False, False

    def create_reprocessed_atom_feature_entry(self,mol_id,conf_id,atom_features,unique_radii):

        with h5py.File(self._filename, "a") as f:
            # Create entry for extracted Forces
            f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].create_dataset('reprocessed_atom_features',data=atom_features)
            f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].create_dataset('reprocessed_unique_radii',data=unique_radii)

    def get_repocessed_atom_features_and_unique_radii(self,mol_id,conf_id):
        with h5py.File(self._filename, "r") as f:
            if "reprocessed_atom_features" in f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].keys():
                return f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/reprocessed_atom_features"][:], f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/reprocessed_unique_radii"][:]
            else:
                return False, False

    def get_atom_features_and_unique_radii(self,mol_id,conf_id):
        print('WARNING check if reprocessed forces are used')
        with h5py.File(self._filename, "r") as f:
            if "atom_features" in f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].keys():
                return f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/atom_features"][:], f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces/unique_radii"][:]
            else:
                return False, False

    def get_entire_training_set(self,reprocessed_forces = True,reextracted_forces = False):
        coords_l = []
        forces_l = []
        atom_features_l = []
        smiles_l = []
        molids_l = []
        confids_l = []
        hdf5s_l = []
        uniquer_l = []
        error_files = []

        with h5py.File(self._filename, "r") as f:
            if "smileIDs" in f.keys():  
                for mol in f["smileIDs/"].keys():
                    for conf in f["smileIDs/%s" % mol].keys():
                        dataset = f["smileIDs/%s/%s" % (mol,conf)]
                        if 'extracted_forces' in dataset.keys():
                            extracted_forces = dataset['extracted_forces']
                            if 'atom_features' in extracted_forces.keys():
                                if reprocessed_forces:
                                    if ('reprocessed_forces' not in extracted_forces.keys()) or ('reprocessed_atom_features' not in extracted_forces.keys()):
                                        print('WARNING: No reprocessed forces/atom features found')
                                        print(self._filename,mol,conf)
                                        error_files.append(self._filename)
                                        continue
                                    if reextracted_forces and ('reextracted_forces' in extracted_forces.keys()):
                                        forces_l.append(extracted_forces['reextracted_forces'][:])
                                    else:
                                        forces_l.append(extracted_forces['reprocessed_forces'][:])
                                    atom_features_l.append(extracted_forces['reprocessed_atom_features'][:])
                                    uniquer_l.append(extracted_forces['reprocessed_unique_radii'][:])
                                else:
                                    forces_l.append(extracted_forces['forces'][:])
                                    atom_features_l.append(extracted_forces['atom_features'][:])
                                    uniquer_l.append(extracted_forces['unique_radii'][:])

                                coords_l.append(extracted_forces['positions'][:])
                                smiles_l.append(f["smileIDs/%s" % (mol)].attrs['smiles'])
                                molids_l.append(mol)
                                confids_l.append(conf)
                                hdf5s_l.append(self._filename)

        #write error files to file
        with open('error_files.txt','a') as f:
            for error_file in error_files:
                f.write(error_file+'\n')
        
        return coords_l, forces_l, atom_features_l, uniquer_l, smiles_l, molids_l, confids_l, hdf5s_l

    def get_existing_entries(self,mol_id,mol,check_for_reprocessed = False):
        '''
        # 1. Does SDF entry exist
        # 2. Does Simulation entry exist
        # 3. Does Extracted Forces exist
        # 4. Do Atom features exist
        '''
        sdf_exist, sim_exist, extracted_exist, atomfeatures_exist, reprocessed_exist = (False, False, False, False, False)

        smiles = Chem.MolToSmiles(mol)
        with h5py.File(self._filename, "r") as f:
            # Create entry for Compound
            if ("smileIDs" in f.keys()) and (mol_id in f["smileIDs"].keys()):
                smileid = f["smileIDs/" + mol_id]
                assert smileid.attrs["smiles"] == smiles
                sdf_exist = True
                confID = list(smileid.keys())[0]
            else:
                return sdf_exist, sim_exist, extracted_exist, atomfeatures_exist

            if "simulation" in f["smileIDs/" + mol_id + '/' + confID].keys():
                sim_exist = True
            else:
                return sdf_exist, sim_exist, extracted_exist, atomfeatures_exist
            
            if "extracted_forces" in f["smileIDs/" + mol_id + '/' + confID]:
                extracted_exist = True
            else:
                return sdf_exist, sim_exist, extracted_exist, atomfeatures_exist
            
            if "atom_features" in f["smileIDs/" + mol_id + '/' + confID +  "/extracted_forces"].keys():
                atomfeatures_exist = True
            
            if "reprocessed_forces" in f["smileIDs/" + mol_id + '/' + confID +  "/extracted_forces"].keys():
                reprocessed_exist = True
        if check_for_reprocessed:
            return sdf_exist, sim_exist, extracted_exist, atomfeatures_exist, reprocessed_exist
        else:
            return sdf_exist, sim_exist, extracted_exist, atomfeatures_exist

    def check_if_reprocessed_exists(self,mol_id,conf_id):
        with h5py.File(self._filename, "r") as f:
            if "reprocessed_forces" in f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].keys():
                return True
            else:
                return False
    
    def check_if_reprocessed_features_exist(self,mol_id,conf_id):
        with h5py.File(self._filename, "r") as f:
            if "reprocessed_atom_features" in f["smileIDs/" + mol_id + '/' + conf_id +  "/extracted_forces"].keys():
                return True
            else:
                return False

    def show_timestamp_and_hash(self):
        with h5py.File(self._filename, "r") as f:
            print('time: %s, git hash: %s' % (f.attrs['timestamp'],f.attrs['githash']))

    def show_content(self):
        with h5py.File(self._filename, "r") as f:
            for key in f["smileIDs/"].keys():
                for conf in f["smileIDs/%s" % key].keys():
                    print(key,conf)
                    for ent in f["smileIDs/%s/%s" % (key,conf)].keys():
                        print('\t\t' + ent)
    
    def show_traj(self,mol_id,conf_id,idx=-1):
        import nglview
        traj = self.get_trajectory(mol_id,conf_id)
        if idx > -1:
            indices = traj.topology.select('resid %i' % (traj.n_residues -1))
            view = nglview.show_mdtraj(traj[idx])
            view.clear_representations()
            view.add_representation('licorice',selection=indices)
            return view
        else:
            indices = traj.topology.select('resid %i' % (traj.n_residues -1))
            view = nglview.show_mdtraj(traj)
            view.clear_representations()
            view.add_representation('licorice',selection=indices)
            return view
    
    def show_forces_of_traj(self,mol_id,conf_id,force_id,view=None,forces=None,add_forces=None):
        id = force_id
        dontshow = False
        force, pos, frames = self.get_extraction(mol_id,conf_id)
        if view is None:
            dontshow = True
            view = self.show_traj(mol_id,conf_id,idx=frames[id])

        if not forces is None:
            force[id] = forces
        for aidx in range(pos[id].shape[0]):
            view.shape.add_arrow(pos[id][aidx]*10, pos[id][aidx]*10 + force[id][aidx,1:]/50, [ 1, 0, 1 ], 0.3,'%.1f' % np.linalg.norm(force[id][aidx,1:]))
        
        if not add_forces is None:
            force[id] = add_forces
            for aidx in range(pos[id].shape[0]):
                view.shape.add_arrow(pos[id][aidx]*10, pos[id][aidx]*10 + force[id][aidx,1:]/50, [ 0, 0, 1 ], 0.3)

        if not dontshow:
            return 0
        else:
            return view
    
    def show_starting_structure(self,mol_id,conf_id):
        import nglview
        pdb = self.get_pdb_string(mol_id,conf_id)
        tname = '/tmp/' + str(np.random.randint(1000000000)) + '.pdb'
        with open(tname, 'w') as f:
            f.write(pdb)
        traj = mdtraj.load(tname)
        view = nglview.show_mdtraj(traj)
        return view
