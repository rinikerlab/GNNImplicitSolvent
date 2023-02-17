import pandas as pd
import os
import sys
from openmm.app import PDBFile, AmberPrmtopFile, AmberInpcrdFile
from pdbfixer import PDBFixer

import mdtraj

class DataHandler:
    '''
    Class to handle all data files
    '''

    def __init__(self,work_dir : str = "", pdb_id : str = ""):
        '''
        Initialize Data Handler
        :param work_dir: Work Directory
        :param pdb_id: Id of Protein to work with
        '''

        self._work_folder = work_dir + "Data/data/"
        if not os.path.isdir(self._work_folder):
            os.mkdir(self._work_folder)
        self._pdb_id = pdb_id
        self._prmtop = None
        self._inpcrd = None

        if pdb_id != "":
            if os.path.isfile(self._work_folder + self._pdb_id + ".parm7"):
                self.from_amber(self._work_folder + self._pdb_id + ".parm7",self._work_folder + self._pdb_id + ".crd")
                self._ready_for_usage = True
            else:
                self._file_path = self.get_data()
                self._pdb = PDBFile(self._file_path)
                self._ready_for_usage = True
        else:
            self._file_path = ""
            self._ready_for_usage = False

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
        else:
            return self._pdb.topology
    
    @topology.setter
    def topology(self,topology):
        if self._prmtop is None:
            self._pdb.topology = topology
        else:
            self._prmtop.topology = topology

    @property
    def positions(self):
        if not self._inpcrd is None:
            return self._inpcrd.positions
        else:
            return self._pdb.positions

    @positions.setter
    def positions(self,positions):
        if not self._inpcrd is None:
            self._inpcrd.positions = positions
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