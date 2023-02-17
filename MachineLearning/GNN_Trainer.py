import time
import warnings

import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
import os, sys
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
import seaborn
import dill

sys.path.append('..')

try:
    from GNN_Models import *
except:
    from MachineLearning.GNN_Models import *

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import joblib
import tempfile
from functools import lru_cache
from Simulation.Simulator import Simulator
from openmm.app.internal.customgbforces import GBSAGBn2Force
from ForceField.Forcefield import Vacuum_force_field
import matplotlib as mpl

try:
    from GNN_Graph import get_Graph_for_one_frame
except:
    from MachineLearning.GNN_Graph import get_Graph_for_one_frame


class Trainer:
    def __init__(self, name='name', path='.', verbose=False, enable_tmp_dir=True, force_mode=False, device=None,random_state=161311):
        self._name = name
        self._model = None
        self._optimizer = None
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device
        self._path = path
        self._verbose = verbose
        self._random_state = random_state
        self._force_mode = force_mode

        if enable_tmp_dir == True:
            try:
                self._tmp_folder = os.environ['TMPDIR'] + "/"
            except:
                enable_tmp_dir = False
        self._use_tmpdir = enable_tmp_dir
        self._tmpdir_in_use = False
        self._explicit_data = False

        self._model_path = self._path + '/' + self._name + 'model.model'

    def set_lossfunction(self, lossfunction=None):

        if lossfunction is None:
            self.calculate_loss = self.calculate_loss_default
        else:
            self.calculate_loss = lossfunction

    def train_model2(self, runs, batch_size=100):
        assert self._optimizer != None
        assert self._model != None

        self.initialize_optimizer(lr)
        self._model.train()

        for i in range(runs):
            start = time.time()

            for j, data in enumerate(self._training_data):

                # Get DataLoader
                loader = DataLoader(data, batch_size=batch_size, shuffle=True)

                for d, ldata in enumerate(loader):
                    # set optimizer gradients
                    self._optimizer.zero_grad()

                    # sent data to device
                    ldata.to(self._device)

                    # Make prediction
                    pre_energy, pre_forces = self._model(ldata)
                    loss = self.calculate_loss(pre_energy, pre_forces, ldata)

                    # Do backpropagation
                    loss.backward()
                    self._optimizer.step()
                    del ldata, pre_energy, pre_forces, loss
                    torch.cuda.empty_cache()

                if self._verbose:
                    print(loss / 100 / 66 / 3)
                    # print("Iteration %i avg time: %f3 loss: %f3" % (j,(time.time()-start)/(j+1),np.sqrt(np.mean(loss)/(100*66*3+1))) , end = "\r")

    def calculate_loss_default(self, pre_energy, pre_forces, ldata):

        # For summed energies compare to sum of energies
        if pre_energy.size() == torch.Size([]):
            energy_loss = F.mse_loss(pre_energy, ldata.energies.sum())
        else:
            energy_loss = F.mse_loss(pre_energy.unsqueeze(1), ldata.energies)
        force_loss = F.mse_loss(pre_forces, ldata.forces)
        el_val = energy_loss.tolist()
        fl_val = force_loss.tolist()

        loss_e = (fl_val / (el_val + fl_val)) * energy_loss * 1 / 4
        loss_f = (el_val / (el_val + fl_val)) * force_loss * 3 / 4

        loss = loss_e + loss_f
        # print(el_val,fl_val)

        return loss

    def train_model(self, runs, batch_size=100):
        assert self._optimizer != None
        assert self._model != None
        self._model.train()

        # Load Training data
        Losses = []

        Val_Losses = np.empty((runs))
        for i in range(runs):
            start = time.time()
            train_losses = []
            for j, data in enumerate(self._training_data):
                # Get DataLoader
                loader = DataLoader(data, batch_size=batch_size, shuffle=True)
                # start = time.time()
                for d, ldata in enumerate(loader):
                    # print(d)

                    # set optimizer gradients
                    self._optimizer.zero_grad()

                    # sent data to device
                    ldata.to(self._device)

                    # Make prediction
                    pre_energy, pre_forces = self._model(ldata)
                    loss = self.calculate_loss(pre_energy, pre_forces, ldata)

                    # Do backpropagation
                    loss.backward()
                    self._optimizer.step()
                    
                    Losses.append(np.mean(loss.to('cpu').tolist()))
                    train_losses.append(np.mean(loss.to('cpu').tolist()))
                    del ldata, pre_energy, pre_forces
                    torch.cuda.empty_cache()
                    if self._verbose:
                        print("Iteration %i avg time: %f3 loss: %f3" % (
                        j, (time.time() - start) / (j + 1), np.mean(loss.to('cpu').tolist())), end="\r")

            Val_Losses[i] = self.validate_model(batch_size)
            print("Run %i avg time: %f3 loss: %f3" % (
            i, (time.time() - start) / (j + 1), np.sqrt(np.mean(Val_Losses[i]))), end="\n")
            self._scheduler.step()

        self.save_training_log(Losses, Val_Losses)

        return Losses, Val_Losses

    def save_training_log(self, train_loss, val_loss):

        save_path = self._path + '/' + self._name
        np.save(arr=train_loss, file=save_path + 'train_loss.txt')
        np.save(arr=val_loss, file=save_path + 'val_loss.txt')

    def validate_model(self,batch_size=64):

        # Chuck up validation data
        # chunk = 1
        # num_samples = self._validation_data.num_datasets
        validation_loss = []
        for i, data in enumerate(self._validation_data):
            loader = DataLoader(data, batch_size=batch_size)

            for l, ldata in enumerate(loader):
                ldata.to(self._device)
                pre_energy, pre_forces = self._model(ldata)
                loss = self.calculate_loss(pre_energy, pre_forces, ldata)
                validation_loss.append(loss.to('cpu').tolist())
                del ldata, pre_energy, pre_forces, loss
                torch.cuda.empty_cache()

        return np.mean(validation_loss)

    def initialize_optimizer(self, lr,schedule=None):
        assert self._model != None
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        if schedule == 'Plateau':
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, verbose=True,factor=0.8)
        elif schedule == 'Exponential':
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer,gamma=0.01**(1/1000))
        elif schedule == 'Exponential100':
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer,gamma=0.01**(1/100))
        else:
            self._scheduler = Dummy_scheduler()

    @staticmethod
    def get_gbneck2_param(pdb_id,work_dir):

        sim = Simulator(work_dir=work_dir,pdb_id=pdb_id,run_name='getparam')
        sim.forcefield = Vacuum_force_field()
        sim._system.getForces()
        topology = sim._datahandler.topology
        charges = np.array([sim._system.getForces()[3].getParticleParameters(i)[0]._value for i in range(topology._numAtoms)])
        force = GBSAGBn2Force(cutoff=None,SA=None,soluteDielectric=1)
        gbn2_parameters = np.empty((topology.getNumAtoms(),7))
        gbn2_parameters[:,0] = charges # Charges
        gbn2_parameters[:,1:6] = force.getStandardParameters(topology)
        radii = gbn2_parameters[:,1]
        uniqueRadii = list(sorted(set(radii)))
        radiusToIndex = {r: i for i, r in enumerate(uniqueRadii)}
        gbn2_parameters[:,6] = [radiusToIndex[r] for r in gbn2_parameters[:,1]]
        offset = 0.0195141
        gbn2_parameters[:,1] = gbn2_parameters[:,1] - offset
        gbn2_parameters[:, 2] = gbn2_parameters[:, 2] * gbn2_parameters[:, 1]

        return gbn2_parameters

    def prepare_training_data_for_multiple_peptides(self,peptide_id_list=[],train_test_split=0.8,work_dir='.',force_out_suffix='force_out.txt.npy',position_out_suffix='pos_out.txt.npy'):

        assert self.explicit == True

        Dataset = []
        unique_gbneckparam = []
        for pid in peptide_id_list:

            filepath = work_dir + '/Simulation/simulation/' + pid + '/'
            files = os.listdir(filepath)
            atom_features = self.get_gbneck2_param(pid,work_dir)
            unique_gbneckparam.append(atom_features)

            for file in [file for file in files if force_out_suffix in file]:
                core = file.split('force')[0]
                coord_set = np.load(filepath + core + position_out_suffix)
                force_energy_set = np.load(filepath + core + force_out_suffix)

                data = [Data(pos=torch.tensor(coord_set[i], dtype=torch.float, requires_grad=True),
                            energies=torch.tensor([force_energy_set[i][0, 0]], dtype=torch.float).unsqueeze(1),
                            forces=torch.tensor(force_energy_set[i][:, 1:], dtype=torch.float),
                            atom_features=torch.tensor(atom_features, dtype=torch.float)) for i in
                        range(len(coord_set))]
                Dataset += data

        print('Constructed Dataset with %i frames from %i compound sets' % (len(Dataset),len(peptide_id_list)))

        training_data, validation_data, test_data = self.do_split(Dataset, 1 - train_test_split)

        self._training_data = Data_Holder(training_data)
        self._validation_data = Data_Holder(validation_data)
        self._test_data = Data_Holder(test_data)
        
        return np.concatenate(unique_gbneckparam)

    def prepare_sumarized_training_data(self, train_test_split=0.8):

        # Analyse data path
        files = os.listdir(self._data_path)
        from_to_pairlist, num_frames, num_datasets = self.check_training_files(files)

        # Move training data if tmpdir is enabled
        if self._use_tmpdir and not self._tmpdir_in_use:
            copy_command = 'cp ' + self._data_path + '/* ' + self._tmp_folder

            if self._verbose:
                print('Copy Data to TMPDIR', copy_command)

            os.system(copy_command)
            self._tmpdir_in_use = True

        # Sort From to pairlist
        # from_to_pairlist = np.sort(from_to_pairlist, axis=0)

        # Generate data_functions
        if self._explicit_data:
            data_preparation_function = self.get_explicit_data_from_files
        else:
            if self._force_mode:
                data_preparation_function = self.get_force_data_from_files
            else:
                data_preparation_function = self.get_graphs_from_files

        full_data = data_preparation_function(from_to_pairlist)
        training_data, validation_data, test_data = self.do_split(full_data, 1 - train_test_split)

        self._training_data = Data_Holder(training_data)
        self._validation_data = Data_Holder(validation_data)
        self._test_data = Data_Holder(test_data)

    def prepare_training_data(self, train_test_split=0.8):

        # Analyse data path
        files = os.listdir(self._data_path)
        from_to_pairlist, num_frames, num_datasets = self.check_training_files(files)

        # Move training data if tmpdir is enabled
        if self._use_tmpdir and not self._tmpdir_in_use:
            copy_command = 'cp ' + self._data_path + '/* ' + self._tmp_folder

            if self._verbose:
                print('Copy Data to TMPDIR', copy_command)

            os.system(copy_command)
            self._tmpdir_in_use = True

        # Sort From to pairlist
        from_to_pairlist = np.sort(from_to_pairlist, axis=0)

        training_from_to, val_from_to, test_from_to = self.do_split(from_to_pairlist, 1 - train_test_split)

        # Generate data Iterators
        if self._force_mode:
            data_preparation_function = self.get_force_data_from_files
        else:
            data_preparation_function = self.get_graphs_from_files

        self._training_data = Data_Iterator(training_from_to, len(training_from_to), data_preparation_function)
        self._validation_data = Data_Iterator(val_from_to, len(val_from_to), data_preparation_function)
        if not test_from_to is None:
            self._test_data = Data_Iterator(test_from_to, len(test_from_to), data_preparation_function)

    def do_split(self, pairlist, split=0.2):

        if split != 0.0:
            train_idx, test_idx = train_test_split(np.arange(len(pairlist)), test_size=split,
                                                   random_state=self._random_state)
            return [pairlist[i] for i in train_idx], [pairlist[i] for i in test_idx], None
        else:
            return pairlist, pairlist, None

    def get_explicit_data_from_files(self, from_to_pair):

        assert self.explicit == True

        Dataset = []
        for fr_to in from_to_pair:
            fr, to = fr_to
            coord_set, force_energy_set, protein_set = self.get_from_to(fr, to)
            atom_features = np.array(protein_set[:, 11:13], dtype=float)

            # Test if single energy or atomwise
            data = [Data(pos=torch.tensor(coord_set[i], dtype=torch.float, requires_grad=True),
                         energies=torch.tensor([force_energy_set[i][0, 0]], dtype=torch.float).unsqueeze(1),
                         forces=torch.tensor(force_energy_set[i][:, 1:], dtype=torch.float),
                         atomic_features=torch.tensor(atom_features, dtype=torch.float)) for i in
                    range(len(coord_set))]
            Dataset += data
        return Dataset

    def get_force_data_from_files(self, from_to_pair):
        '''
        Get Force Data from file
        :param from_to_pair: start and end index
        :return:
        '''

        Dataset = []
        for fr_to in from_to_pair:
            fr, to = fr_to
            coord_set, force_energy_set, protein_set = self.get_from_to(fr, to)
            atom_features = np.array(protein_set[:, 11:13], dtype=float)

            # Test if single energy or atomwise
            if force_energy_set[0][1, 0] == 0:
                data = [Data(pos=torch.tensor(coord_set[i], dtype=torch.float, requires_grad=True),
                             energies=torch.tensor([force_energy_set[i][0, 0]], dtype=torch.float).unsqueeze(1),
                             forces=torch.tensor(force_energy_set[i][:, 1:], dtype=torch.float),
                             atomic_features=torch.tensor(atom_features, dtype=torch.float)) for i in
                        range(len(coord_set))]
            else:
                data = [Data(pos=torch.tensor(coord_set[i], dtype=torch.float, requires_grad=True),
                             energies=torch.tensor(force_energy_set[i][:, 0], dtype=torch.float).unsqueeze(1) * -1,
                             forces=torch.tensor(force_energy_set[i][:, 1:], dtype=torch.float) * 10,
                             atomic_features=torch.tensor(atom_features, dtype=torch.float)) for i in
                        range(len(coord_set))]
            Dataset += data
        return Dataset

    def get_graphs_from_files(self, from_to_pair):
        '''
        Get Graph for one from to pair
        :param from_to_pair: start and end index
        :return:
        '''

        Graphs = []
        for fr_to in from_to_pair:
            fr, to = fr_to
            coord_set, radii_set, protein_set = self.get_from_to(fr, to)
            graphs = [get_Graph_for_one_frame(coord, protein_set[:, 11:13].astype(float), y=radii_set[i, :]) for
                      i, coord in
                      enumerate(coord_set)]
            Graphs += graphs
        return Graphs

    def check_training_files(self, files):
        '''
        Check whether files exist for both radii and coord and return infos
        :param files:
        :return: from,to  as array and amount of frames in one file
        '''

        # Check if protein_info is available
        assert 'Protein_Info.csv' in files

        # separate

        if self.explicit:
            force_energy_files = [f for f in files if 'force_out.txt.npy' in f]
            coord_files = [f for f in files if 'pos_out.txt.npy' in f]

            # Check if all files have partners
            exist_from_to = [from_to.split('.')[0] for from_to in coord_files if
                             from_to.split('pos_out')[0] + 'force_out.txt.npy' in force_energy_files]
            print(exist_from_to)
            from_to = np.array(
                [[from_to.split('_')[0], from_to.split('_')[1].split('pos')[0]] for from_to in exist_from_to])
            from_to = from_to.astype(int)
            print(from_to)

        else:
            coord_files = [f for f in files if '.npy' in f]
            if self._force_mode:
                radii_files = [f for f in files if '_out.txt.npy' in f]

                # Check if all files have partners
                exist_from_to = [from_to.split('.')[0] for from_to in coord_files if
                                 from_to.split('.')[0] + '_out.txt.npy' in radii_files]

            else:
                radii_files = [f for f in files if '_out.txt' in f]

                # Check if all files have partners
                exist_from_to = [from_to.split('.')[0] for from_to in coord_files if
                                 from_to.split('.')[0] + '_out.txt' in radii_files]

            # Get data as from to array
            from_to = np.array([[from_to.split('_')[0], from_to.split('_')[1]] for from_to in exist_from_to])
            from_to = from_to.astype(int)

        if self._verbose:
            print('Dataset consists of %i files with %i frames resulting in %i total frames' % (
                from_to.shape[0], from_to[0, 1] - from_to[0, 0] + 1,
                (from_to[0, 1] - from_to[0, 0] + 1) * from_to.shape[0]))

        return from_to, from_to[0, 1] - from_to[0, 0] + 1, from_to.shape[0]

    def get_from_to(self, fr, to):
        '''
        Get data for one data file
        :param fr: start index
        :param to: end index
        :return: coord_data, radii_data, protein_data
        '''

        if self.explicit:
            if self._use_tmpdir:
                coord_path = self._tmp_folder + str(fr) + "_" + str(to) + "pos_out.txt.npy"
                force_energy_path = self._tmp_folder + str(fr) + "_" + str(to) + "force_out.txt.npy"
                proteinpath = self._tmp_folder + "Protein_Info.csv"
            else:
                coord_path = self._data_path + str(fr) + "_" + str(to) + "pos_out.txt.npy"
                force_energy_path = self._data_path + str(fr) + "_" + str(to) + "force_out.txt.npy"
                proteinpath = self._data_path + "Protein_Info.csv"
        else:
            if self._use_tmpdir:
                coord_path = self._tmp_folder + str(fr) + "_" + str(to) + ".npy"
                if self._force_mode:
                    force_energy_path = self._tmp_folder + str(fr) + "_" + str(to) + "_out.txt.npy"
                else:
                    force_energy_path = self._tmp_folder + str(fr) + "_" + str(to) + "_out.txt"
                proteinpath = self._tmp_folder + "Protein_Info.csv"
            else:
                coord_path = self._data_path + str(fr) + "_" + str(to) + ".npy"
                if self._force_mode:
                    force_energy_path = self._data_path + str(fr) + "_" + str(to) + "_out.txt.npy"
                else:
                    force_energy_path = self._data_path + str(fr) + "_" + str(to) + "_out.txt"
                proteinpath = self._data_path + "Protein_Info.csv"

        coord_set = np.load(coord_path)
        if self._force_mode:
            data_set = np.load(force_energy_path)
        else:
            data_set = pd.read_csv(force_energy_path, header=None)
        protein_set = pd.read_csv(proteinpath)

        if self._force_mode:
            return coord_set, data_set, protein_set.values
        else:
            return coord_set, data_set.values, protein_set.values

    def save_model(self):

        torch.save(self._model, self._path + '/' + self._name + 'model.model')

    def save_dict(self):
        self._model.eval()
        torch.save(self._model.state_dict(), self._path + '/' + self._name + 'model.dict')

    def load_dict(self):
        self._model.load_state_dict(torch.load(self._path + '/' + self._name + 'model.dict'))
        self._model.eval()

    def load_model(self, path=None):
        if path is None:
            self._model = torch.load(self._path + '/' + self._name + 'model.model')
        else:
            self._model = torch.load(path)
        self._model.eval()

    def load_model_dict_for_finetuning(self,model_path):

        pretrained_model = torch.load(model_path)
        pretrained_dict = pretrained_model.state_dict()
        # Filter out gbneck parameters
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'aggregate_information' not in k}
        state_dict = self._model.state_dict()
        state_dict.update(pretrained_dict)
        self._model.load_state_dict(state_dict)

        self._model_path = model_path


    def predict(self, data):
        data.to(self._device)
        return self._model(data)

    def plot_train_v_predicted(self, return_forces = False, max_s = None):
        
        t_forces = []
        p_forces = []
        for data in self._validation_data:
            loader = DataLoader(data, batch_size=20)

            for l, ldata in enumerate(loader):
                ldata.to(self._device)
                pre_energy, pre_forces = self._model(ldata)
                loss = self.calculate_loss(pre_energy, pre_forces, ldata)
                t_forces.append(ldata.forces.tolist())
                p_forces.append(pre_forces.tolist())
                del ldata, pre_energy, pre_forces
                torch.cuda.empty_cache()
                continue
        
        p_forces = self.transform_regged(p_forces)
        t_forces = self.transform_regged(t_forces)
        
        self.plot_forces(p_forces,t_forces,max_s)
        if return_forces:
            return t_forces, p_forces

    def plot_train_v_predicted_pos(self, return_forces = False, max_s = None,fromtopairs=[[0,10]]):
        
        t_forces = []
        p_forces = []
        distances = []
        for data in self._validation_data:
            loader = DataLoader(data, batch_size=20)

            for l, ldata in enumerate(loader):
                ldata.to(self._device)
                pre_energy, pre_forces = self._model(ldata)
                tmpdis = []
                for b in torch.unique(ldata.batch):
                    dism = torch.cdist(ldata.pos[ldata.batch==b],ldata.pos[ldata.batch==b])
                    tmpdis.append([dism[pair[0],pair[1]].tolist() for pair in fromtopairs])
                distances.append(tmpdis)
                t_forces.append(ldata.forces.tolist())
                p_forces.append(pre_forces.tolist())
                del ldata, pre_energy, pre_forces
                torch.cuda.empty_cache()
                continue
        
        p_forces = self.transform_regged(p_forces)
        t_forces = self.transform_regged(t_forces)
        
        return t_forces, p_forces, distances


    @staticmethod
    def plot_forces(pre, tru, max_s = None):

        pre = np.array(pre).flatten()
        tru = np.array(tru).flatten()
        print('RMSE: %.3f' % np.sqrt(mean_squared_error(tru,pre)))
        plt.rcParams["figure.figsize"] = (15,10)
        plt.rcParams.update({'font.size': 18})
        axd = plt.figure(constrained_layout=True).subplot_mosaic(
            [
                ["trupre",'hist']
            ],
            empty_sentinel="BLANK",
            gridspec_kw={"width_ratios":[2,1]}
        )

        hist,x,y,im = axd['trupre'].hist2d(tru,pre,bins=1000,norm=mpl.colors.LogNorm(),zorder=2)
        _ = axd['hist'].hist(np.abs(tru)-np.abs(pre),log=False,bins=1000)
        axd['hist'].set_xlabel(r'$\Delta$ Force $\left [\frac{\mathrm{kJ}}{\mathrm{mol} \cdot \mathrm{nm}} \right ]$')
        axd['hist'].set_ylabel(r'Count')
        if max_s is None:
            max_s = np.max([pre,tru])
        axd['trupre'].plot([-max_s,max_s],[-max_s,max_s],zorder=1)
        axd['trupre'].set_xlabel(r'Explicit Mean Force $\left [\frac{\mathrm{kJ}}{\mathrm{mol} \cdot \mathrm{nm}} \right ]$')
        axd['trupre'].set_ylabel(r'Predicted Mean Force $\left [\frac{\mathrm{kJ}}{\mathrm{mol} \cdot \mathrm{nm}} \right ]$')
        axd['trupre'].set_xlim([-max_s,max_s])
        axd['trupre'].set_ylim([-max_s,max_s])

        plt.colorbar(im,ax = axd['trupre'],pad = -0.98,shrink=0.3,anchor=(0.95,0.5),location='top',ticks=[1,10,100,1000,10000,100000],label=r'Count')


    def reformat_tensor_to_array(self, tensor):
        tensor = tensor.tolist()
        tensor = np.array(tensor)
        tensor = tensor.reshape(len(tensor))
        return tensor

    def prepare_run_model(self,pdbid,work_folder,run_model,radius,fraction,savedir,device='cuda',savename=None):

        gbneckparam = self.get_gbneck2_param(pdbid,work_folder)

        gnn_run = run_model(radius=radius,max_num_neighbors=256,parameters=gbneckparam,device=device,fraction=fraction,jittable=True)
        gnn_run.load_state_dict(self._model.state_dict())
        gnn_run.to(device)
        if device == 'cuda':
            gnn_run._lock_device = True
        savedir = savedir
        if savename is None:
            savename = self._model_path.split('/')[-1].split('model.model')[0] + '_run_%s.pt' % pdbid
        torch.jit.optimize_for_inference(torch.jit.script(gnn_run.eval())).save(savedir + savename)

    @property
    def explicit(self):
        return self._explicit_data

    @explicit.setter
    def explicit(self, explicit_data):
        self._explicit_data = explicit_data

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if type(model) == str:
            self._model_path = model
            model = torch.load(model)
        self._model = model
        self._model.to(self._device)

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, path):
        path += '/'
        if os.path.isdir(path):
            self._data_path = path
        elif os.path.isdir(self._path + path):
            self._data_path = self._path + path
        else:
            exit('Data Path does not exist')
    
    @staticmethod
    def transform_regged(rlist):
        count =0
        for t in rlist:
            for j in t:
                count += 1
        t_arr = np.empty((count,3))

        c=0
        for t in rlist:
            for j in t:
                t_arr[c] = j
                c += 1
        return t_arr

class Evaluator(Trainer):
    '''
    Class to analyse, training and saving options disabled
    '''

    def __init__(self, name='name', path='.', verbose=True, enable_tmp_dir=True, force_mode=False):
        super().__init__(name=name, path=path, verbose=verbose, enable_tmp_dir=enable_tmp_dir, force_mode=force_mode)

    def save_model(self):
        warnings.warn('Method disabled in Evaluation mode')

    def save_training_log(self, train_loss, val_loss):
        warnings.warn('Method disabled in Evaluation mode')

    def train_model(self, lr, runs):
        warnings.warn('Method disabled in Evaluation mode')

    def get_validation_data(self):
        if self._model_path is not None:
            val_path = self._model_path.split('model.model')[0] + 'val_loss.txt.npy'
            val_data = np.load(val_path)
            return val_data
        else:
            warnings.warn('Model Path was not provided; please use Evaluator.model = str')

    def val_for_frame(self, val_idx, frame_idx):
        '''
        Get Validation for a specific Frame
        :param val_idx:
        :param frame_idx:
        :return:
        '''

        data = self._validation_data[val_idx][frame_idx]
        tru = data.y
        data.to(self._device)
        pre = self.predict(data)

        return self.reformat_tensor_to_array(tru), self.reformat_tensor_to_array(pre)

    def plot_val_for_frame(self, val_idx, frame_idx, save_txt=''):
        tru, pre = self.val_for_frame(val_idx, frame_idx)

        p = sns.relplot(x=tru, y=pre)
        p.set_xlabels('PB inverse effective radii [1/A]')
        p.set_ylabels('predicted inverse effective radii [1/A]')
        plt.plot([0.3, 0.8], [0.3, 0.8], linewidth=1, linestyle='dashed', color='black')
        plt.text(0.3, 0.785, self._name, fontsize=16)
        plt.text(0.3, 0.75, 'RMSE: %.4f' % np.sqrt(mean_squared_error(tru, pre)), fontsize=16)
        if save_txt != '':
            plt.savefig(save_txt, dpi=300)


class Data_Holder:
    def __init__(self, data_set):
        '''
        Function to generate iterations over data points
        :param from_to_pairlist:
        :param num_datasets:
        :param graph_from_files:
        '''
        self._data_set = data_set
        self._num_datasets = 1
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx < self._num_datasets:
            self._idx += 1
            return self._data_set
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self._data_set

    @property
    def num_datasets(self):
        return self._num_datasets


class Data_Iterator:
    def __init__(self, from_to_pairlist, num_datasets, graph_from_files):
        '''
        Function to generate iterations over data points
        :param from_to_pairlist:
        :param num_datasets:
        :param graph_from_files:
        '''
        self._from_to_pairlist = from_to_pairlist
        self._num_datasets = num_datasets
        self.graph_from_files = graph_from_files
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx < self._num_datasets:
            data_pair = [self._from_to_pairlist[self._idx]]
            self._idx += 1
            data = self.graph_from_files(data_pair)
            return data
        else:
            raise StopIteration

    def __getitem__(self, item):
        data_pair = self._from_to_pairlist[item]
        if len(data_pair.shape) == 1:
            data_pair = [data_pair]
        return self.graph_from_files(data_pair)

    @property
    def num_datasets(self):
        return self._num_datasets

class Dummy_scheduler:
    def step(self,validation_loss):
        pass
