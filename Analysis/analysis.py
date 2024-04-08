"""
File: analysis of simulations

Description:
    Analyse simulations

Author: Paul Katzberger
Date: 03.04.2023
"""

import time
import warnings
from scipy.stats import wasserstein_distance
from numba import njit
from multiprocessing import Pool

from multiprocessing import Pool
import nglview as nj
from numba import int32, float32
from numba import prange, njit
import numpy as np
import os
from sys import exit
import pandas as pd
import seaborn as sns

# Modules for handling trajectories
import mdtraj
from mdtraj.geometry import compute_distances
from itertools import combinations

# PYEMMA
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt

# Ploting
import matplotlib as mtl
import matplotlib.pyplot as plt
from copy import deepcopy
from mdtraj import shrake_rupley, compute_rg
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from skimage.measure import EllipseModel
from sklearn.decomposition import PCA
from typing import DefaultDict
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

class Trajectory_Analysis:
    '''
    Class to analyse trajectories
    '''

    def __init__(self, workdir: str = "", id=-1):
        '''
        ID : to keep files on disks and having multiple instances!
        '''
        self._id = str(id)
        self._workdir = workdir
        self._add_trajectories = None
        self._tica_mapper = None
        self._view = None

    def convert_pdb_to_h5(self, pdb_infile: str, outfile: str = "", topology_file: str = ""):
        if outfile == "":
            outfile = pdb_infile.split(".")[0] + ".h5"

        commandline = "mdconvert " + pdb_infile + " -o " + outfile

        if topology_file != "":
            commandline += " -t " + topology_file

        # print(commandline)
        exit_code = os.system(commandline)
        assert exit_code == 0

        return outfile

    def read_trajectory(self, infile: str, atom_indices_to_load=None):
        if infile.split(".")[1] != "h5":
            exit("Only hdf5 files are allowed, please convert first")

        # top
        self._trajectory = mdtraj.load_hdf5(infile, atom_indices=atom_indices_to_load)

    def read_additional_trajectory(self, infile: str):
        if infile.split(".")[1] != "h5":
            exit("Only hdf5 files are allowed, please convert first")

        self._add_trajectories.append(mdtraj.load_hdf5(infile))

    def get_features(self, kind="alpha", pb=True):

        kind_atoms = self._trajectory.topology.select_atom_indices(kind)
        atom_pairs = list(combinations(kind_atoms, 2))
        self._features = compute_distances(traj=self._trajectory, atom_pairs=atom_pairs, periodic=pb)

        return 0

    def get_features_of_add_trajectories(self, kind, traj_id=0):
        kind_atoms = self._trajectory.topology.select_atom_indices(kind)
        atom_pairs = list(combinations(kind_atoms, 2))
        self._add_features = compute_distances(traj=self._add_trajectories[traj_id], atom_pairs=atom_pairs)

        return 0

    def build_mapper(self, n_neighbors=15, min_dist=1):
        self._mapper = umap.UMAP(low_memory=True, n_neighbors=n_neighbors, min_dist=min_dist, metric='manhattan',
                                 output_metric='manhattan', ).fit(self._features)

    def plot_UMAP(self, n_frames=100000):
        plot = umap_plot.points(self._mapper, values=np.array([i for i in range(n_frames)]))
        return plot

    def plot_connectivity(self):
        plot = umap_plot.connectivity(self._mapper, show_points=True)
        return plot

    def plot_interactive(self):
        hover_data = pd.DataFrame({'index': np.arange(len(self._features))})
        plot = umap_plot.interactive(self._mapper, hover_data=hover_data, point_size=2)
        return umap_plot.show(plot)

    def plot_histogram(self, kind="UMAP"):
        if kind == "UMAP":
            embedding = self._mapper.transform(self._features)
            x = embedding[:, 0]
            y = embedding[:, 1]
        elif kind == "TICA":
            x = np.vstack(self._tica_out)[:, 0]
            y = np.vstack(self._tica_out)[:, 1]

        plot = sns.JointGrid(x=x, y=y, space=0)
        plot.plot_joint(sns.kdeplot,
                        fill=True,
                        thresh=0, levels=100, cmap="rocket")
        plot.plot_marginals(sns.histplot, alpha=1, color="black", bins=25, kde=True)
        return plot

    def plot_histogram_of_add_traj(self):
        embedding = self._mapper.transform(self._add_features)
        x = embedding[:, 0]
        y = embedding[:, 1]
        plot = sns.JointGrid(x=x, y=y, space=0)
        plot.plot_joint(sns.kdeplot,
                        fill=True,
                        thresh=0, levels=100, cmap="rocket")
        plot.plot_marginals(sns.histplot, alpha=1, color="black", bins=25, kde=True)
        return plot

    def generate_TICA_features(self, kind="alpha"):

        self._tica_features = coor.featurizer(self._trajectory.topology)
        if kind == "alpha":
            self._tica_features.add_distances(self._tica_features.select_Ca())
        elif kind == "angle":
            self._tica_features.add_backbone_torsions()
        elif kind == "ca_angles":
            self._tica_features.add_angles(self._tica_features.select_Ca())

    def generate_TICA_input(self, kind="alpha"):
        if self._id == '-1':
            exit('id needs to be unique')
        self.generate_TICA_features(kind)
        self._trajectory.save_xtc("/tmp/" + self._id + "tmp.xtc")
        self._tica_in = coor.source("/tmp/" + self._id + "tmp.xtc", self._tica_features)

    def generate_tica_input_from_multiple_files(self, filenames, atom_indices_to_load=None, load_every_frame=1):
        if self._id == '-1':
            exit('id needs to be unique')
        # Create Folder to store xtcs
        folder = "/tmp/" + self._id + "/"
        if not os.path.isdir(folder):
            os.makedirs(folder)

        Files = []

        for f, filename in enumerate(filenames):
            try:
                self.read_trajectory(filename, atom_indices_to_load)
                self._trajectory[::load_every_frame].save_xtc(folder + str(f) + "tmp.xtc")
                print(self._trajectory[::load_every_frame])
                Files.append(folder + str(f) + "tmp.xtc")
            except:
                print("Could not load: ", filename)

        self.generate_TICA_features()
        self._tica_in = coor.source(Files, self._tica_features)

    def generate_tica_input_from_multiple_trajectories(self, trajectory_list):
        
        folder = "/tmp/" + self._id + "/"
        if not os.path.isdir(folder):
            os.makedirs(folder)

        Files = []

        for f, traj in enumerate(trajectory_list):
            traj.save_xtc(folder + str(f) + "tmp.xtc")
            Files.append(folder + str(f) + "tmp.xtc")

        # save last to trajectory
        self._trajectory = traj

        self.generate_TICA_features()
        self._tica_in = coor.source(Files, self._tica_features)

    def write_out_tica_outs(self, filename=None):
        if self._id == '-1':
            exit('id needs to be unique')
        if filename == None:
            folder = "/tmp/" + self._id + "/"
            if not os.path.isdir(folder):
                os.makedirs(folder)
            filename = folder + "coor.txt"

    def get_tica_values_of_pretrained_tica(self):
        self._tica_out = self._tica_mapper.transform(self._tica_in.get_output())

    def get_tica_multiple(self, tica_in_ops=[]):
        self._tica_in = tica_in_ops

    def get_tica_multiple_xcs(self, folder: str, pdb_infile: str = ""):
        '''
        Function to extract presampled trajectories from a folder
        '''

        folder += "/"

        content = os.listdir(folder)
        if pdb_infile == "":
            for cont in content:
                if ".pdb" in cont:
                    pdb_infile = folder + cont
                    break
        if pdb_infile == "":
            exit("No pdb file was provided and none where found in the folder")

        # Find pdb and extract features
        self._tica_features = coor.featurizer(pdb_infile)
        self._tica_features.add_distances(self._tica_features.select_Ca())

        # explore content and extract
        Files = []
        for cont in content:
            path = folder + cont
            if os.path.isdir(path):
                path_cont = os.listdir(path)
                for pc in path_cont:
                    if os.path.isfile(path + "/" + pc):
                        if pc[0] != ".":
                            Files.append(path + "/" + pc)

        self._tica_in = coor.source(Files, self._tica_features)

    def get_tica(self, lag=500):
        self._tica_mapper = coor.tica(self._tica_in, lag=lag, dim=4, kinetic_map=True)
        self._tica_out = self._tica_mapper.get_output()  # get tica coordinates

    def plot_tica(self, ax=None, vmin=None, vmax=None):
        if not ax is None:
            ax = plt.gca()
        # ax.text(-2, -4.7, '1', fontsize=20, color='black')
        # ax.text(-1.2, -5, '2', fontsize=20, color='black')
        # ax.text(-2.5, 1.5, '3', fontsize=20, color='black')
        # ax.text(-2.5, 3, '4', fontsize=20, color='white')
        # ax.set_xlim(left=-2.5, right=2)
        # ax.set_ylim(bottom=-3, top=4)
        mplt.plot_free_energy(np.vstack(self._tica_out)[:, 0], np.vstack(self._tica_out)[:, 1], ax=ax, vmin=vmin,
                              vmax=vmax)

    def get_pca(self):
        self._pca_mapper = coor.pca(self._tica_in)
        self._pca_out = self._pca_mapper.get_output()  # get tica coordinates

    def plot_pca(self):
        mplt.plot_free_energy(np.vstack(self._pca_out)[:, 0], np.vstack(self._pca_out)[:, 1])

    def plot_tica_components(self):
        Y = self._tica_out
        plt.rcParams.update({'font.size': 14})
        dt = 0.002
        plt.figure(figsize=(8, 5))
        ax1 = plt.subplot(311)
        x = dt * np.arange(Y[0].shape[0])
        plt.plot(x, Y[0][:, 0]);
        plt.ylabel('IC 1');
        plt.xticks([]);
        plt.yticks(np.arange(-2, 4, 2))
        ax1 = plt.subplot(312)
        plt.plot(x, Y[0][:, 1]);
        plt.ylabel('IC 2');
        plt.xticks([]);
        plt.yticks(np.arange(-3.5, 4, 2))
        ax1 = plt.subplot(313)
        plt.plot(x, Y[0][:, 2]);
        plt.xlabel('time / ns');
        plt.ylabel('IC 3');
        plt.yticks(np.arange(-2, 2, 2));
        return plt

    def do_tica_clustering(self, n_clusters=250):

        self._clustering = coor.cluster_kmeans(self._tica_out, k=n_clusters,tolerance=0.001,max_iter=20)

    def plot_clustering(self):
        mplt.plot_free_energy(np.vstack(self._tica_out)[:, 0], np.vstack(self._tica_out)[:, 1])
        cc_x = self._clustering.clustercenters[:, 0]
        cc_y = self._clustering.clustercenters[:, 1]
        plt.plot(cc_x, cc_y, linewidth=0, marker='o', markersize=5, color='black')

    def get_cluster_output(self):
        self._clustering_output = self._clustering.get_output()

    def find_cluster(self, x, y, exclusion_list=[]):
        '''
        Find Cluster based on euclidian distance
        :param x:
        :param y:
        :return:
        '''
        cc_x = self._clustering.clustercenters[:, 0]
        cc_y = self._clustering.clustercenters[:, 1]

        closest = 0
        closest_distance = 999999
        for i in range(len(cc_x)):
            distance = (cc_x[i] - x) ** 2 + (cc_y[i] - y) ** 2
            if distance < closest_distance and i not in exclusion_list:
                closest = i
                closest_distance = distance
        return closest

    def get_frame_for_cluster(self, cluster_id):

        for traj_id, traj in enumerate(self._clustering_output):
            for frame_id, frame in enumerate(traj):
                if frame == cluster_id:
                    return traj_id, frame_id

    def get_frame_from_file_folder(self, folder: str, pdb_infile: str = "", traj_id=0, frame_id=0):
        '''
        Function to extract presampled trajectories from a folder
        '''

        folder += "/"

        content = os.listdir(folder)
        if pdb_infile == "":
            for cont in content:
                if ".pdb" in cont:
                    pdb_infile = folder + cont
                    break
        if pdb_infile == "":
            exit("No pdb file was provided and none where found in the folder")

        # explore content and extract
        Files = []
        for cont in content:
            path = folder + cont
            if os.path.isdir(path):
                path_cont = os.listdir(path)
                for pc in path_cont:
                    if os.path.isfile(path + "/" + pc):
                        if pc[0] != ".":
                            Files.append(path + "/" + pc)

        file = Files[traj_id]
        pdb = mdtraj.load_pdb(pdb_infile)
        traj = mdtraj.load(file, top=pdb.topology)

        return traj[frame_id]

    def perform_msm(self, lags=200, nits=10):
        self._msm_its = msm.timescales_msm(self._clustering.dtrajs, lags=lags, nits=nits)

    def plot_msm_comp(self):
        mtl.rcParams.update({'font.size': 14})
        mplt.plot_implied_timescales(self._msm_its, ylog=False, units='steps', linewidth=2)

    def estimate_markov_model(self, msm_lag=12):
        self._msm_M = msm.estimate_markov_model(self._clustering.dtrajs, msm_lag)
        print('fraction of states used = ', self._msm_M.active_state_fraction)
        print('fraction of counts used = ', self._msm_M.active_count_fraction)

    def test_msm(self, msm_lag=12):
        self._msm_M = msm.bayesian_markov_model(self._clustering.dtrajs, msm_lag)
        self._msm_ck = self._msm_M.cktest(4, mlags=11, err_est=False)

    def plot_test_msm(self):
        mtl.rcParams.update({'font.size': 14})
        mplt.plot_cktest(self._msm_ck, diag=True, figsize=(7, 7), layout=(2, 2), padding_top=0.1, y01=False,
                         padding_between=0.3,
                         dt=0.002, units='ns')

    def plot_msm_weighted_free(self):
        xall = np.vstack(self._tica_out)[:, 0]
        yall = np.vstack(self._tica_out)[:, 1]
        W = np.concatenate(self._msm_M.trajectory_weights())
        mtl.rcParams.update({'font.size': 12})
        mplt.plot_free_energy(xall, yall, weights=W, vmax=10.5, )
        ax = plt.gca()
        ax.set_xlim(-2, 1.8)
        ax.set_ylim(-2, 3.5)

    def calculate_radius_of_gyration(self, n_radii=-1, every=1):

        if n_radii == -1:
            n_radii = len(self._trajectory)
            every = 1

        self._radii_of_gyration = np.empty((n_radii))
        for i in range(n_radii):
            self._radii_of_gyration[i] = compute_rg(self._trajectory[i * every])

    def visualize_radius_of_gyration(self):

        plot_data = pd.DataFrame()
        plot_data["radius of Gyration"] = self._radii_of_gyration
        plot_data["Time [ns]"] = [0.002 * i for i in range(len(self._radii_of_gyration))]

        g = sns.relplot(
            data=plot_data,
            x="Time [ns]", y="radius of Gyration",
            kind="line", palette="rocket",
            height=10, aspect=1.5, facet_kws=dict(sharex=False)
        )
        g.set_xlabels("Simulation Time [ns]", fontsize=20)
        g.set_ylabels("Radius of Gyration", fontsize=20)
        # g.set(ylim=(0.05, 0.3))
        # g.savefig("/home/kpaul/implicitml/Data/Graphs/" + "test.png", dpi=300)
        return g

    def calculate_sasa(self, n_area=-1, every=1):
        '''
        Calculate the Surface acessible surface area using the Shake Ruply algorithm
        :param n_area: number or areas to calculate
        :param every: Take trajectory of di
        :return:
        '''

        if n_area == -1:
            n_area = len(self._trajectory)
            every = 1

        self._sasa = np.empty((n_area))
        self._sasa = Parallel(n_jobs=8)(delayed(calc_sasa_radius)(self._trajectory[i * every]) for i in range(n_area))

    def visualize_sasa(self, every=1):

        plot_data = pd.DataFrame()
        plot_data["Sasa"] = self._sasa
        plot_data["Time [ns]"] = [0.002 * i * every for i in range(len(self._sasa))]

        g = sns.relplot(
            data=plot_data,
            x="Time [ns]", y="Sasa",
            kind="line", palette="rocket",
            height=10, aspect=1.5, facet_kws=dict(sharex=False)
        )
        g.set_xlabels("Simulation Time [ns]", fontsize=20)
        g.set_ylabels("Surface accessible Surface Area", fontsize=20)
        return g

    def perform_PMF_analysis(self, infile, dis_from, dis_to, num_atoms=None, plot=False):
        '''
        Function to analyse PMF with respect to a reaction coordinate in this case a distance between atoms
        :param infile: Trajectory infile
        :param dis_from: Starting id
        :param dis_to: End id
        :return:
        '''

        # Read Trajectory
        if num_atoms == None:
            self.read_trajectory(infile)
        else:
            self.read_trajectory(infile, atom_indices_to_load=[i for i in range(num_atoms)])
        self._trajectory.save_xtc(self.xtc)

        # Get Features
        self.generate_TICA_features("None")
        self._tica_features.add_distances([dis_from, dis_to])

        # Get Distance Output
        self._dis_data = coor.source(self.xtc, self._tica_features).get_output()[0]

        # Convert to angstrom
        self._dis_data = [d[0] * 10 for d in self._dis_data]

        self._PMF_energies, self._PMF_positions = get_energies_from_distances(self._dis_data, plot=plot)

        return self._PMF_energies, self._PMF_positions

    def calculate_eccentricity_one_frame(self,
                                         ring_indices=[7, 8, 10, 11, 12, 13, 14, 28, 29, 30, 38, 39, 40, 48, 49, 52, 53,
                                                       54], index = 0):

        return _get_eccentricity(self._trajectory[index].xyz[0], indices=ring_indices, plot_ell=False,
                                                                     get_area=True)

    def calculate_eccentricity(self,
                               ring_indices=[7, 8, 10, 11, 12, 13, 14, 28, 29, 30, 38, 39, 40, 48, 49, 52, 53, 54]):
        self._eccentricity = np.empty((len(self._trajectory)))
        self._area = np.empty((len(self._trajectory)))

        for t, traj in tqdm(enumerate(self._trajectory)):
            self._eccentricity[t], self._area[t] = _get_eccentricity(traj.xyz[0], indices=ring_indices, plot_ell=False,
                                                                     get_area=True)

    @property
    def tica_mapper(self):
        return deepcopy(self._tica_mapper)

    @tica_mapper.setter
    def tica_mapper(self, tica_mapper):
        self._tica_mapper = tica_mapper

    @property
    def mapper(self):
        return self._mapper

    @mapper.setter
    def mapper(self, mapper):
        self._mapper = mapper

    @property
    def trajectory(self):
        return self._trajectory

    @trajectory.setter
    def trajectory(self, input):
        if type(input) == str:
            self.read_trajectory(input)
        elif type(input) == mdtraj.Trajectory:
            self._trajectory = input
        else:
            exit("Only filnames or mdtraj.Trajectories are allowed")

    @property
    def xtc(self):
        if self._id == '-1':
            exit('id needs to be unique')
        return "/tmp/" + self._id + "tmp.xtc"

    @property
    def view(self):

        if self._view is None:
            self._view = nj.show_mdtraj(self._trajectory)
            self._view.clear_representations()
            self._view.add_representation("licorice", selection="protein")
            self._view.add_representation("licorice", selection=[i for i in range(77)])
            # self._view.add_representation("distance", atomPair=[14, 54])
            # self._view.add_contact(hydrogenBond=True)
            # self._view.add_distance(atom_pair=[[14, 54]], label_color="black")
            self._view.center()

        return self._view

    def add_water_to_view(self, water_id=0,n_atoms=66,n_residues=7):
        self._view.add_representation("licorice", selection=[n_atoms + (water_id - n_residues) * 5 + i for i in range(5)])

    def visualize_shell(self,n_solute_atoms=66,label_distance=False,n_residues=7):

        iswat = lambda x: x >= n_solute_atoms

        tatip5p_water_traj = self._trajectory[self.view.frame]

        hbonds = mdtraj.baker_hubbard(tatip5p_water_traj, periodic=True, exclude_water=False)

        for hbond in hbonds:
            # only consider water solute
            if iswat(hbond[0]) != iswat(hbond[2]):
                # do labeling
                if iswat(hbond[0]):
                    watid = int(str(tatip5p_water_traj.topology.atom(hbond[0])).split('OH')[1].split('-')[0])
                    self.add_water_to_view(watid,n_solute_atoms,n_residues)
                else:
                    watid = int(str(tatip5p_water_traj.topology.atom(hbond[2])).split('OH')[1].split('-')[0])
                    self.add_water_to_view(watid,n_solute_atoms,n_residues)
                if label_distance:
                    self.view.add_distance(atom_pair=[[hbond[0], hbond[2]]], label_color="white")

class PeptideAnalyzer:

    def __init__(self,folder,pdbid,autoprepare = False,reprocess = True):

        self._folder = folder
        self._pdbid = pdbid
        self._traj_dict = DefaultDict(lambda : {})
        self._all_z = DefaultDict(lambda: DefaultDict(lambda: {}))
        self._reprocess = reprocess
        self._multiprocessing = False

        self.initialize_plt()
        if autoprepare:
            self._autoprepare()
        
        self.create_dicts()

    def create_dicts(self):
        variations = ['VAL','LEU','ILE','PHE','SER','THR','TYR','PRO','','ALA ALA','ALA']
        letters = ['v','l','i','f','s','t','y','p','','aa','a']
        self._vardict = {}
        for l,let in enumerate(letters):
            self._vardict['ka%sae' % let] = variations[l]
        

    def savefig(self,figname):
        plt.savefig(figname)
    
    def initialize_plt(self,figsize = (15,10),fontsize = 22):
        # initialice Matplotlib
        _=plt.figure()
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams.update({'font.size': fontsize})
        plt.rcParams.update({'font.family':'Sans'})

    def _autoprepare(self,gnn_core='_pub_0601'):


        if gnn_core == '_pub_0601':
            trainon_dict = {'kasae':['vlifp','vliftyp'],
            'katae':['vlifp','vlifsyp'],
            'kayae':['vlifp','vlifstp'],
            'kapae':['sty','vlifsty'],
            'kaaaae':['vlifstyp'],
            'kaae':['vlifstyp'],
            'kaaae':['vlifstyp'],
            'kavae':['sty'],
            'kaiae':['sty'],
            'kafae':['sty'],
            'kalae':['sty']}
        else:
            trainon_dict = {'kasae':['vliftyp'],
            'katae':['vlifsyp'],
            'kayae':['vlifstp']}


        # register Trajectories
        for trainon in trainon_dict[self._pdbid]:
            try:
                self.register_gnn_traj(self._folder,[trainon],10,['1','2','161311'],gnn_core)
            except:
                print(trainon, ' not available')
        self.register_explicit_traj('TIP5P',folder=self._folder)
        try:
            self.register_explicit_traj('TIP3P',folder=self._folder)
        except:
            pass
        self.register_traj('GBNeck2','GB_Neck2_0_output',folder=self._folder)

        # Do processing pipeline
        self.process_trajectories()
        self.detect_key_atoms()
        self.calculate_distances_for_trajs()
        self.calculate_histogram_data()
        self.calculate_colormaps()


    def get_distance_by_name(self,tkey,n1,n2):

        for a,atom in enumerate(self._traj_dict[tkey]['TA'].trajectory.top.atoms):
            if n1 == str(atom):
                n1_id = a
            if n2 == str(atom):
                n2_id = a

        return self.calculate_distance(self.tdict[tkey]['TA'].trajectory.xyz,[n1_id , n2_id])[:,0]

    @staticmethod
    def map_keys_and_plot(tkeys,plot_functions,height_ratios=None,width_ratios=None,return_caption=True,filename=None,filedir=None,caption_loc=None):
        """Plot each key into the same plot this is a static function as one can combine functions from different instances

        Args:
            tkeys (list): List of keys
            plot_functions (PeptideAnalyzer.PLOTFUNCTION): Funktions to use for plotting 2D list
        """

        if height_ratios is None:
            height_ratios = [0.1] + [1 for l in range(len(plot_functions)-1)]

        # convert to string
        axd_strs = [list(map(str, x)) for x in plot_functions]

        axd = plt.figure(constrained_layout=True).subplot_mosaic(
            axd_strs,
            empty_sentinel="BLANK",
            gridspec_kw={"width_ratios": width_ratios,"height_ratios":height_ratios}
        )

        # Initialice Figure caption
        cur_figure_caption = DefaultDict(lambda : [])

        # Reshape flet to get unique
        ll = [i for s in plot_functions for i in s]
        unique_plot_functions = list(set(ll))
        for p in unique_plot_functions:
            for key in tkeys:
                caption = p(key,axd[str(p)])
                cur_figure_caption[str(p)].append(caption)
        caption = PeptideAnalyzer.create_caption(cur_figure_caption,axd_strs,axd,filename,caption_loc)

        if not filename is None:
            plt.savefig('%s/%s' % (filedir,filename))

        if return_caption:
            return caption

    @staticmethod
    def create_caption(caption_dict,axd_position,axd,filename=None,caption_loc=None):

        labeling = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
        text = []
        i=0
        for ax in axd_position:
            for f in ax:
                if not caption_dict[f][0] is None:
                    if not caption_dict[f][0][1] in text:
                        text.append(caption_dict[f][0][1])
                        if caption_loc is None:
                            axd[f].text(0.025,0.925,r'%s' % labeling[i],horizontalalignment='left',verticalalignment='top', transform=axd[f].transAxes,weight='bold')
                        else:
                            if caption_loc[i] == 0:
                                axd[f].text(0.025,0.925,r'%s' % labeling[i],horizontalalignment='left',verticalalignment='top', transform=axd[f].transAxes,weight='bold')
                            else:
                                axd[f].text(0.975,0.9,r'%s' % labeling[i],horizontalalignment='right',verticalalignment='top', transform=axd[f].transAxes,weight='bold')


                        i+= 1
        
        caption_str = 'Comparison of solvent models; '
        for t,te in enumerate(text):
            caption_str += labeling[t] + ') ' + te + ' '
        caption_str = caption_str[:-1]
        caption_str += '.'
        
        figure_str = '''
        \\begin{figure}[t]
            \centering
            \includegraphics[width=\columnwidth]{Graphs/%s}
            \caption{%s}
            \label{fig:%s}
        \end{figure}
        ''' % (filename,caption_str,filename.split('.')[0])
        if filename is None:
            return caption_str
        else:
            return figure_str

    def plot_image(self,tkey,ax,image_file,caption):

        img = mpimg.imread(image_file)
        ax.imshow(img)
        ax.axis('off')

        return caption


    def plot_solvent_legend(self,tkey,ax,models,cols=1):
        
        legend_keys = ['TIP5P','TIP3P','GBNeck2','GNN',' ']
        legend_keys = [key for key in legend_keys for m in models if key in m]

        legend_elements = [Line2D([0], [0], color=self._colordict[m], lw=4, label=m) for m in legend_keys]
        ph = [ax.plot([],marker="", ls="",label='Solvent Model:')[0]]
        legend_elements = ph + legend_elements

        ax.legend(handles=legend_elements,loc='upper left',ncol=1 + int(len(legend_keys)/cols),frameon=False,bbox_to_anchor=(-0.05, 0))
        ax.grid(False)
        ax.axis('off')
        ax.axis('off')

        return None

    def plot_custom2d(self,tkey,ax,idx=(),bins_2d = 200,vmax=None,levels=10,limits=[[0,3],[0,2]],xvis=True,yvis=True,mark_state=None):
        if not hasattr(ax, '_2dcustom'):
            id1,id2 = idx
            if type(id1) == int:
                disdata = self._traj_dict[tkey]['TA']._dis_data
                dis1 = disdata[:, id1]
            elif type(id1) == str:
                phi, psi = self.calculate_phi_psi(self.tdict[tkey]['TA'].trajectory,2)
                if id1 == 'psi':
                    dis1 = psi/np.pi*180
                else:
                    dis1 = phi/np.pi*180
            elif type(id1) == tuple:
                dis1 = self.get_distance_by_name(tkey,id1[0],id1[1])
            else:
                print('type not supported')
                return 0
            
            if type(id2) == int:
                disdata = self._traj_dict[tkey]['TA']._dis_data
                dis2 = disdata[:, id2]
            elif type(id2) == str:
                phi, psi = self.calculate_phi_psi(self.tdict[tkey]['TA'].trajectory,2)
                if id2 == 'psi':
                    dis2 = psi/np.pi*180
                else:
                    dis2 = phi/np.pi*180
            elif type(id2) == tuple:
                dis2 = self.get_distance_by_name(tkey,id2[0],id2[1])
            else:
                print('type not supported')
                return 0
            
            z, xedge, yedge = np.histogram2d(dis1, dis2,density=False,bins=bins_2d,range=limits)

            #z, xedge, yedge = np.histogram2d(disdata[:, 0], disdata[:, 1],density=False,bins=200)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            y = 0.5 * (yedge[:-1] + yedge[1:])
            z = z.T

            kjz = self.in_kjmol(z)
            kjz[kjz == np.max(kjz)] = vmax + vmax/levels

            ax.contourf(x,y,kjz,cmap=self._colordict2D[tkey],vmax=vmax,levels=levels)
            ax._2dcustom = True

            O_letter = {'kasae':'gamma','katae':'gamma','kayae':'eta','kapae':'n.a.',
            'kavae':'n.a.','kalae':'n.a.','kaiae':'n.a.','kafae':'n.a.','kaae':'n.a.','kaaae':'n.a.','kaaaae':'n.a.'}


            legend_dict = {0:'%s salt-bridge distance [nm]' % self._pdbid.upper(),
            1:'',
            2:'%s $O_\%s$ - LYS $N_\zeta$ distance [nm]' % (self._vardict[self._pdbid],O_letter[self._pdbid]),
            3:'%s $O_\%s$ - GLU $C_\delta$ distance [nm]'  % (self._vardict[self._pdbid],O_letter[self._pdbid]),
            'phi' : '%s $\phi$' % self._vardict[self._pdbid],
            'psi' : '%s $\psi$' % self._vardict[self._pdbid]}

            if type(id1) == tuple:
                xlab = '%s  - %s distance [nm]' % id1
            else:
                xlab = legend_dict[id1]

            if type(id2) == tuple:
                ylab = '%s  - %s distance [nm]' % id2
            else:
                ylab = legend_dict[id2]
            if type(id2) == str:
                ax.set_yticks(ticks=[-90,0,90],labels=[r'-$\frac{\pi}{2}$','',r'$\frac{\pi}{2}$'])
            if type(id1) == str:
                ax.set_xticks(ticks=[-90,0,90],labels=[r'-$\frac{\pi}{2}$','',r'$\frac{\pi}{2}$'])
            
            if not mark_state is None:
                for ms in mark_state:
                    p = Rectangle(ms[0], ms[1], ms[2], fill=None,linestyle='--',linewidth=2)
                    ax.add_patch(p)
                    if not ms[3] is None:
                        if ms[0][0] + ms[1] + 10 < limits[0][1]:
                            ax.text(ms[0][0] + ms[1] + 10, ms[0][1] + ms[2]/2 ,ms[3],horizontalalignment='left',verticalalignment='center', weight='bold')
                        else:
                            ax.text(ms[0][0] + ms[1]/2, ms[0][1] - 20 ,ms[3],horizontalalignment='center',verticalalignment='center', weight='bold')


            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.get_xaxis().set_visible(xvis)
            ax.get_yaxis().set_visible(yvis)

            caption = (tkey,'2D free energy profile of %s versus %s for %s solvent' % (xlab,ylab,tkey))
            return caption

    def plot_custom1d(self,tkey,ax,idx=0,bins_1d = 200,limits=([0.1,2],[0,15]),selection=[()],xvis=True,yvis=True,uncertainty=True,linestyle='solid'):
        """Create plot of custom 1dx with selections

        Args:
            tkey (_type_): key of trajectory
            ax (_type_): Axis to plot to
            idx (int, optional): index to make histogram for. Defaults to ().
            bins_1d (int, optional): How many bins. Defaults to 200.
            limits (list, optional): _description_. Defaults to [[0,2]].
            selection (list, optional): List of tuples; tuples are defined as (idx,from,to). Defaults to ().
        """
        if not tkey in self._all_z.keys():
            return 0

        tkeys = self.get_range_of_keys(tkey)
        if type(idx) == tuple:
            xhis = 'histspe%s%s_x' % idx
            zhis = 'histspe%s%s_z' % idx
        elif type(idx) == str:
            xhis = 'histspe%s_x' % idx
            zhis = 'histspe%s_z' % idx
        else:
            xhis = 'histspe%i_x' % idx
            zhis = 'histspe%i_z' % idx
            # tkeys.append(tkey)

        for tk in tkeys:
            
            if type(idx) == int:
                disdata = self._traj_dict[tk]['TA']._dis_data
                hisdis = disdata[:, idx]
            elif type(idx) == tuple:
                hisdis = self.get_distance_by_name(tk,idx[0],idx[1])
            elif type(idx) == str:
                phi, psi = self.calculate_phi_psi(self.tdict[tk]['TA'].trajectory,2)
                if idx == 'psi':
                    hisdis = psi/np.pi*180
                else:
                    hisdis = phi/np.pi*180
            else:
                print('type not supported')
                return 0

            truths = np.ones(hisdis.shape,dtype=np.bool8)
            seletext = ''
            O_letter = {'kasae':'gamma','katae':'gamma','kayae':'eta'}

            sele_dict = {0:'%s salt-bridge distance [nm]' % self._pdbid.upper(),
            1:'',
            2:'%s $O_\%s$ - LYS $N_\zeta$ distance [nm]' % (self._vardict[self._pdbid],O_letter[self._pdbid]),
            3:'%s $O_\%s$ - GLU $C_\delta$ distance [nm]'  % (self._vardict[self._pdbid],O_letter[self._pdbid])}

            for sel in selection:
                if seletext == '':
                    seletext += ' for '
                else:
                    seletext += ' and '
                if type(sel[0]) == tuple:
                    dis = self.get_distance_by_name(tk,sel[0][0],sel[0][1])
                    truths = truths & (dis > sel[1]) & (dis < sel[2])
                    seletext += '%s %s distance between \SI{%.2f}{nm} and \SI{%.2f}{nm}' % (sel[0][0],sel[0][1],sel[1],sel[2])
                elif type(sel[0]) == str:
                    phi, psi = self.calculate_phi_psi(self.tdict[tk]['TA'].trajectory,2)
                    if sel[0] == 'psi':
                        dis = psi/np.pi*180
                        seletext += '$\psi$ angles between %.0f\degree and %.0f\degree' % (sel[1],sel[2])
                    else:
                        dis = phi/np.pi*180
                        seletext += '$\phi$ angles between %.0f\degree and %.0f\degree' % (sel[1],sel[2])
                    truths = truths & (dis > sel[1]) & (dis < sel[2])
                    
                else:
                    disdata = self._traj_dict[tk]['TA']._dis_data
                    truths = truths & (disdata[:,sel[0]] > sel[1]) & (disdata[:,sel[0]] < sel[2])
                
                    seletext += '%s between \SI{%.2f}{nm} and \SI{%.2f}{nm}' % (sele_dict[sel[0]],sel[1],sel[2])

            z, xedge = np.histogram(hisdis[truths],bins=bins_1d,range=limits[0])
            x = 0.5 * (xedge[:-1] + xedge[1:])
            z = z.T
            self._all_z[tk]['main'][xhis] = x
            self._all_z[tk]['main'][zhis] = z

            # if idx == 'psi':
            #     print(np.count_nonzero(hisdis),len(hisdis))
            #     print(hisdis)
            #     print(z)
            #     print(self.in_kjmol(z))
        O_letter = {'kasae':'gamma','katae':'gamma','kayae':'eta'}

        legend_dict = {0:'%s salt-bridge distance [nm]' % self._pdbid.upper(),
        1:'',
        2:'%s $O_\%s$ - LYS $N_\zeta$ distance [nm]' % (self._vardict[self._pdbid],O_letter[self._pdbid]),
        3:'%s $O_\%s$ - GLU $C_\delta$ distance [nm]'  % (self._vardict[self._pdbid],O_letter[self._pdbid]),
        'phi' : '%s $\phi$' % self._vardict[self._pdbid],
        'psi' : '%s $\psi$' % self._vardict[self._pdbid]}

        if type(idx) == tuple:
            xlab = '%s  - %s distance [nm]' % idx
        else:
            xlab = legend_dict[idx]
        ylab = r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$'
        
        if type(idx) == str:
            self._plot_generic(tkey,ax,limits,xhis,zhis,xlab,ylab,jakobian_correction=False,uncertainty=uncertainty,linestyle=linestyle)
        else:
            self._plot_generic(tkey,ax,limits,xhis,zhis,xlab,ylab,jakobian_correction=True,uncertainty=uncertainty,linestyle=linestyle)
        
        catption_text = 'free energy profile of %s' % xlab
        catption_text += seletext
        caption = (tkey,catption_text)

        ax.get_xaxis().set_visible(xvis)
        ax.get_yaxis().set_visible(yvis)

        return caption

    def plot_o_lys(self,tkey,ax,limits=([0.1,2],[0,15]),jakobian_correction=True,xvis=True,yvis=True):
        
        O_letter = {'kasae':'gamma','katae':'gamma','kayae':'eta'}
        xhis = 'hist2_x'
        zhis = 'hist2_z'
        xlab = '%s $O_\%s$ - LYS $N_\zeta$ distance [nm]' % (self._vardict[self._pdbid],O_letter[self._pdbid])
        ylab = r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$'
        self._plot_generic(tkey,ax,limits,xhis,zhis,xlab,ylab,jakobian_correction,xvis,yvis)

        caption = (tkey,'free energy profile of %s $O_\gamma$ - LYS $N_\zeta$ distance' % self._vardict[self._pdbid])
        return caption

    def plot_o_glu(self,tkey,ax,limits=([0.1,2],[0,15]),jakobian_correction=True,xvis=True,yvis=True):
        
        O_letter = {'kasae':'gamma','katae':'gamma','kayae':'eta'}
        xhis = 'hist3_x'
        zhis = 'hist3_z'
        xlab = '%s $O_\%s$ - GLU $C_\delta$ distance [nm]'  % (self._vardict[self._pdbid],O_letter[self._pdbid])
        ylab = r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$'
        self._plot_generic(tkey,ax,limits,xhis,zhis,xlab,ylab,jakobian_correction,xvis,yvis)

        caption = (tkey,'free energy profile of %s $O_\gamma$ - GLU $C_\delta$ distance' % self._vardict[self._pdbid])
        return caption

    def plot_saltbridge(self,tkey,ax,limits=([0.1,2],[0,15]),plot_axins=False,jakobian_correction=True,xvis=True,yvis=True):
        
        xhis = 'hist_x'
        zhis = 'hist_z'
        xlab = '%s salt-bridge distance [nm]' % self._pdbid.upper()
        ylab = r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$'
        
        if plot_axins:
            if not hasattr(ax,'_axins'):

                if self._pdbid == 'kapae': 
                    ax._axins = ax.inset_axes([0.45, 0.075, 0.45, 0.15])
                else:
                    ax._axins = ax.inset_axes([0.45, 0.075, 0.45, 0.3])
                
                
                x1, x2, y1, y2 = 0.3,0.43,0,4
                ax._axins.set_xlim(x1, x2)
                ax._axins.set_ylim(y1, y2)
                ax._axins.set_yticks([0,1,2,3,4])
                ax._axins.set_yticklabels([])
                _,lines = ax.indicate_inset_zoom(ax._axins, edgecolor="black")
                lines[1].set(visible=True)
                lines[3].set(visible=False)
            self._plot_generic(tkey,ax._axins,([[0.3,0.43],[0,4]]),xhis,zhis,None,None,jakobian_correction,xvis,yvis)
        
        self._plot_generic(tkey,ax,limits,xhis,zhis,xlab,ylab,jakobian_correction,xvis,yvis)

        caption = (tkey,'free energy profile of %s saltbridge distance' % self._pdbid.upper())
        return caption

    def _plot_generic(self,tkey,ax,limits,xhis,zhis,xlab,ylab,jakobian_correction=True,xvis=True,yvis=True,uncertainty=True,linestyle='solid'):
        tkeys = self.get_range_of_keys(tkey)
        if not tkeys[0] in self._all_z.keys():
            return 0
        valuex = self._all_z[tkeys[0]]['main'][xhis]
        if jakobian_correction:
            valuey = np.array([self.in_kjmol(self._all_z[m]['main'][zhis] / (4*np.pi*self._all_z[m]['main'][xhis]**2) ) for m in tkeys])
        else:
            valuey = np.array([self.in_kjmol(self._all_z[m]['main'][zhis]) for m in tkeys])

        mean_y = np.nanmean(valuey,axis=0) - np.min(np.nanmean(valuey,axis=0)) # static shift
        std_y = np.nanstd(valuey,axis=0)
        ax.plot(valuex,mean_y,linewidth=5.0,color=self._colordict[tkey],label=tkey,linestyle=linestyle,dash_capstyle = 'round')
        if uncertainty:
            ax.fill_between(x=valuex,y1=mean_y - std_y,y2=mean_y + std_y,color=self._colordict[tkey], alpha=0.3)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_ylim(limits[1])
        ax.set_xlim(limits[0])

        ax.get_xaxis().set_visible(xvis)
        ax.get_yaxis().set_visible(yvis)

    def get_range_of_keys(self,tkey):

        keys = []
        #count occurance
        for ak in self.tdict.keys():
            if 'all' in tkey:
                if tkey[:-4] in ak:
                    keys.append(ak)
            else:
                if tkey in ak:
                    keys.append(ak)
        if tkey in keys:
            keys.remove(tkey)

        if len(keys) == 0:
            return [tkey]
        else:
            return keys

    def plot_phi_psi_grid(self,keys=[[]],idx=2,levels=10,vmax=16,mark_state=None,models=None,filename=None,filedir=None):
        """Plot Grid of multiple trajectories for Phi and Psi angles

        Args:
            keys (list, optional): Keys of trajectories in correct shape. Defaults to [[]].
            idx (int, optional): Index for the AAs to calculate angles for. Defaults to 2.
        """

        axd = plt.figure(constrained_layout=True).subplot_mosaic(
            keys,
            empty_sentinel="BLANK",
            gridspec_kw={"height_ratios":[0.1,1,1,0.1]}
        )


        for key in axd:
            if key == 'legend':
                self.plot_solvent_legend(key,axd[key],models)
            elif key == 'cbar':
                self.plot_colorbar_legend(key,axd[key],levels=levels,vmax=vmax,horizontal=True)
            else:
                self.plot_phi_psi(key,axd[key],idx,levels=levels,vmax=vmax,mark_state=mark_state)
        
        if not filename is None:
            plt.savefig('%s/%s' % (filedir,filename))

    def plot_blank(self,tkey,ax):
        ax.axis('off')

    def plot_colorbar_legend(self,tkey,ax,levels,vmax,horizontal=False):
        levels = levels
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')

        
        bounds = [vmax/levels*i for i in range(levels+2)]
        cmap = self.color_to_ListedColormap((0,0,0))
        norm = mtl.colors.Normalize(vmin=0, vmax=vmax)
        if horizontal:
            axins = ax.inset_axes([0.35, 0, 0.3, 0.5])
            cb1 = mtl.colorbar.ColorbarBase(axins, cmap=cmap,norm=norm,boundaries=bounds,ticks=bounds[::2], orientation='horizontal')
        else:
            axins = ax.inset_axes([0, 0.35, 0.5, 0.3])
            cb1 = mtl.colorbar.ColorbarBase(axins, cmap=cmap,norm=norm,boundaries=bounds, orientation='vertical')
        cb1.set_label(r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$')

    def plot_phi_psi(self,tkey,ax,idx=2,levels=10,vmax=16,mark_state=None):
        """Plot single trajectory into provided axis

        Args:
            tkey (str): key for traj
            ax (plt.axis): Axis to plot into
            idx (int, optional): index for AA. Defaults to 2.
            levels (int): number of levels
            vmax (int): maximum zvalue
            mark_state (List(Tuple)): rectangles to show 
        """

        if not hasattr(ax, '_hascbar'):
            if isinstance(idx,list):
                zs = []
                for id in idx:
                    phi, psi = self.calculate_phi_psi(self.tdict[tkey]['TA'].trajectory,id)
                    z, xedge, yedge = np.histogram2d(phi,psi,density=False,bins=np.linspace(-np.pi,np.pi,num=101))
                    zs.append(z)

                z = np.mean(np.array(zs),axis=0)
            else:
                phi, psi = self.calculate_phi_psi(self.tdict[tkey]['TA'].trajectory,idx)
                z, xedge, yedge = np.histogram2d(phi,psi,density=False,bins=np.linspace(-np.pi,np.pi,num=101))
            x = 0.5 * (xedge[:-1] + xedge[1:])
            y = 0.5 * (yedge[:-1] + yedge[1:])
            z = z.T

            #main_2d_hist = ax.contourf(x/np.pi*180,y/np.pi*180,self.in_kjmol(z),cmap=self._colordict2D[tkey],vmax=vmax,levels=levels)
            kjz = self.in_kjmol(z)
            kjz[kjz == np.max(kjz)] = vmax + vmax/levels
            main_2d_hist = ax.contourf(x/np.pi*180,y/np.pi*180,kjz,cmap=self._colordict2D[tkey],vmax=vmax,levels=levels)


            if isinstance(idx,list):
                ax.set_ylabel('$\psi$',labelpad=-20)
                ax.set_xlabel('$\phi$',labelpad=-20)
            else:
                ax.set_ylabel('%s $\psi$' % self._vardict[self._pdbid],labelpad=-20)
                ax.set_xlabel('%s $\phi$' % self._vardict[self._pdbid],labelpad=-20)

            ax.set_yticks(ticks=[-90,0,90],labels=[r'-$\frac{\pi}{2}$','',r'$\frac{\pi}{2}$'])
            ax.set_xticks(ticks=[-90,0,90],labels=[r'-$\frac{\pi}{2}$','',r'$\frac{\pi}{2}$'])
            ax._hascbar = True
            if not mark_state is None:
                for ms in mark_state:
                    p = Rectangle(ms[0], ms[1], ms[2], fill=None,linestyle='--',linewidth=2)
                    ax.add_patch(p)
                    if not ms[3] is None:
                        ax.text(ms[0][0] + ms[1] + 10, ms[0][1] + ms[2]/2 ,ms[3],horizontalalignment='left',verticalalignment='center', weight='bold')

            ax.set_aspect('equal', 'box')

            ax._main2dhist = main_2d_hist
            #plt.colorbar(main_2d_hist,ax=[ax],location='top',label=r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$',pad = -0.98,shrink=0.5,anchor=(0.95,0.5),ticks=[0,vmax])
            #plt.colorbar(main_2d_hist,ax=[ax],label=r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$',shrink=0.5,ticks=[0,vmax],location='right',pad = -0.4)

            caption = (tkey,'Ramachandran plot of %s amino acid simulated with %s solvent' % (self._vardict[self._pdbid],tkey))
            return caption

    def calculate_colormaps(self):
        """Assigns colors to trajectories based on Defaults
        """
        
        self._colordict2D = {}
        colordict2d_base = {'TIP5P':self.color_to_ListedColormap((0,20,140)),'GNN':'Oranges_r','GBNeck2':'Purples_r','TIP3P':self.color_to_ListedColormap((0,100,200))}
        colordict2d_base = {'TIP5P':self.color_to_ListedColormap((0,20,140)),'GNN':self.color_to_ListedColormap((255, 120, 0)),'GBNeck2':self.color_to_ListedColormap((89, 0, 179)),'TIP3P':self.color_to_ListedColormap((0,100,200))}
        for tk in self._traj_dict.keys():
            for ck in colordict2d_base:
                if ck in tk:
                    self._colordict2D[tk] = colordict2d_base[ck]
                self._colordict2D[ck] = colordict2d_base[ck]
        
        self._colordict = {}
        colordict_base = {'TIP5P':'darkblue','GBNeck2':'purple','GNN':'orange','TIP3P':'lightblue'}
        for tk in self._traj_dict.keys():
            for ck in colordict_base:
                if ck in tk:
                    self._colordict[tk] = colordict_base[ck]
                self._colordict[ck] = colordict_base[ck]
        self._colordict[' '] = 'white'

    def plot_2dhist_for_model(self,modelname,color='Blues',titlename=None,dynamic_scale=False,vmax=14,xlim=[0.31,2],ylim=[0.25,1.1]):
        """Plot 2D histogram of one trajectory

        Args:
            modelname (str): key in dictunary
            color (str, optional): Color. Defaults to 'Blues'.
            titlename (str, optional): Titel. Defaults to None.
            dynamic_scale (bool, optional): whether to rescale z axis. Defaults to False.
            vmax (int, optional): Maximum. Defaults to 14.
            xlim (list, optional): xlimits. Defaults to [0.31,2].
            ylim (list, optional): ylimits. Defaults to [0.25,1.1].
        """
        axd = plt.figure(constrained_layout=True).subplot_mosaic(
            [
                ["zoom", "title"],
                ["zoom", "main"],
                ["zoom_hist", "main"],
                ["zoom_hist", "main_hist"],
            ],
            empty_sentinel="BLANK",
            gridspec_kw={"width_ratios": [1, 2],"height_ratios":[1,2,2,1]}
        )
        if titlename is None:
            titlename = modelname

        cp_main_zoom = ConnectionPatch(xyA=(0.43,10), xyB=(0.43,10),coordsA=axd['zoom_hist'].transData, coordsB=axd['main_hist'].transData, linestyle = 'dotted')
        cp_main_zoom.set_in_layout(False)
        axd['main_hist'].add_artist(cp_main_zoom)

        cp_main_zoom = ConnectionPatch(xyA=(0.31,0), xyB=(0.31,0),coordsA=axd['zoom_hist'].transData, coordsB=axd['main_hist'].transData, linestyle = 'dotted')
        cp_main_zoom.set_in_layout(False)
        axd['main_hist'].add_artist(cp_main_zoom)
        axd['main_hist'].set_zorder(-1)

        cp_main_zoom = ConnectionPatch(xyA=(0.43,1.1), xyB=(0.43,1.09),coordsA=axd['main'].transData, coordsB=axd['zoom'].transData, linestyle = 'dotted')
        cp_main_zoom.set_in_layout(False)
        axd['zoom'].add_artist(cp_main_zoom)
        cp_main_zoom = ConnectionPatch(xyA=(0.31,0.25), xyB=(0.311,ylim[0] + 0.001),coordsA=axd['main'].transData, coordsB=axd['zoom'].transData, linestyle = 'dotted')
        cp_main_zoom.set_in_layout(False)
        axd['main'].add_artist(cp_main_zoom)

        main = self._all_z[modelname]['main']
        zoom = self._all_z[modelname]['zoom']

        if dynamic_scale:
            vmax = np.max(self.in_kjmol(main['z']))
        
        main_2d_hist = axd['main'].contourf(main['x'],main['y'],self.in_kjmol(main['z']),cmap=self._colordict2D[modelname],vmax=vmax)
        zoom_2d_hist = axd['zoom'].contourf(zoom['x'],zoom['y'],self.in_kjmol(zoom['z']),cmap=self._colordict2D[modelname],vmax=vmax,levels=20)
        axd['main'].set_xlim(xlim)


        mainrect = Rectangle(xy=(min(zoom['x']),min(zoom['y'])),width=0.115,height=0.848,fc ='none',ec ='black',lw = 2)
        mainrect.set_in_layout(False)
        axd['main'].add_patch(mainrect)
        axd['main'].set_xlim(xlim)
        axd['main'].set_ylim(ylim)
        axd['main'].get_xaxis().set_visible(False)
        axd['main'].get_yaxis().set_visible(True)
        axd['main'].set_ylabel(r'Intramolecular H-Bond distance [nm]')
        axd['main'].yaxis.set_label_position("right")
        axd['main'].yaxis.tick_right()


        axd['zoom'].get_xaxis().set_visible(False)
        axd['zoom'].get_yaxis().set_visible(True)

        axd['zoom'].set_ylabel(r'Intramolecular H-Bond distance [nm]')
        #axd['zoom'].set_ylim(ylim)

        axd['main_hist'].plot(main['hist_x'],self.in_kjmol(main['hist_z']),linewidth=5.0,color=self._colordict[modelname])
        axd['main_hist'].set_xlim(xlim)

        axd['main_hist'].set_xlim(xlim)
        axd['main_hist'].set_ylim([0,20])

        axd['main_hist'].get_yaxis().set_visible(True)
        axd['main_hist'].set_xlabel(r'Salt Bridge Distance [nm]')

        mainhistrect = Rectangle(xy=(0.315,0),width=0.115,height=10,fc ='none',ec ='black',lw = 2)
        mainhistrect.set_in_layout(False)
        axd['main_hist'].add_patch(mainhistrect)

        axd['zoom_hist'].plot(zoom['hist_x'],self.in_kjmol(zoom['hist_z']),linewidth=5.0,color=self._colordict[modelname])
        axd['zoom_hist'].set_xlim([0.31,0.43])
        axd['zoom_hist'].set_ylim([0,10])
        axd['zoom_hist'].set_xlabel(r'Salt Bridge Distance [nm]')
        axd['zoom_hist'].set_ylabel(r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$')

        axd['main_hist'].set_ylabel(r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$')
        axd['main_hist'].yaxis.set_label_position("right")
        axd['main_hist'].yaxis.tick_right()

        axd['title'].text(0.1,0.3,'salt-bridge Analysis: %s' % (titlename),name='Helvetica',size=30,ha='left')
        axd['title'].axis('off')

        plt.colorbar(main_2d_hist,ax=[axd['main']],location='top',label=r'free energy $\left[\frac{\mathrm{kJ}}{\mathrm{mol}}\right]$',pad = -0.98,shrink=0.3,anchor=(0.95,0.5),ticks=[0,vmax])

    def calculate_histogram_data(self):

        for t,traj in enumerate(self._traj_dict.keys()):
            disdata = self._traj_dict[traj]['TA']._dis_data
            z, xedge, yedge = np.histogram2d(disdata[:, 0], disdata[:, 1],density=False,bins=60,range=[[0.31,0.43],[0.25,1.1]])
            #z, xedge, yedge = np.histogram2d(disdata[:, 0], disdata[:, 1],density=False,bins=400)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            y = 0.5 * (yedge[:-1] + yedge[1:])
            z = z.T
            self._all_z[traj]['zoom']['x'] = x
            self._all_z[traj]['zoom']['y'] = y
            self._all_z[traj]['zoom']['z'] = z

            z, xedge = np.histogram(disdata[:, 0],bins=60,range=[0.31,0.43])
            x = 0.5 * (xedge[:-1] + xedge[1:])
            z = z.T
            self._all_z[traj]['zoom']['hist_x'] = x
            self._all_z[traj]['zoom']['hist_z'] = z

            bins_2d = 200

            z, xedge, yedge = np.histogram2d(disdata[:, 0], disdata[:, 1],density=False,bins=bins_2d,range=[[0,3],[0,2]])
            #z, xedge, yedge = np.histogram2d(disdata[:, 0], disdata[:, 1],density=False,bins=200)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            y = 0.5 * (yedge[:-1] + yedge[1:])
            z = z.T
            self._all_z[traj]['main']['x'] = x
            self._all_z[traj]['main']['y'] = y
            self._all_z[traj]['main']['z'] = z

            z, xedge = np.histogram(disdata[:, 0],bins=bins_2d*2,range=[0.1,3])
            #z, xedge = np.histogram(disdata[:, 0],bins=200)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            z = z.T
            self._all_z[traj]['main']['hist_x'] = x
            self._all_z[traj]['main']['hist_z'] = z

            z, xedge = np.histogram(disdata[:, 1][disdata[:,0] < 0.43],bins=bins_2d,range=[0.1,2])
            #z, xedge = np.histogram(disdata[:, 1],bins=200)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            z = z.T
            self._all_z[traj]['main']['hist1_x'] = x
            self._all_z[traj]['main']['hist1_z'] = z


            z, xedge = np.histogram(disdata[:, 2],bins=bins_2d,range=[0.1,2])
            #z, xedge = np.histogram(disdata[:, 1],bins=200)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            z = z.T
            self._all_z[traj]['main']['hist2_x'] = x
            self._all_z[traj]['main']['hist2_z'] = z

            z, xedge = np.histogram(disdata[:, 3],bins=bins_2d,range=[0.1,2])
            #z, xedge = np.histogram(disdata[:, 1],bins=200)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            z = z.T
            self._all_z[traj]['main']['hist3_x'] = x
            self._all_z[traj]['main']['hist3_z'] = z


            z, xedge, yedge = np.histogram2d(disdata[:, 2], disdata[:, 3],density=False,bins=bins_2d)
            #z, xedge, yedge = np.histogram2d(disdata[:, 0], disdata[:, 1],density=False,bins=200)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            y = 0.5 * (yedge[:-1] + yedge[1:])
            z = z.T
            self._all_z[traj]['main']['specialx'] = x
            self._all_z[traj]['main']['specialy'] = y
            self._all_z[traj]['main']['specialz'] = z
            
            
            z, xedge, yedge = np.histogram2d(disdata[:, 0], disdata[:, 2],density=False,bins=bins_2d)
            #z, xedge, yedge = np.histogram2folder + '/' + pdbid + '_barostat' + d(disdata[:, 0], disdata[:, 1],density=False,bins=200)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            y = 0.5 * (yedge[:-1] + yedge[1:])
            z = z.T
            self._all_z[traj]['main']['hist2d_02x'] = x
            self._all_z[traj]['main']['hist2d_02y'] = y
            self._all_z[traj]['main']['hist2d_02z'] = z
            
            z, xedge, yedge = np.histogram2d(disdata[:, 0], disdata[:, 3],density=False,bins=bins_2d)
            #z, xedge, yedge = np.histogram2d(disdata[:, 0], disdata[:, 1],density=False,bins=200)
            x = 0.5 * (xedge[:-1] + xedge[1:])
            y = 0.5 * (yedge[:-1] + yedge[1:])
            z = z.T
            self._all_z[traj]['main']['hist2d_03x'] = x
            self._all_z[traj]['main']['hist2d_03y'] = y
            self._all_z[traj]['main']['hist2d_03z'] = z

    def calculate_distances_for_trajs(self):
        """Calculate distances for all trajectories
        """
        for idx,trajkey in tqdm(enumerate(self._traj_dict.keys())):
            self.calculate_distances(self._traj_dict[trajkey]['TA'])

    def calculate_distances(self,ta_a):
        f = self.calculate_distance(ta_a.trajectory.xyz,[self._nz_id , self._cd_id])
        s = self.calculate_distance(ta_a.trajectory.xyz,[self._cd_id, self._ah_id])
        d = self.calculate_distance(ta_a.trajectory.xyz,[self._spe, self._nz_id ])
        z = self.calculate_distance(ta_a.trajectory.xyz,[self._spe, self._cd_id])
        distances = np.concatenate((f,s,d,z),axis=1)
        ta_a._dis_data = distances

    def detect_key_atoms(self):
        """Find key atoms for analysis
        """
        self._spe = 0
        for a,atom in enumerate(self._traj_dict['TIP5P']['TA'].trajectory.top.atoms):
            if 'LYS2-NZ' == str(atom):
                self._nz_id = a
            if 'ALA3-N' == str(atom):
                self._ah_id = a
            if 'GLU6-CD' == str(atom):
                self._cd_id = a
            if 'GLU7-CD' == str(atom):
                self._cd_id = a
            if 'GLU5-CD' == str(atom): #for kaae
                self._cd_id = a
            if 'OG' in str(atom):
                self._spe = a
            if 'OH' in str(atom):
                self._spe = a
        
    def process_trajectories(self):
        """Read in trajectories for all simulations
        """
        
        for idx,trajkey in tqdm(enumerate(self._traj_dict.keys())):
            self.process_single(trajkey)

    def process_single(self,traj):
        """Read in single trajectory

        Args:
            traj (str): key of trajectory to read in 
        """
        self._traj_dict[traj]['TA'] = Trajectory_Analysis()
        self._traj_dict[traj]['TA'].read_trajectory(self._traj_dict[traj]['file'])

    def test_trajectories_for_completness(self):
        one_incomplete = False
        print(self._pdbid)
        for idx,trajkey in enumerate(self._traj_dict.keys()):
            if self._traj_dict[trajkey]['TA']._trajectory.n_frames % 10000:
                print('Potentially incomplete: ',trajkey,self._traj_dict[trajkey]['TA']._trajectory.n_frames)
                one_incomplete = True
        if not one_incomplete:
            print('all trajectories are complete')

    def register_explicit_traj(self,explicit_model='',folder=None,chunks=20,random_slices=5):
        """Add explicit simulation to dictonary

        Args:
            explicit_model (str, optional): model type. Defaults to ''.
            folder (str, optional): folder where trajectory is located. Defaults to None.
            chunks (int, optional): How many chunks should be read in. Defaults to 20.
            random_slices (int, optional): how many slices should be made. Defaults to 5.
        """

        if folder is None:
            folder = self._folder

        processed_traj_file = folder + self._pdbid + '/' + '%s_%s_output_protein.h5' % (self._pdbid,explicit_model)

        # check if all chunks are present and not too much!
        pres = True
        for c in range(random_slices):
            processed_traj_file = folder + self._pdbid + '/' + '%s_%s_output_protein_c%i.h5' % (self._pdbid,explicit_model,c)
            pres = pres and os.path.isfile(processed_traj_file)
        
        add_file = folder + self._pdbid + '/' + '%s_%s_output_protein_c%i.h5' % (self._pdbid,explicit_model,random_slices)
        if os.path.isfile(add_file):
            pres = False
            exit('Additional file is present, remove them!')

        if not self.reprocess and os.path.isfile(processed_traj_file) and pres:
            pass
        else:
            for i in range(chunks):
                file = folder + self._pdbid + '/' + '%s_%s_%i_output_protein.h5' % (self._pdbid,explicit_model,i)
                traj = mdtraj.load(file)
                try:
                    traj = mdtraj.load(file)
                except:
                    print('failed, continue with shorter sim: ',i*1000/20)
                if i == 0:
                    Traj = traj
                else:
                    Traj += traj

            processed_traj_file = folder + self._pdbid + '/' + '%s_%s_output_protein.h5' % (self._pdbid,explicit_model)
            Traj.save_hdf5(processed_traj_file)

            # slice
            len_of_slice = int(len(Traj)/random_slices)
            for c in range(random_slices):
                processed_traj_file = folder + self._pdbid + '/' + '%s_%s_output_protein_c%i.h5' % (self._pdbid,explicit_model,c)
                Traj[c*len_of_slice:(c+1)*len_of_slice].save_hdf5(processed_traj_file)
        
        processed_traj_file = folder + self._pdbid + '/' + '%s_%s_output_protein.h5' % (self._pdbid,explicit_model)
        self._traj_dict['%s' % explicit_model]['file'] = processed_traj_file
        for c in range(random_slices):
            processed_traj_file = folder + self._pdbid + '/' + '%s_%s_output_protein_c%i.h5' % (self._pdbid,explicit_model,c)
            self._traj_dict['%s_c%i' % (explicit_model,c)]['file'] = processed_traj_file

    def register_gnn_traj(self,folder,trainons,number_of_chunks=10,random_seeds = ['1','2','161311'],postfix=''):
        """Add new gnn trajectories to dict, automatically adds seeds seperately and as one

        Args:
            folder (str): folder were trajectories are stored
            trainon (list): list of keyskeys for which peptides the model was trained on
            number_of_chunks (int, optional): Number of chunks to concatenate. Defaults to 10.
            random_seeds (list, optional): Seeds which should be analysed. Defaults to ['1','2','161311'].
            postfix (str, optional): Postfix for which simulations to load. Defaults to ''.
        """

        for trainon in trainons:
            for ra in random_seeds:
                self._traj_dict['GNN_tr_%sra_%s' % (trainon,ra)]['file'] = self.concatenate_gnn(folder,trainon,self._pdbid,number_of_chunks,[ra],postfix)
            
            self._traj_dict['GNN_tr_%sra_all' % (trainon)]['file'] = self.concatenate_gnn(folder,trainon,self._pdbid,number_of_chunks,random_seeds,postfix)

    def concatenate_gnn(self,folder,trained_on,run_id,number=10,random_sl=[''],model_postfix=''):
        """Concatenate GNN simulations

        Args:
            folder (str): folder were trajectories are stored
            trained_on (str): traind on keys
            run_id (str): pdbid which was simulated
            number (int, optional): number of chunks to concatenate. Defaults to 10.
            random_sl (list, optional): random seeds which should be used for analysis, if multiple all are concatenated together. Defaults to [''].
            model_postfix (str, optional): Model Postfix with aditional information which model should be used. Defaults to ''.

        Returns:
            str: path to new concatenated hdf5 file
        """

        # Check if file was preprocessed before
        name = ''
        for random_s in random_sl:
            name += random_s

        processed_traj_file = folder + trained_on + run_id + random_s + model_postfix + '/' + '%s_vacuum_output.h5' % name
        if not self.reprocess:
            if os.path.isfile(processed_traj_file):
                return processed_traj_file

        # add up trajectory information
        Traj = None
        name = ''
        for random_s in random_sl:
            for n in range(number):
                try:
                    file = folder + trained_on + run_id + random_s + model_postfix +'/' + '%s_vacuum_%i_output.h5' % (run_id,n)
                    traj = mdtraj.load(file)
                    if traj.n_frames % 1000:
                        print('Trajectory might be broken: ',file)
                    if Traj is None:
                        Traj = traj
                    else:
                        Traj += traj
                except Exception as e: print(e)
            name += random_s
        Traj.save_hdf5(processed_traj_file)
        return processed_traj_file

    def register_traj(self,name,core,folder):
        """Register Trajectory manually

        Args:
            name (str): name of solventsystem
            core (str): corename
            folder (str): folder in which the trajectory is stored
        """
        if os.path.isfile(folder + '/' + self._pdbid + '/' + '%s_%s.h5' % (self._pdbid,core)):
            self._traj_dict[name]['file'] = folder + '/' + self._pdbid + '/' + '%s_%s.h5' % (self._pdbid,core)
        else:
            warnings.warn('No such File: ' + folder + '/' + self._pdbid + '/' + '%s_%s.h5' % (self._pdbid,core))

    @property
    def reprocess(self):
        return self._reprocess

    @reprocess.setter
    def reprocess(self,reprocess):
        self._reprocess = reprocess

    @property
    def tdict(self):
        return self._traj_dict

    @property
    def multiprocessing(self):
        return self._multiprocessing
    
    @property
    def gnn_trainons(self):
        return [t for t in self.tdict.keys() if 'GNN' in t]

    @multiprocessing.setter
    def multiprocessing(self,mp):
        self._multiprocessing = mp

    @staticmethod
    def normalize(arr):
        arr[arr==0] = np.min(arr[arr>0])
        return arr / np.sum(arr)

    @staticmethod
    def in_kjmol(z):
        kT = 2.479
        z[z==0] = np.min(z[z>0])
        return (-np.log(PeptideAnalyzer.normalize(z)) - np.min(-np.log(PeptideAnalyzer.normalize(z)))) * kT

    @staticmethod
    def in_kjmol_min_dif(z,z2):
        kT = 2.479
        print(np.min(z[z>0]),np.min(z2[z2>0]))
        z[z==0] = np.min(z2[z2>0])
        return (-np.log(self.normalize(z)) - np.min(-np.log(self.normalize(z)))) * kT

    @staticmethod
    def calculate_distance(xyz,fromto):
        return np.reshape(np.sqrt(np.sum((xyz[:,fromto[0]]-xyz[:,fromto[1]])**2,axis=1)),(xyz.shape[0],1))

    @staticmethod
    def calculate_phi_psi(traj,idx=2):
        _, tor = mdtraj.compute_phi(traj)
        pro_tor_phi = tor[:,idx]
        _, tor = mdtraj.compute_psi(traj)
        pro_tor_psi = tor[:,idx]
        return pro_tor_phi, pro_tor_psi

    @staticmethod
    def color_to_ListedColormap(rgb):
        N = 256
        vals = np.ones((N,3))
        vals[:, 0] = np.linspace(rgb[0]/256, 1, N)
        vals[:, 1] = np.linspace(rgb[1]/256, 1, N)
        vals[:, 2] = np.linspace(rgb[2]/256, 1, N)
        newcmp = ListedColormap(vals)
        return newcmp



################################

# Utility functions

################################


def _get_eccentricity(xyz, indices, plot_ell=False, get_area=False):
    xyz = xyz[indices, :]

    pca = PCA(n_components=3)
    xy = pca.fit_transform(xyz)
    ell = EllipseModel()
    ell.estimate(xy[:, [0, 1]])
    xc, yc, a, b, theta = ell.params

    if plot_ell:
        g = sns.relplot(x=xy[:, 0], y=xy[:, 1])
        xy_pre = ell.predict_xy(np.linspace(0, np.pi * 2))
        plt.plot(xy_pre[:, 0], xy_pre[:, 1], color='black')

        if a < b:
            text = 'eccentricity: %.3f' % np.sqrt(1 - a ** 2 / b ** 2)
        else:
            text = 'eccentricity: %.3f' % np.sqrt(1 - b ** 2 / a ** 2)
        plt.text(x=-0.4, y=0.4, s=text)

    if a < b:
        e = np.sqrt(1 - a ** 2 / b ** 2)
    else:
        e = np.sqrt(1 - b ** 2 / a ** 2)
    if get_area:
        return e, a * b * np.pi
    else:
        return e


@njit()
def reverse_boltzmann(P, T=300, kcal=True, max_P=1):
    Na = 6.02214076e23
    kb = 1.380649e-23
    R = 0.593 / 298
    if kcal:
        return -1 * np.log(P) * R * T  # kcal/mol
    else:
        return -1 * np.log(P) * kb * T * Na / 1000  # kj/mol


from scipy.stats import gaussian_kde


def get_energies_from_distances(data, plot=False, start=None, end=None):
    kde = gaussian_kde(data)

    if start == None:
        start = np.min(data)
    if end == None:
        end = np.max(data)

    steps = 100
    positions = np.arange(start, end, (end - start) / int((end - start) / 0.02))

    probabilities = kde(positions)
    # probabilities = np.histogram(data,bins=int((end-start)/0.02),density=True,range=(start,end))[0]

    # print(probabilities)
    # print(reverse_boltzmann(probabilities[50]),reverse_boltzmann(probabilities[50],kcal=False))
    probabilities = probabilities / np.max(probabilities)  # normalize
    # print(reverse_boltzmann(probabilities[50]))
    # max_p = max(probabilities)
    energies = np.array([reverse_boltzmann(p) for p in probabilities])
    # print(energies[50])
    # Normalize energies
    # energies = energies - min(energies)
    # print(energies[50])
    if plot:
        sns.lineplot(x=positions, y=energies)
    return energies, positions


@njit()
def rmse(ar1, ar2):  # fastes
    errors = 0
    for i in range(len(ar1)):
        error = (ar1[i] - ar2[i]) ** 2
        errors += error
    return np.sqrt(errors / len(ar1))


def rmse_np(ar1, ar2):
    rmse = np.sqrt(np.mean((ar1 - ar2) ** 2))
    return rmse


def rmse_sp(ar1, ar2):
    return np.sqrt(mean_squared_error(ar1, ar2))


def random():
    pass


@njit()
def smooth(y, box_pts):
    y_smooth = np.empty(y.size, dtype=np.float32)
    assert not box_pts % 2
    half_box = box_pts // 2
    y_smooth[:half_box] = y[:half_box]
    y_smooth[(y.size - half_box):] = y[(y.size - half_box):]
    for i in prange(y.size - box_pts):
        m = i + half_box
        y_smooth[m] = np.mean(y[i:(i + box_pts)])
    return y_smooth


@njit(parallel=True)
def get_radii_data(data_arr, n_range: int = 0, n_atoms: int = 0, ns_per_step: float = 0.1):
    '''
    Function to extract The Radii in plotable format
    :param data: dataframe containing the data
    :return: dataframe with plotable data
    '''

    np_arr = np.zeros((n_atoms, 6, n_range), dtype=np.float32)

    for j in prange(n_atoms):
        arr = data_arr[j]
        for i in prange(n_range):
            np_arr[j, 0, i] = i * ns_per_step
            np_arr[j, 1, i] = arr[i]
            np_arr[j, 3, i] = 1 / arr[i]

        np_arr[j, 2, :] = smooth(arr, 10)
        np_arr[j, 4, :] = smooth(np_arr[j, 3, :], 10)
        np_arr[j, 5, :] = smooth(np_arr[j, 3, :], 20)

    return np_arr


def process_lines(charge_dict, radius_dict, matching, infile, num_atoms, last_line=0, prev_res_id=-1, sub=0, start=0,
                  calculate_scale=False, scale_dict=None):
    prev_res_id = -1
    start = 0
    sub = 0
    # Get Data
    with open(infile, 'r') as lines:
        # print(last_line)
        lines.seek(0)
        if calculate_scale:
            Data = np.empty((num_atoms, 13), dtype="object")
        else:
            Data = np.empty((num_atoms, 12), dtype="object")
        for i, line in enumerate(lines):
            if i < last_line + 1:
                continue

            context = line.split()
            if context[0] == "ENDMDL":
                # print(line, i)
                last_line = i
                break

            s = ""
            d = ""
            for char in context[0]:
                if char.isdigit():
                    d += char
                else:
                    s += char
            if d != "":
                acontext = np.empty((len(context) + 1), dtype=object)
                acontext[0] = s
                acontext[1] = d
                acontext[2:] = context[1:]
                context = acontext

            if start == 0 and context[0] == "ATOM":
                start = i
            if context[0] != "ATOM":
                continue

            if len(context) < 9:
                continue
            s = ""
            d = ""
            for char in context[4]:
                if char.isdigit():
                    d += char
                else:
                    s += char
            if d != "":
                acontext = np.empty((len(context) + 1), dtype=object)
                acontext[4] = s
                acontext[5] = d
                acontext[:4] = context[:4]
                acontext[6:] = context[5:]
                context = acontext

            res = context[3]
            atom = context[2]
            resid = int(context[5]) - 1

            if prev_res_id != resid:
                prev_res_id = resid
                sub = i

            x = context[6]
            y = context[7]
            z = context[8]

            atom_in_res = i - sub
            atom_type = matching[resid].atoms[atom_in_res].type
            atom_name = matching[resid].atoms[atom_in_res].name
            charge = charge_dict[atom_type]['charge']
            try:
                radius = radius_dict[atom_type]['radius']  # + 0.009 # add constant dielectric if necessary
            except:
                radius = 0

            if calculate_scale:
                scale = scale_dict[atom_type]['scale']
                data = [str(context[0]), int(i - start) + 1, str(atom_name), str(res), "A", int(resid), "   ", float(x),
                        float(y), float(z), float(charge), float(radius * 10), float(scale)]

                for j, d in enumerate(data):
                    Data[int(i - start), j] = d
            else:
                data = [str(context[0]), int(i - start) + 1, str(atom_name), str(res), "A", int(resid), "   ", float(x),
                        float(y), float(z), float(charge), float(radius * 10)]

                for j, d in enumerate(data):
                    Data[int(i - start), j] = d

    return Data, last_line, prev_res_id, sub, start


@njit(parallel=True)
def reformat(values, leng=6):
    # len inclusive of sign
    out = np.empty(len(values))
    for j in prange(out.shape[0]):
        i = 0
        value = np.abs(values[j])
        while (value != value % 10 ** i):
            i += 1
        out[j] = float(np.round(values[j], leng - i))
    return out


def calc_sasa_radius(traj):
    return np.mean(shrake_rupley(traj))


# utility analysis functions

def normalize(arr):
    arr[arr==0] = np.min(arr[arr>0])
    return arr / np.sum(arr)

def in_kjmol(z):
    kT = 2.479
    z[z==0] = np.min(z[z>0])
    return (-np.log(normalize(z)) - np.min(-np.log(normalize(z)))) * kT

def get_dihedrals_by_name(traj,i,j,k,l,ins=2):


    if isinstance(traj,list):
        dis = []
        for t in traj:
            dis.append(get_dihedrals_by_name(t,i,j,k,l,ins))
        return dis

    oc = 0
    for a,atom in enumerate(traj.top.atoms):
        if str(atom)[-len(i):] == i:
            i_id = a
            oc += 1
        if str(atom)[-len(j):] == j:
            j_id = a
            oc += 1
        if str(atom)[-len(k):] == k:
            k_id = a
            oc += 1
        if str(atom)[-len(l):] == l:
            l_id = a
            oc += 1
    assert oc == 4
    
    return mdtraj.compute_dihedrals(traj,[[i_id,j_id,k_id,l_id]])


def get_prob(d1,d2,num=50):
    z, xedge, yedge = np.histogram2d(d1,d2,density=True,bins=np.linspace(-np.pi,np.pi,num=num))
    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])
    z = z.T

    return x,y,z

def get_xyz(d1,d2,num_bins=50,bins=None):
    if bins is None:
        bins = np.linspace(-np.pi,np.pi,num=num_bins)
    z, xedge, yedge = np.histogram2d(d1,d2,density=False,bins=bins)
    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])
    z = z.T

    kjz = in_kjmol(z)

    return x,y,kjz

def get_xz(d1):
    z, xedge = np.histogram(d1,bins=100)
    x = 0.5 * (xedge[:-1] + xedge[1:])
    z = z.T

    return x,in_kjmol(z)

def initialize_plt(figsize = (15,10),fontsize = 22):
    # initialice Matplotlib
    _=plt.figure()
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family':'Sans'})


def plot_free_energy(ax,dis,nbins,range=[0.1,0.7],color='blue',label='',ret_xz=False,linewidth=5,linestyle='solid',use_jakobian_correction=True):

    if not isinstance(dis,list):
        dis = [dis]

    zs = []
    for d in dis:
        z, xedge = np.histogram(d,bins=nbins,range=range)
        x = 0.5 * (xedge[:-1] + xedge[1:])
        z = z.T
        zs.append(z)
    if use_jakobian_correction:
        valuey = np.array([in_kjmol(z/(4*np.pi*(x**2))) for z in zs])
    else:
        valuey = np.array([in_kjmol(z) for z in zs])
    mean_y = np.nanmean(valuey,axis=0) - np.min(np.nanmean(valuey,axis=0)) # static shift
    std_y = np.nanstd(valuey,axis=0)
    if ret_xz:
        return x, mean_y, std_y
    ax.plot(x,mean_y,linewidth=linewidth,color=color,label=label,linestyle=linestyle)
    ax.fill_between(x=x,y1=mean_y - std_y,y2=mean_y + std_y,color=color, alpha=0.3)

def load_parallel_traj(file,nrep):
    traj = mdtraj.load(file)
    individual_trajs = []
    individual_trajs += [traj.atom_slice(traj.top.select('chainid %i' % i)) for i in range(nrep)]
    return individual_trajs

def plot_contour(ax, x, y, z, levels, cmap, vmin, vmax,location=None,label=None):
    cf = ax.contourf(x/np.pi*180, y/np.pi*180, z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    if location != 'share':
        cbar = plt.colorbar(cf, ax=ax, shrink=0.5,location=location,label=label)
        cbar.ax.set_aspect(20)
        cbar.ax.tick_params(labelsize=10)
        return cf, cbar
    else:
        return cf

def plot_wass_dist(ax, gnz, z, label):
    import ot
    M = ot.dist(gnz, z)
    wass_dist = ot.emd2([], [], M)
    if not label is None:
        ax.text(0.5, 0.925, f'{label}: {wass_dist:.4f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    return wass_dist

def get_distance_by_name(traj,n1,n2):
    if isinstance(traj,list):
        dis = []
        for t in traj:
            dis.append(get_distance_by_name(t,n1,n2))
        return dis

    oc = 0
    for a,atom in enumerate(traj.top.atoms):
        if str(atom)[-len(n1):] == n1:
            n1_id = a
            oc += 1
        if str(atom)[-len(n2):] == n2:
            n2_id = a
            oc += 1
    assert oc == 2
    
    return calculate_distance(traj.xyz,(n1_id, n2_id))

def calculate_distance(xyz,fromto):
    return np.reshape(np.sqrt(np.sum((xyz[:,fromto[0]]-xyz[:,fromto[1]])**2,axis=1)),(xyz.shape[0],1))


def get_wasserstein_distance(d1,d2,nbins=50,drange=(0.1,0.5),use_jakobian_correction=False):
    if use_jakobian_correction:
        hist1, bin_edges1 = np.histogram(d1, bins=nbins, range=drange,density=True)
        hist2, bin_edges2 = np.histogram(d2, bins=nbins, range=drange,density=True)

        x = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])
        hist1 = hist1/(4*np.pi*(x**2))
        hist1 = normalize(hist1)

        x = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])
        hist2 = hist2/(4*np.pi*(x**2))
        hist2 = normalize(hist2)

        return wasserstein_distance(hist1,hist2)
    else:
        hist1, bin_edges1 = np.histogram(d1, bins=nbins, range=drange, density=True)
        hist2, bin_edges2 = np.histogram(d2, bins=nbins, range=drange, density=True)
        return wasserstein_distance(hist1, hist2)


@njit(fastmath=True)
def get_cluster_assignment(coords,new_coord,ref_labels):

    diff = np.sum((coords - new_coord)**2,axis=1)
    return ref_labels[np.argmin(diff)]

@njit(fastmath=True)
def get_cluster_assignment_multiple(coords,new_coords,ref_labels):
    return np.array([get_cluster_assignment(coords,new_coord,ref_labels) for new_coord in new_coords])

@njit(fastmath=True,parallel=True)
def get_cluster_assignment_multiple_parallel(coords,new_coords,ref_labels):

    labels = np.empty(len(new_coords))
    for i in prange(len(new_coords)):
        labels[i] = get_cluster_assignment(coords,new_coords[i],ref_labels)
    
    return labels

@njit(fastmath=True)
def get_cluster_assignment_multiple_vec(coords,new_coords,ref_labels):
    positions = np.argmin(np.sum((coords - new_coords.reshape((len(new_coords),1,2)))**2,axis=2),axis=1)
    return ref_labels[positions]

COLORDICT =  {'TIP5P':'darkblue','GBNeck2':'purple','GNN':'orange','TIP3P':'lightblue'}

def plot_solvent_legend(ax,models,cols=1,legend_ops={'loc':'center'}):
    
    legend_keys = ['TIP5P','TIP3P','GBNeck2','GNN',' ']
    legend_keys = [key for key in legend_keys for m in models if key in m]

    legend_elements = [Line2D([0], [0], color=COLORDICT[m], lw=4, label=m) for m in legend_keys]
    ph = [ax.plot([],marker="", ls="",label='Solvent Model:')[0]]
    legend_elements = ph + legend_elements

    ax.legend(handles=legend_elements,ncol=1 + int(len(legend_keys)/cols),frameon=False,**legend_ops)
    ax.grid(False)
    ax.axis('off')
    ax.axis('off')

    return None