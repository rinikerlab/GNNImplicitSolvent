'''
File to define Neural Networks
'''

import torch_cluster
from torch_geometric.transforms import RadiusGraph
from torch_geometric.nn import radius_graph
from torch.nn import PairwiseDistance
import torch
from torch import nn
from torch_scatter import scatter
from torch_sparse import SparseTensor

#from torch_geometric.nn.models.dimenet import swish, BesselBasisLayer, SphericalBasisLayer, EmbeddingBlock, OutputBlock, InteractionBlock


#from torch_geometric.nn import DimeNet

from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from torch.cuda.amp import autocast


T = TypeVar('T', bound='Module')

from MachineLearning.GNN_Layers import *

torch.backends.cudnn.benchmark = True


class GNN_GBNeck(torch.nn.Module):
    def __init__(self, radius=0.4, max_num_neighbors=32, parameters=None, device=None, jittable=False):
        '''
        GNN to reproduce the GBNeck Model
        '''
        super().__init__()

        # In order to be differentiable all tensors *need* to be created on the same device
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device

        self._nobatch = False

        # Initiate Graph Builder
        self._gbparameters = torch.tensor(parameters,dtype=torch.float,device=self._device)
        self._radius = radius
        self._max_num_neighbors = max_num_neighbors
        self._grapher = RadiusGraph(r=self._radius, loop=False, max_num_neighbors=self._max_num_neighbors)

        # Init Distance Calculation
        self._distancer = PairwiseDistance()
        self._jittable = jittable
        if jittable:
            self.aggregate_information = GBNeck_interaction(parameters,self._device).jittable()
            self.calculate_energies = GBNeck_energies(parameters,self._device).jittable()
        else:
            self.aggregate_information = GBNeck_interaction(parameters,self._device)
            self.calculate_energies = GBNeck_energies(parameters,self._device)

        self.lin = nn.Linear(1,1)

    def get_edge_features(self, distances, alpha=2, max_range=0.4, min_range=0.1, num_kernels=32):
        m = alpha * (max_range - min_range) / (num_kernels + 1)
        lower_bound = min_range + m
        upper_bound = max_range - m
        centers = torch.linspace(lower_bound, upper_bound, num_kernels, device=self._device)
        k = distances - centers
        return torch.maximum(torch.tensor(0, device=self._device), torch.pow(1 - (k / m) ** 2, 3))

    def build_graph(self, data):

        # Get Radius Graph
        graph = self._grapher(data)

        # Extract edge index
        edge_index = graph.edge_index

        # Extract node features
        node_features = graph.atomic_features

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes

    def forward(self, data):

        # Enable tracking of gradients
        # Get input as Tensor create on device
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)

        x = self._gbparameters.repeat(torch.max(data.batch)+1,1)

        # Do message passing
        Bc = self.aggregate_information(x=x, edge_index=edge_index, edge_attributes=edge_attributes) # B and charges
        energies = self.calculate_energies(x=Bc, edge_index=edge_index, edge_attributes=edge_attributes)

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(energies.sum(), inputs=data.pos, create_graph=True)[0]
        forces = -1 * gradients

        if self._nobatch:
            energy = energies.sum()
            energy = energy.unsqueeze(0)
            energy = energy.unsqueeze(0)
        else:
            energy = torch.empty((torch.max(data.batch) + 1,1), device=self._device)
            for batch in data.batch.unique():
                energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy, forces

class GNN_GBNeck_2(GNN_GBNeck):

    def forward(self, data):

        # Enable tracking of gradients
        # Get input as Tensor create on device
        data.pos = data.pos.clone().detach().requires_grad_(True)

        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)

        x = data.atom_features

        # Do message passing
        Bc = self.aggregate_information(x=x, edge_index=edge_index, edge_attributes=edge_attributes) # B and charges
        energies = self.calculate_energies(x=Bc, edge_index=edge_index, edge_attributes=edge_attributes)

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(energies.sum(), inputs=data.pos, create_graph=True)[0]
        forces = -1 * gradients

        if self._nobatch:
            energy = energies.sum()
            energy = energy.unsqueeze(0)
            energy = energy.unsqueeze(0)
        else:
            energy = torch.empty((torch.max(data.batch) + 1,1), device=self._device)
            for batch in data.batch.unique():
                energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy, forces

    def build_graph(self, data):

        # Get Radius Graph
        graph = self._grapher(data)

        # Extract edge index
        edge_index = graph.edge_index

        # Extract node features
        node_features = graph.atom_features

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes

class GNN_Grapher:

    def __init__(self,radius,max_num_neighbors) -> None:
        self._gnn_grapher = RadiusGraph(r=radius, loop=False, max_num_neighbors=max_num_neighbors)

    def build_gnn_graph(self, data):

        # Get Radius Graph
        graph = self._gnn_grapher(data)

        # Extract edge index
        edge_index = graph.edge_index

        # Extract node features
        node_features = graph.atomic_features

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes

class GNN_Grapher_2(GNN_Grapher):

    def build_gnn_graph(self, data):

        # Get Radius Graph
        graph = self._gnn_grapher(data)

        # Extract edge index
        edge_index = graph.edge_index

        # Extract node features
        node_features = graph.atom_features

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes


class GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr(GNN_GBNeck_2,GNN_Grapher_2):

    def __init__(self,fraction=0.5,radius=0.4, max_num_neighbors=32, parameters=None, device=None, jittable=False):

        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck_2.__init__(self,radius=gbneck_radius, max_num_neighbors=max_num_neighbors, parameters=parameters, device=device, jittable=jittable)
        GNN_Grapher_2.__init__(self,radius=radius, max_num_neighbors=max_num_neighbors)

        self._fraction = fraction
        if self._jittable:
            self.interaction1 = IN_layer_all_swish_2pass(3 + 3, 128).jittable()
            self.interaction2 = IN_layer_all_swish_2pass(128 + 128, 128).jittable()
            self.interaction3 = IN_layer_all_swish_2pass(128 + 128, 1).jittable()
        else:
            self.interaction1 = IN_layer_all_swish_2pass(3 + 3, 128)
            self.interaction2 = IN_layer_all_swish_2pass(128 + 128, 128)
            self.interaction3 = IN_layer_all_swish_2pass(128 + 128, 1)

        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)
        _, gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)

        x = data.atom_features
        # Do message passing
        Bc = self.aggregate_information(x=x, edge_index=edge_index, edge_attributes=edge_attributes)  # B and charges

        # ADD small correction
        Bcn = torch.concat((Bc,x[:,1].unsqueeze(1)),dim=1)
        Bcn = self.interaction1(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:,0].unsqueeze(1) * (self._fraction + self.sigmoid(Bcn)*(1-self._fraction)*2)

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn,Bc[:,1].unsqueeze(1)),dim=1)

        energies = self.calculate_energies(x=Bc, edge_index=edge_index, edge_attributes=edge_attributes)

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(energies.sum(), inputs=data.pos, create_graph=True)[0]
        forces = -1 * gradients
        if self._nobatch:
            energy = energies.sum()
            energy = energy.unsqueeze(0)
            energy = energy.unsqueeze(0)
        else:
            energy = torch.empty((torch.max(data.batch) + 1, 1), device=self._device)
            for batch in data.batch.unique():
                energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy, forces

class GNN3_true_delta_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs(GNN_GBNeck_2,GNN_Grapher_2):

    def __init__(self,fraction=0.5,radius=0.4, max_num_neighbors=32, parameters=None, device=None, jittable=False):

        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck_2.__init__(self,radius=gbneck_radius, max_num_neighbors=max_num_neighbors, parameters=parameters, device=device, jittable=jittable)
        GNN_Grapher_2.__init__(self,radius=radius, max_num_neighbors=max_num_neighbors)

        self.lin1 = nn.Linear(2 + 7,1)
        self._fraction = fraction
        if self._jittable:
            self.interaction1 = IN_layer_all_swish_2pass(2 + 2, 128).jittable()
            self.interaction2 = IN_layer_all_swish_2pass(128 + 128, 128).jittable()
            self.interaction3 = IN_layer_all_swish_2pass(128 + 128, 1).jittable()
        else:
            self.interaction1 = IN_layer_all_swish_2pass(2 + 2, 128)
            self.interaction2 = IN_layer_all_swish_2pass(128 + 128, 128)
            self.interaction3 = IN_layer_all_swish_2pass(128 + 128, 1)

        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)
        _, gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)

        x = data.atom_features

        # Do message passing
        Bc = self.aggregate_information(x=x, edge_index=edge_index, edge_attributes=edge_attributes)  # B and charges
        gbn_energies = self.calculate_energies(x=Bc, edge_index=edge_index, edge_attributes=edge_attributes)

        # ADD small correction
        gnn_in = x[:,:2] # charge and radius
        x = self.interaction1(edge_index=gnn_edge_index,x=gnn_in,edge_attributes=gnn_edge_attributes)
        x = self._silu(x)
        x = self.interaction2(edge_index=gnn_edge_index,x=x,edge_attributes=gnn_edge_attributes)
        x = self._silu(x)
        x = self.interaction3(edge_index=gnn_edge_index,x=x,edge_attributes=gnn_edge_attributes)

        energies = gbn_energies + x

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(energies.sum(), inputs=data.pos, create_graph=True)[0]
        forces = -1 * gradients

        if self._nobatch:
            energy = energies.sum()
            energy = energy.unsqueeze(0)
            energy = energy.unsqueeze(0)
        else:
            energy = torch.empty((torch.max(data.batch) + 1, 1), device=self._device)
            for batch in data.batch.unique():
                energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy, forces


class _GNN_fix_cuda:

    _lock_device = False

    def to(self, *args, **kwargs):
        if self._lock_device:
            pass
        else:
            super().to(*args, **kwargs)


class GNN3_all_swish_GBNeck_trainable_dif_graphs_corr_run(GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr,_GNN_fix_cuda):
    
    def __init__(self,fraction=0.5,radius=0.4, max_num_neighbors=32, parameters=None, device=None, jittable=False):

        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck_2.__init__(self,radius=gbneck_radius, max_num_neighbors=max_num_neighbors, parameters=parameters, device=device, jittable=jittable)
        GNN_Grapher_2.__init__(self,radius=radius, max_num_neighbors=max_num_neighbors)

        self._fraction = fraction
        if self._jittable:
            self.interaction1 = IN_layer_all_swish_2pass(3 + 3, 128,device=device).jittable()
            self.interaction2 = IN_layer_all_swish_2pass(128 + 128, 128,device=device).jittable()
            self.interaction3 = IN_layer_all_swish_2pass(128 + 128, 1,device=device).jittable()
        else:
            self.interaction1 = IN_layer_all_swish_2pass(3 + 3, 128,device=device)
            self.interaction2 = IN_layer_all_swish_2pass(128 + 128, 128,device=device)
            self.interaction3 = IN_layer_all_swish_2pass(128 + 128, 1,device=device)

        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, positions):
        positions = positions.to(dtype=torch.float,device=self._device)

        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(positions)
        _, gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(positions)

        # Get atom features
        x = self._gbparameters

        # Do message passing
        Bc = self.aggregate_information(x=x, edge_index=edge_index, edge_attributes=edge_attributes)  # B and charges

        # ADD small correction
        Bcn = torch.concat((Bc,x[:,1].unsqueeze(1)),dim=1)
        Bcn = self.interaction1(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:,0].unsqueeze(1) * (self._fraction + self.sigmoid(Bcn)*(1-self._fraction)*2)

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn,Bc[:,1].unsqueeze(1)),dim=1)

        energies = self.calculate_energies(x=Bc, edge_index=edge_index, edge_attributes=edge_attributes)
        return energies.sum()

    def build_gnn_graph(self, positions):

        # Extract edge index
        edge_index = torch_cluster.radius_graph(
            positions, self._gnn_radius, None, False, self._max_num_neighbors,
            'source_to_target')

        # Extract edge features
        distances = self._distancer(positions[edge_index[0]], positions[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return None, edge_index, edge_attributes

    def build_graph(self, positions):

        # Extract edge index
        edge_index = torch_cluster.radius_graph(
            positions, 10.0, None, False, self._max_num_neighbors,
            'source_to_target')

        # Extract edge features
        distances = self._distancer(positions[edge_index[0]], positions[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return None, edge_index, edge_attributes
