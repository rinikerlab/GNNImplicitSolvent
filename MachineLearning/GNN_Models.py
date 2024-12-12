"""
File to define Neural Networks
"""

import torch_cluster
from torch_geometric.transforms import RadiusGraph
from torch_geometric.nn import radius_graph
from torch.nn import PairwiseDistance
import torch
from torch import nn
from torch_scatter import scatter
from torch_sparse import SparseTensor
from typing import (
    Union,
    Tuple,
    Any,
    Callable,
    Iterator,
    Set,
    Optional,
    overload,
    TypeVar,
    Mapping,
    Dict,
    List,
)
from torch.cuda.amp import autocast

T = TypeVar("T", bound="Module")

from MachineLearning.GNN_Layers import *

from torch.nn.functional import one_hot

torch.backends.cudnn.benchmark = True


DEFAULT_UNIQUE_RADII = [0.14, 0.117, 0.155, 0.15, 0.21, 0.185, 0.18, 0.17, 0.12, 0.13]


class GNN_GBNeck(torch.nn.Module):
    def __init__(
        self,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        solvent_dielectric=78.5,
        no_batch=True,
    ):
        """
        GNN to reproduce the GBNeck Model
        """
        super().__init__()

        # In order to be differentiable all tensors *need* to be created on the same device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        self._nobatch = no_batch

        # Initiate Graph Builder
        if not parameters is None:
            self._gbparameters = torch.tensor(
                parameters, dtype=torch.float, device=self._device
            )
        self._radius = radius
        self._max_num_neighbors = max_num_neighbors
        self._grapher = RadiusGraph(
            r=self._radius, loop=False, max_num_neighbors=self._max_num_neighbors
        )

        # Init Distance Calculation
        self._distancer = PairwiseDistance()
        self._jittable = jittable
        if jittable:
            self.aggregate_information = GBNeck_interaction(
                parameters, self._device, unique_radii=unique_radii
            ).jittable()
            self.calculate_energies = GBNeck_energies(
                parameters,
                self._device,
                unique_radii=unique_radii,
                solvent_dielectric=solvent_dielectric,
            ).jittable()
        else:
            self.aggregate_information = GBNeck_interaction(
                parameters, self._device, unique_radii=unique_radii
            )
            self.calculate_energies = GBNeck_energies(
                parameters,
                self._device,
                unique_radii=unique_radii,
                solvent_dielectric=solvent_dielectric,
            )

        self.lin = nn.Linear(1, 1)

    def get_edge_features(
        self, distances, alpha=2, max_range=0.4, min_range=0.1, num_kernels=32
    ):
        m = alpha * (max_range - min_range) / (num_kernels + 1)
        lower_bound = min_range + m
        upper_bound = max_range - m
        centers = torch.linspace(
            lower_bound, upper_bound, num_kernels, device=self._device
        )
        k = distances - centers
        return torch.maximum(
            torch.tensor(0, device=self._device), torch.pow(1 - (k / m) ** 2, 3)
        )

    # @torch.compiler.disable
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

        x = self._gbparameters.repeat(torch.max(data.batch) + 1, 1)

        # Do message passing
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges
        energies = self.calculate_energies(
            x=Bc, edge_index=edge_index, edge_attributes=edge_attributes
        )

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(
            energies.sum(), inputs=data.pos, create_graph=True
        )[0]
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


class GNN_GBNeck_2(GNN_GBNeck):

    def forward(self, data):

        # Enable tracking of gradients
        # Get input as Tensor create on device
        data.pos = data.pos.clone().detach().requires_grad_(True)

        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)

        x = data.atom_features

        # Do message passing
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges
        energies = self.calculate_energies(
            x=Bc, edge_index=edge_index, edge_attributes=edge_attributes
        )

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(
            energies.sum(), inputs=data.pos, create_graph=True
        )[0]
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

    # @torch.compiler.disable
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


class GNN_GBNeck_2_multisolvent(GNN_GBNeck_2):

    def __init__(
        self,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        solvent_dielectric=78.5,
        no_batch=True,
    ):
        super().__init__(
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            unique_radii,
            solvent_dielectric,
            no_batch,
        )

        self.calculate_energies = GBNeck_energies_no_dielectric(
            parameters,
            self._device,
            unique_radii=unique_radii,
            solvent_dielectric=solvent_dielectric,
        )


class GNN_Grapher:

    def __init__(self, radius, max_num_neighbors) -> None:
        self._gnn_grapher = RadiusGraph(
            r=radius, loop=False, max_num_neighbors=max_num_neighbors
        )

    # @torch.compiler.disable
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

    # @torch.compiler.disable
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


class _GNN_fix_cuda:

    _lock_device = False

    def to(self, *args, **kwargs):
        if self._lock_device:
            pass
        else:
            super().to(*args, **kwargs)


class GNN3_Multisolvent_embedding(GNN_GBNeck_2_multisolvent, GNN_Grapher_2):

    def __init__(
        self,
        fraction=0.1,
        radius=0.6,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=DEFAULT_UNIQUE_RADII,
        hidden=64,
        num_solvents=39,
        dropout_rate=0.0,
        hidden_token=128,
        scaling_factor=2.0,
        no_batch=True,
    ):

        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck_2_multisolvent.__init__(
            self,
            radius=gbneck_radius,
            max_num_neighbors=max_num_neighbors,
            parameters=parameters,
            device=device,
            jittable=jittable,
            unique_radii=unique_radii,
            no_batch=no_batch,
        )
        GNN_Grapher_2.__init__(self, radius=radius, max_num_neighbors=max_num_neighbors)

        self._fraction = fraction
        self._scaling_factor = scaling_factor
        self.solvent_embedding = torch.nn.Embedding(num_solvents, hidden)
        self.gamma_embedding = torch.nn.Embedding(num_solvents, 1)
        self.interaction1 = IN_layer_all_swish_2pass_tokens(
            3 + 3,
            hidden,
            radius,
            device,
            hidden,
            hidden,
            dropout_rate,
            hidden_token,
        )
        self.interaction2 = IN_layer_all_swish_2pass_tokens(
            hidden + hidden,
            hidden,
            radius,
            device,
            hidden,
            hidden,
            dropout_rate,
            hidden_token,
        )
        self.interaction3 = IN_layer_all_swish_2pass_tokens(
            hidden + hidden,
            2,
            radius,
            device,
            hidden,
            hidden,
            dropout_rate,
            hidden_token,
        )

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.register_buffer(
            "_num_solvents",
            torch.tensor(num_solvents, dtype=torch.int, device=self._device),
        )

        self.register_buffer(
            "_no_batch",
            torch.tensor(no_batch, dtype=torch.bool, device=self._device),
        )

        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data.pos = data.pos.requires_grad_(True)

        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)
        _, gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)
        x = data.atom_features

        # Do message passing
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges

        # Embedd solvent
        solvent_embedding = self.solvent_embedding(data.solvent_model_tensor)

        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.dropout(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.dropout(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius + 0.14) ** 2

        # Each solvent has a learnable gamma parameter
        gamma = self.gamma_embedding(data.solvent_model_tensor)
        sa_energies = gamma * sasa

        # Scale the GBNeck born radii
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1))
            * (1 - self._fraction)
            * self._scaling_factor
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            solvent_dielectrics=data.solvent_dielectric.unsqueeze(1),
        )
        # Add SA term
        energies = energies + sa_energies

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(
            energies.sum(), inputs=data.pos, create_graph=True
        )[0]
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


class GNN3_Multisolvent_embedding_message(GNN_GBNeck_2_multisolvent, GNN_Grapher_2):

    def __init__(
        self,
        fraction=0.1,
        radius=0.6,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=DEFAULT_UNIQUE_RADII,
        hidden=64,
        num_solvents=39,
        dropout_rate=0.0,
        hidden_token=128,
        scaling_factor=2.0,
        no_batch=True,
    ):

        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck_2_multisolvent.__init__(
            self,
            radius=gbneck_radius,
            max_num_neighbors=max_num_neighbors,
            parameters=parameters,
            device=device,
            jittable=jittable,
            unique_radii=unique_radii,
            no_batch=no_batch,
        )
        GNN_Grapher_2.__init__(self, radius=radius, max_num_neighbors=max_num_neighbors)

        self._fraction = fraction
        self._scaling_factor = scaling_factor
        self.solvent_embedding = torch.nn.Embedding(num_solvents, hidden)
        self.gamma_embedding = torch.nn.Embedding(num_solvents, 1)
        self.interaction1 = IN_layer_all_swish_2pass_tokens_mes_tokens(
            3 + 3,
            hidden,
            radius,
            device,
            hidden,
            hidden,
            dropout_rate,
            hidden_token,
        )
        self.interaction2 = IN_layer_all_swish_2pass_tokens_mes_tokens(
            hidden + hidden,
            hidden,
            radius,
            device,
            hidden,
            hidden,
            dropout_rate,
            hidden_token,
        )
        self.interaction3 = IN_layer_all_swish_2pass_tokens_mes_tokens(
            hidden + hidden,
            2,
            radius,
            device,
            hidden,
            hidden,
            dropout_rate,
            hidden_token,
        )

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.register_buffer(
            "_num_solvents",
            torch.tensor(num_solvents, dtype=torch.int, device=self._device),
        )

        self.register_buffer(
            "_no_batch",
            torch.tensor(no_batch, dtype=torch.bool, device=self._device),
        )

        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def build_gnn_graph_messageat(self, data):

        # Get Radius Graph
        graph = self._gnn_grapher(data)

        # Extract edge index
        edge_index = graph.edge_index

        # Extract node features
        node_features = graph.atom_features

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])
        solvent_embeddings = data.solvent_model_tensor[edge_index[0]]

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)
        solvent_attributes = solvent_embeddings.unsqueeze(1)

        return node_features, edge_index, edge_attributes, solvent_attributes

    def forward(self, data):
        data.pos = data.pos.requires_grad_(True)

        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)
        _, gnn_edge_index, gnn_edge_attributes, solvent_attributes = (
            self.build_gnn_graph(data)
        )
        x = data.atom_features

        # Do message passing
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges

        # Embedd solvent
        solvent_embedding = self.solvent_embedding(data.solvent_model_tensor)
        solvent_embedding_attr = self.solvent_embedding(solvent_attributes)

        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.dropout(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.dropout(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius + 0.14) ** 2

        # Each solvent has a learnable gamma parameter
        gamma = self.gamma_embedding(data.solvent_model_tensor)
        sa_energies = gamma * sasa

        # Scale the GBNeck born radii
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1))
            * (1 - self._fraction)
            * self._scaling_factor
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            solvent_dielectrics=data.solvent_dielectric.unsqueeze(1),
        )
        # Add SA term
        energies = energies + sa_energies

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(
            energies.sum(), inputs=data.pos, create_graph=True
        )[0]
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


class GNN3_Multisolvent_embedding_run_multiple(GNN3_Multisolvent_embedding):

    def __init__(
        self,
        fraction=0.1,
        radius=0.6,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10.0,
        unique_radii=DEFAULT_UNIQUE_RADII,
        hidden=64,
        solvent_dielectric=[78.5],
        num_solvents=1,
        solvent_models=[0],
        hidden_token=128,
        scaling_factor=2.0,
    ):
        super().__init__(
            fraction=fraction,
            radius=radius,
            max_num_neighbors=max_num_neighbors,
            parameters=parameters,
            device=device,
            jittable=jittable,
            unique_radii=unique_radii,
            hidden=hidden,
            num_solvents=num_solvents,
            hidden_token=hidden_token,
            scaling_factor=scaling_factor,
        )

        self.to(self._device)
        self.set_num_reps(num_reps, solvent_models, solvent_dielectric)
        self._edge_index = self.build_edge_idx(len(parameters), num_reps).to(
            self._device
        )
        self._refzero = torch.zeros(1, dtype=torch.long, device=self._device)

    def set_num_reps(self, num_reps=1, solvent_models=[0], solvent_dielectric=[78.5]):

        self._num_reps = num_reps
        self._batch = torch.zeros(
            (num_reps * len(self._gbparameters)), dtype=torch.int64, device=self._device
        )
        self._solvent_model_tensor = torch.zeros(
            (num_reps * len(self._gbparameters)), dtype=torch.int64, device=self._device
        )
        self._solvent_dielectric = torch.zeros(
            (num_reps * len(self._gbparameters)), dtype=torch.float, device=self._device
        )
        for i in range(num_reps):
            self._batch[
                i * len(self._gbparameters) : (i + 1) * len(self._gbparameters)
            ] = i
            self._solvent_model_tensor[
                i * len(self._gbparameters) : (i + 1) * len(self._gbparameters)
            ] = solvent_models[i]
            self._solvent_dielectric[
                i * len(self._gbparameters) : (i + 1) * len(self._gbparameters)
            ] = solvent_dielectric[i]
        self._batch_gbparameters = self._gbparameters.repeat(self._num_reps, 1)

        # update edge index
        self._edge_index = self.build_edge_idx(len(self._gbparameters), num_reps).to(
            self._device
        )
        return 0

    def _compute_energy_tensors(self, positions, solvent_model, solvent_dielectric):
        """Compute the per-atom polar and apolar NN implicit solvent energy for given positions.

        :param positions: torch.tensor (on the correct device!) with the positions.
        :param solvent_model: int, index of the desired solvent in Simulation/solvents.yml
        :param solvent_dielectric: float, dielectric of the solvent matching Simulation/solvents.yml
        :return energies: tuple of torch.tensors for polar and apolar energies.
        """

        if solvent_model != -1:
            self._solvent_model_tensor = torch.tensor(
                solvent_model, dtype=torch.int64, device=self._device
            ).repeat(len(self._gbparameters))
            self._solvent_dielectric = torch.tensor(
                solvent_dielectric, dtype=torch.float, device=self._device
            ).repeat(len(self._gbparameters))
            # Encode solvent

        # Get Graph
        if positions.device != self._device:
            positions = positions.float().to(self._device)
        
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(positions)
        gnn_slices = edge_attributes < 0.6
        sgnn_slices = torch.squeeze(gnn_slices)
        gnn_edge_attributes = torch.unsqueeze(edge_attributes[gnn_slices], 1)
        gnn_edge_index = torch.cat(
            (
                torch.unsqueeze(edge_index[0][sgnn_slices], 0),
                torch.unsqueeze(edge_index[1][sgnn_slices], 0),
            ),
            dim=0,
        )

        # Get atom features
        x = self._batch_gbparameters
        # Do message passinge
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges

        # Embedd solvent
        solvent_embedding = self.solvent_embedding(self._solvent_model_tensor)

        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius + 0.14) ** 2

        # Each solvent has a learnable gamma parameter
        gamma = self.gamma_embedding(self._solvent_model_tensor)
        sa_energies = gamma * sasa

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1))
            * (1 - self._fraction)
            * self._scaling_factor
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            solvent_dielectrics=self._solvent_dielectric.unsqueeze(1),
        )

        return energies, sa_energies

    def forward(self, positions, solvent_model, solvent_dielectric):
        """Compute the total NN implicit solvent energy for given positions.

        :param positions: torch.tensor (on the correct device!) with the positions.
        :param solvent_model: int, index of the desired solvent in Simulation/solvents.yml
        :param solvent_dielectric: float, dielectric of the solvent matching Simulation/solvents.yml
        :return energy: float, predicted solvation free energy
        """
        energies, sa_energies = self._compute_energy_tensors(
            positions, solvent_model, solvent_dielectric
        )
        total = energies.sum() + sa_energies.sum()
        return total

    def build_gnn_graph(self, positions):

        # Extract edge index
        edge_index = torch_cluster.radius_graph(
            positions,
            self._gnn_radius,
            self._batch,
            False,
            self._max_num_neighbors,
            "source_to_target",
        )

        # Extract edge features
        distances = self._distancer(positions[edge_index[0]], positions[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return None, edge_index, edge_attributes

    def build_graph(self, positions):

        edge_index = self._edge_index

        # Extract edge features
        distances = self._distancer(positions[edge_index[0]], positions[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return None, edge_index, edge_attributes

    def build_edge_idx(self, num_nodes, num_reps):

        elements_per_rep = num_nodes * (num_nodes - 1)
        edge_index = torch.zeros(
            (2, num_reps * elements_per_rep), dtype=torch.long, device=self._device
        )

        for rep in range(num_reps):
            for node in range(num_nodes):
                for con in range(num_nodes):
                    if con < node:
                        edge_index[
                            0, rep * elements_per_rep + node * (num_nodes - 1) + con
                        ] = (rep * num_nodes + node)
                        edge_index[
                            1, rep * elements_per_rep + node * (num_nodes - 1) + con
                        ] = (rep * num_nodes + con)
                    elif con > node:
                        edge_index[
                            0, rep * elements_per_rep + node * (num_nodes - 1) + con - 1
                        ] = (rep * num_nodes + node)
                        edge_index[
                            1, rep * elements_per_rep + node * (num_nodes - 1) + con - 1
                        ] = (rep * num_nodes + con)

        return edge_index


class GNN3_Multisolvent_embedding_run_multiple_Delta(
    GNN3_Multisolvent_embedding_run_multiple
):

    def _compute_energy_tensors(self, positions, solvent_model, solvent_dielectric):
        if solvent_model != -1:
            self._solvent_model_tensor = torch.tensor(
                solvent_model, dtype=torch.int64, device=self._device
            ).repeat(len(self._gbparameters))
            self._solvent_dielectric = torch.tensor(
                solvent_dielectric, dtype=torch.float, device=self._device
            ).repeat(len(self._gbparameters))
            # Encode solvent

        # Get Graph
        if positions.device != self._device:
            positions = positions.float().to(self._device)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(positions)
        gnn_slices = edge_attributes < 0.6
        sgnn_slices = torch.squeeze(gnn_slices)
        gnn_edge_attributes = torch.unsqueeze(edge_attributes[gnn_slices], 1)
        gnn_edge_index = torch.cat(
            (
                torch.unsqueeze(edge_index[0][sgnn_slices], 0),
                torch.unsqueeze(edge_index[1][sgnn_slices], 0),
            ),
            dim=0,
        )

        # Get atom features
        x = self._batch_gbparameters
        # Do message passinge
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges

        # Compute GBNeck energies
        gb_energies = self.calculate_energies(
            x=Bc,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            solvent_dielectrics=self._solvent_dielectric.unsqueeze(1),
        )

        # Embedd solvent
        solvent_embedding = self.solvent_embedding(self._solvent_model_tensor)

        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius + 0.14) ** 2

        # Each solvent has a learnable gamma parameter
        gamma = self.gamma_embedding(self._solvent_model_tensor)
        sa_energies = gamma * sasa

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1))
            * (1 - self._fraction)
            * self._scaling_factor
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            solvent_dielectrics=self._solvent_dielectric.unsqueeze(1),
        )

        return energies - gb_energies, sa_energies

    def forward(self,positions):
        """Compute the total NN implicit solvent energy for given positions.

        :param positions: torch.tensor (on the correct device!) with the positions.
        :param solvent_model: int, index of the desired solvent in Simulation/solvents.yml
        :param solvent_dielectric: float, dielectric of the solvent matching Simulation/solvents.yml
        :return energy: float, predicted solvation free energy
        """
        solvent_model = torch.tensor(-1, dtype=torch.int64, device=self._device)
        solvent_dielectric = torch.tensor(78.5, dtype=torch.float, device=self._device)

        energies, sa_energies = self._compute_energy_tensors(positions, solvent_model, solvent_dielectric)
        total = energies.sum() + sa_energies.sum()
        return total


class GNN3_Multisolvent_embedding_run_multiple_split(
    GNN3_Multisolvent_embedding_run_multiple
):

    def forward(self, positions, solvent_model, solvent_dielectric):
        """Compute the per-atom polar and apolar NN implicit solvent energy for given positions.

        :param positions: torch.tensor (on the correct device!) with the positions.
        :param solvent_model: int, index of the desired solvent in Simulation/solvents.yml
        :param solvent_dielectric: float, dielectric of the solvent matching Simulation/solvents.yml
        :return energy: tuple of torch.tensors for polar and apolar energies.
        """
        energies, sa_energies = self._compute_energy_tensors(
            positions, solvent_model, solvent_dielectric
        )
        return energies, sa_energies


class GNN3_multisolvent(GNN_GBNeck_2_multisolvent, GNN_Grapher_2):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        hidden=128,
        num_solvents=1,
        dropout_rate=0.0,
    ):

        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck_2_multisolvent.__init__(
            self,
            radius=gbneck_radius,
            max_num_neighbors=max_num_neighbors,
            parameters=parameters,
            device=device,
            jittable=jittable,
            unique_radii=unique_radii,
        )
        GNN_Grapher_2.__init__(self, radius=radius, max_num_neighbors=max_num_neighbors)

        self._fraction = fraction
        if self._jittable:
            self.interaction1 = IN_layer_all_swish_2pass_tokens(
                3 + 3, hidden, radius, device, hidden, num_solvents
            ).jittable()
            self.interaction2 = IN_layer_all_swish_2pass_tokens(
                hidden + hidden, hidden, radius, device, hidden, num_solvents
            ).jittable()
            self.interaction3 = IN_layer_all_swish_2pass_tokens(
                hidden + hidden, 2, radius, device, hidden, num_solvents
            ).jittable()
            self.token_lin1 = nn.Linear(num_solvents, num_solvents)
            self.token_lin2 = nn.Linear(num_solvents, num_solvents)
        else:
            self.interaction1 = IN_layer_all_swish_2pass_tokens(
                3 + 3, hidden, radius, device, hidden, num_solvents
            )
            self.interaction2 = IN_layer_all_swish_2pass_tokens(
                hidden + hidden, hidden, radius, device, hidden, num_solvents
            )
            self.interaction3 = IN_layer_all_swish_2pass_tokens(
                hidden + hidden, 2, radius, device, hidden, num_solvents
            )
            self.token_lin1 = nn.Linear(num_solvents, num_solvents)
            self.token_lin2 = nn.Linear(num_solvents, num_solvents)

        self.register_buffer(
            "_num_solvents",
            torch.tensor(num_solvents, dtype=torch.int, device=self._device),
        )
        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)
        _, gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)
        x = data.atom_features

        # Do message passing
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges

        # Encode solvent
        solvent_embedding = self.token_lin1(
            one_hot(data.solvent_model_tensor, self._num_solvents).float()
        )
        solvent_embedding = self._silu(solvent_embedding)
        solvent_embedding = self.token_lin2(solvent_embedding)

        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            solvent_embedding=solvent_embedding,
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        gamma = 0.00542  # kcal/(mol A^2)
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius + 0.14) ** 2
        sa_energies = 4.184 * gamma * sasa * 100

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1)) * (1 - self._fraction) * 2
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            solvent_dielectrics=data.solvent_dielectric.unsqueeze(1),
        )
        # Add SA term
        energies = energies + sa_energies

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(
            energies.sum(), inputs=data.pos, create_graph=True
        )[0]
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

    def get_solvent_embedding(self):

        pass


class GNN3_multisolvent_run_multiple(GNN3_multisolvent):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10.0,
        unique_radii=None,
        hidden=128,
        solvent_dielectric=78.5,
        num_solvents=1,
        solvent_models=[0],
    ):
        super().__init__(
            fraction=fraction,
            radius=radius,
            max_num_neighbors=max_num_neighbors,
            parameters=parameters,
            device=device,
            jittable=jittable,
            unique_radii=unique_radii,
            hidden=hidden,
            num_solvents=num_solvents,
        )

        self.to(self._device)
        self.set_num_reps(num_reps)
        self._edge_index = self.build_edge_idx(len(parameters), num_reps).to(
            self._device
        )
        self._refzero = torch.zeros(1, dtype=torch.long, device=self._device)

        # self.set_num_reps(num_reps)

    def set_num_reps(self, num_reps=1, solvent_models=[0], solvent_dielectric=[78.5]):

        self._num_reps = num_reps
        self._batch = torch.zeros(
            (num_reps * len(self._gbparameters)), dtype=torch.int64, device=self._device
        )
        self._solvent_model_tensor = torch.zeros(
            (num_reps * len(self._gbparameters)), dtype=torch.int64, device=self._device
        )
        self._solvent_dielectric = torch.zeros(
            (num_reps * len(self._gbparameters)), dtype=torch.float, device=self._device
        )
        for i in range(num_reps):
            self._batch[
                i * len(self._gbparameters) : (i + 1) * len(self._gbparameters)
            ] = i
            self._solvent_model_tensor[
                i * len(self._gbparameters) : (i + 1) * len(self._gbparameters)
            ] = solvent_models[i]
            self._solvent_dielectric[
                i * len(self._gbparameters) : (i + 1) * len(self._gbparameters)
            ] = solvent_dielectric[i]
        self._batch_gbparameters = self._gbparameters.repeat(self._num_reps, 1)

        # Encode solvent
        tokens = self.token_lin1(
            one_hot(self._solvent_model_tensor, self._num_solvents).float()
        )
        tokens = self._silu(tokens)
        self._tokens = self.token_lin2(tokens)

        return 0

    def forward(self, positions, solvent_model, solvent_dielectric):

        if solvent_model != -1:
            self._solvent_model_tensor = torch.tensor(
                solvent_model, dtype=torch.int64, device=self._device
            ).repeat(len(self._gbparameters))
            self._solvent_dielectric = torch.tensor(
                solvent_dielectric, dtype=torch.float, device=self._device
            ).repeat(len(self._gbparameters))
            # Encode solvent
            tokens = self.token_lin1(
                one_hot(self._solvent_model_tensor, self._num_solvents).float()
            )
            tokens = self._silu(tokens)
            self._tokens = self.token_lin2(tokens)

        if positions.device != self._device:
            positions = positions.float().to(self._device)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(positions)
        gnn_slices = edge_attributes < 0.6
        sgnn_slices = torch.squeeze(gnn_slices)
        gnn_edge_attributes = torch.unsqueeze(edge_attributes[gnn_slices], 1)
        gnn_edge_index = torch.cat(
            (
                torch.unsqueeze(edge_index[0][sgnn_slices], 0),
                torch.unsqueeze(edge_index[1][sgnn_slices], 0),
            ),
            dim=0,
        )

        # Get atom features
        x = self._batch_gbparameters
        # Do message passinge
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges

        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            tokens=self._tokens,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            tokens=self._tokens,
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index,
            x=Bcn,
            edge_attributes=gnn_edge_attributes,
            tokens=self._tokens,
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        gamma = 0.00542  # kcal/(mol A^2)
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius + 0.14) ** 2
        sa_energies = 4.184 * gamma * sasa * 100

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1)) * (1 - self._fraction) * 2
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            solvent_dielectrics=self._solvent_dielectric.unsqueeze(1),
        )

        # Add SA term
        energies = energies + sa_energies
        return energies.sum()

    def build_gnn_graph(self, positions):

        # Extract edge index
        edge_index = torch_cluster.radius_graph(
            positions,
            self._gnn_radius,
            self._batch,
            False,
            self._max_num_neighbors,
            "source_to_target",
        )

        # Extract edge features
        distances = self._distancer(positions[edge_index[0]], positions[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return None, edge_index, edge_attributes

    def build_graph(self, positions):

        edge_index = self._edge_index

        # Extract edge features
        distances = self._distancer(positions[edge_index[0]], positions[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return None, edge_index, edge_attributes

    def build_edge_idx(self, num_nodes, num_reps):

        elements_per_rep = num_nodes * (num_nodes - 1)
        edge_index = torch.zeros(
            (2, num_reps * elements_per_rep), dtype=torch.long, device=self._device
        )

        for rep in range(num_reps):
            for node in range(num_nodes):
                for con in range(num_nodes):
                    if con < node:
                        edge_index[
                            0, rep * elements_per_rep + node * (num_nodes - 1) + con
                        ] = (rep * num_nodes + node)
                        edge_index[
                            1, rep * elements_per_rep + node * (num_nodes - 1) + con
                        ] = (rep * num_nodes + con)
                    elif con > node:
                        edge_index[
                            0, rep * elements_per_rep + node * (num_nodes - 1) + con - 1
                        ] = (rep * num_nodes + node)
                        edge_index[
                            1, rep * elements_per_rep + node * (num_nodes - 1) + con - 1
                        ] = (rep * num_nodes + con)

        return edge_index


class GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA(
    GNN_GBNeck_2, GNN_Grapher_2
):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        hidden=128,
        solvent_dielectric=78.5,
    ):

        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck_2.__init__(
            self,
            radius=gbneck_radius,
            max_num_neighbors=max_num_neighbors,
            parameters=parameters,
            device=device,
            jittable=jittable,
            unique_radii=unique_radii,
            solvent_dielectric=solvent_dielectric,
        )
        GNN_Grapher_2.__init__(self, radius=radius, max_num_neighbors=max_num_neighbors)

        self._fraction = fraction
        if self._jittable:
            self.interaction1 = IN_layer_all_swish_2pass(
                3 + 3, hidden, radius, device, hidden
            ).jittable()
            self.interaction2 = IN_layer_all_swish_2pass(
                hidden + hidden, hidden, radius, device, hidden
            ).jittable()
            self.interaction3 = IN_layer_all_swish_2pass(
                hidden + hidden, 2, radius, device, hidden
            ).jittable()
        else:
            self.interaction1 = IN_layer_all_swish_2pass(
                3 + 3, hidden, radius, device, hidden
            )
            self.interaction2 = IN_layer_all_swish_2pass(
                hidden + hidden, hidden, radius, device, hidden
            )
            self.interaction3 = IN_layer_all_swish_2pass(
                hidden + hidden, 2, radius, device, hidden
            )

        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)
        _, gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)
        x = data.atom_features

        # Do message passing
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges

        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        gamma = 0.00542  # kcal/(mol A^2)
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius + 0.14) ** 2
        sa_energies = 4.184 * gamma * sasa * 100

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1)) * (1 - self._fraction) * 2
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc, edge_index=edge_index, edge_attributes=edge_attributes
        )

        # Add SA term
        energies = energies + sa_energies

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(
            energies.sum(), inputs=data.pos, create_graph=True
        )[0]
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


class GNN3_scale_64(
    GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA
):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            unique_radii,
            hidden,
            solvent_dielectric,
        )


class GNN3_scale_64_SA_flex(
    GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA
):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            unique_radii,
            hidden,
            solvent_dielectric,
        )

        self._sa_scale = nn.Parameter(
            torch.tensor([0.0], device=self._device), requires_grad=True
        )
        self._sa_probe_radius = nn.Parameter(
            torch.tensor([0.0], device=self._device), requires_grad=True
        )

    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)
        _, gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)
        x = data.atom_features

        # Do message passing
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges

        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        gamma = 0.00542  # kcal/(mol A^2)
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = (1 - 2 * self.sigmoid(sa_scale.unsqueeze(1))) * (
            radius + self._sa_probe_radius
        ) ** 2
        # sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius+self._sa_probe_radius)**2
        # sa_energies = 4.184 * gamma * sasa * 2000

        # Let the GNN decide on the scaling
        sa_energies = sasa * self._sa_scale

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1)) * (1 - self._fraction) * 2
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc, edge_index=edge_index, edge_attributes=edge_attributes
        )

        # Add SA term
        energies = energies + sa_energies

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(
            energies.sum(), inputs=data.pos, create_graph=True
        )[0]
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


class GNN3_scale_64_gamma_flex(GNN3_scale_64_SA_flex):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            unique_radii,
            hidden,
            solvent_dielectric,
        )

        self._sa_scale = nn.Parameter(
            torch.tensor([2.27], device=self._device), requires_grad=True
        )
        self._sa_probe_radius = nn.Parameter(
            torch.tensor([0.14], device=self._device), requires_grad=False
        )


class GNN3_scale_64_gamma_fix_10(GNN3_scale_64_SA_flex):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            unique_radii,
            hidden,
            solvent_dielectric,
        )

        self._sa_scale = nn.Parameter(
            torch.tensor([22.7], device=self._device), requires_grad=False
        )
        self._sa_probe_radius = nn.Parameter(
            torch.tensor([0.14], device=self._device), requires_grad=False
        )


class GNN3_scale_64_gamma_fix_100(GNN3_scale_64_SA_flex):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            unique_radii,
            hidden,
            solvent_dielectric,
        )

        self._sa_scale = nn.Parameter(
            torch.tensor([227], device=self._device), requires_grad=False
        )
        self._sa_probe_radius = nn.Parameter(
            torch.tensor([0.14], device=self._device), requires_grad=False
        )


class GNN3_scale_64_beta_flex(GNN3_scale_64):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            unique_radii,
            hidden,
            solvent_dielectric,
        )

        self._beta = nn.Parameter(
            torch.tensor([1.0], device=self._device), requires_grad=True
        )

    def forward(self, data):

        energies, forces = super().forward(data)
        return self._beta * energies, self._beta * forces


class GNN3_scale_64_beta_fix(GNN3_scale_64):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=32,
        parameters=None,
        device=None,
        jittable=False,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            unique_radii,
            hidden,
            solvent_dielectric,
        )

        self._beta = nn.Parameter(
            torch.tensor([0.5], device=self._device), requires_grad=False
        )

    def forward(self, data):

        energies, forces = super().forward(data)
        return self._beta * energies, self._beta * forces


class GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA_run_multiple(
    GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA,
    _GNN_fix_cuda,
):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10.0,
        unique_radii=None,
        hidden=128,
        solvent_dielectric=78.5,
        print_separate_energies=False,
    ):

        max_num_neighbors = 10000
        self._gnn_radius = radius
        self._print_separate_energies = print_separate_energies
        GNN_GBNeck_2.__init__(
            self,
            radius=gbneck_radius,
            max_num_neighbors=max_num_neighbors,
            parameters=parameters,
            device=device,
            jittable=jittable,
            unique_radii=unique_radii,
            solvent_dielectric=solvent_dielectric,
        )
        GNN_Grapher_2.__init__(self, radius=radius, max_num_neighbors=max_num_neighbors)

        self.set_num_reps(num_reps)

        self._fraction = fraction
        if self._jittable:
            self.interaction1 = IN_layer_all_swish_2pass(
                3 + 3, hidden, radius, device, hidden
            ).jittable()
            self.interaction2 = IN_layer_all_swish_2pass(
                hidden + hidden, hidden, radius, device, hidden
            ).jittable()
            self.interaction3 = IN_layer_all_swish_2pass(
                hidden + hidden, 2, radius, device, hidden
            ).jittable()
        else:
            self.interaction1 = IN_layer_all_swish_2pass(
                3 + 3, hidden, radius, device, hidden
            )
            self.interaction2 = IN_layer_all_swish_2pass(
                hidden + hidden, hidden, radius, device, hidden
            )
            self.interaction3 = IN_layer_all_swish_2pass(
                hidden + hidden, 2, radius, device, hidden
            )

        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self._edge_index = self.build_edge_idx(len(parameters), num_reps).to(
            self._device
        )
        self._refzero = torch.zeros(1, dtype=torch.long, device=self._device)

    def set_num_reps(self, num_reps=1):

        self._num_reps = num_reps
        self._batch = torch.zeros(
            (num_reps * len(self._gbparameters)), dtype=torch.int64, device=self._device
        )
        for i in range(num_reps):
            self._batch[
                i * len(self._gbparameters) : (i + 1) * len(self._gbparameters)
            ] = i

        self._batch_gbparameters = self._gbparameters.repeat(self._num_reps, 1)

        return 0

    def _compute_energy_tensors(self, positions):

        if positions.device != self._device:
            positions = positions.float().to(self._device)

        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(positions)
        gnn_slices = edge_attributes < 0.6
        sgnn_slices = torch.squeeze(gnn_slices)
        gnn_edge_attributes = torch.unsqueeze(edge_attributes[gnn_slices], 1)
        gnn_edge_index = torch.cat(
            (
                torch.unsqueeze(edge_index[0][sgnn_slices], 0),
                torch.unsqueeze(edge_index[1][sgnn_slices], 0),
            ),
            dim=0,
        )

        # Get atom features
        x = self._batch_gbparameters

        # Do message passinge
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges
        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        gamma = 0.00542  # kcal/(mol A^2)
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius + 0.14) ** 2
        sa_energies = 4.184 * gamma * sasa * 100

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1)) * (1 - self._fraction) * 2
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc, edge_index=edge_index, edge_attributes=edge_attributes
        )

        if self._print_separate_energies:
            print("GBNeck+SA energies")
            print(energies + sa_energies)
            print("SA energies")
            print(sa_energies)

        return energies, sa_energies

    def forward(self, positions):
        energies, sa_energies = self._compute_energy_tensors(positions)
        return (energies + sa_energies).sum()

    def build_gnn_graph(self, positions):

        # Extract edge index
        edge_index = torch_cluster.radius_graph(
            positions,
            self._gnn_radius,
            self._batch,
            False,
            self._max_num_neighbors,
            "source_to_target",
        )

        # Extract edge features
        distances = self._distancer(positions[edge_index[0]], positions[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return None, edge_index, edge_attributes

    def build_graph(self, positions):

        edge_index = self._edge_index

        # Extract edge features
        distances = self._distancer(positions[edge_index[0]], positions[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return None, edge_index, edge_attributes

    def build_edge_idx(self, num_nodes, num_reps):

        elements_per_rep = num_nodes * (num_nodes - 1)
        edge_index = torch.zeros(
            (2, num_reps * elements_per_rep), dtype=torch.long, device=self._device
        )

        for rep in range(num_reps):
            for node in range(num_nodes):
                for con in range(num_nodes):
                    if con < node:
                        edge_index[
                            0, rep * elements_per_rep + node * (num_nodes - 1) + con
                        ] = (rep * num_nodes + node)
                        edge_index[
                            1, rep * elements_per_rep + node * (num_nodes - 1) + con
                        ] = (rep * num_nodes + con)
                    elif con > node:
                        edge_index[
                            0, rep * elements_per_rep + node * (num_nodes - 1) + con - 1
                        ] = (rep * num_nodes + node)
                        edge_index[
                            1, rep * elements_per_rep + node * (num_nodes - 1) + con - 1
                        ] = (rep * num_nodes + con)

        return edge_index


class GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA_run_multiple_split(
    GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA_run_multiple
):
    def forward(self, positions):
        energies, sa_energies = self._compute_energy_tensors(positions)
        return energies, sa_energies


class GNN3_scale_64_run(
    GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA_run_multiple
):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
        print_separate_energies=False,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            num_reps,
            gbneck_radius,
            unique_radii,
            hidden,
            solvent_dielectric,
            print_separate_energies,
        )


class GNN3_scale_64_run_split(
    GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA_run_multiple_split,
    GNN3_scale_64_run,
):
    pass


class GNN3_scale_64_lambda_run(GNN3_scale_64_run):

    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lambda = nn.Parameter(
            torch.tensor([1.0], device=self._device), requires_grad=True
        )

    def forward(*args, **kwargs):

        energies = super().forward(*args, **kwargs)

        return self._lambda * energies


class GNN3_scale_64_run_tip3p_beta(
    GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA_run_multiple
):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
        print_separate_energies=False,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            num_reps,
            gbneck_radius,
            unique_radii,
            hidden,
            solvent_dielectric,
            print_separate_energies,
        )

        self._beta = nn.Parameter(
            torch.tensor([0.5], device=self._device), requires_grad=False
        )

    def forward(self, positions):

        if positions.device != self._device:
            positions = positions.float().to(self._device)

        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(positions)
        gnn_slices = edge_attributes < 0.6
        sgnn_slices = torch.squeeze(gnn_slices)
        gnn_edge_attributes = torch.unsqueeze(edge_attributes[gnn_slices], 1)
        gnn_edge_index = torch.cat(
            (
                torch.unsqueeze(edge_index[0][sgnn_slices], 0),
                torch.unsqueeze(edge_index[1][sgnn_slices], 0),
            ),
            dim=0,
        )

        # Get atom features
        x = self._batch_gbparameters

        # Do message passinge
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges
        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        gamma = 0.00542  # kcal/(mol A^2)
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius + 0.14) ** 2
        sa_energies = 4.184 * gamma * sasa * 100

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1)) * (1 - self._fraction) * 2
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc, edge_index=edge_index, edge_attributes=edge_attributes
        )

        # Add SA term
        energies = energies + sa_energies

        if self._print_separate_energies:
            print("GBNeck energies")
            print(energies)
            print("SA energies")
            print(sa_energies)

        return self._beta * energies.sum()


class GNN3_scale_64_SA_flex_run(
    GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA_run_multiple
):

    def __init__(
        self,
        fraction=0.5,
        radius=0.4,
        max_num_neighbors=10000,
        parameters=None,
        device=None,
        jittable=False,
        num_reps=1,
        gbneck_radius=10,
        unique_radii=None,
        hidden=64,
        solvent_dielectric=78.5,
        print_separate_energies=False,
    ):
        super().__init__(
            fraction,
            radius,
            max_num_neighbors,
            parameters,
            device,
            jittable,
            num_reps,
            gbneck_radius,
            unique_radii,
            hidden,
            solvent_dielectric,
            print_separate_energies,
        )

        self._sa_scale = nn.Parameter(
            torch.tensor([0.0], device=self._device), requires_grad=True
        )
        self._sa_probe_radius = nn.Parameter(
            torch.tensor([0.0], device=self._device), requires_grad=True
        )

    def _compute_energy_tensors(self, positions):

        if positions.device != self._device:
            positions = positions.float().to(self._device)

        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(positions)
        gnn_slices = edge_attributes < 0.6
        sgnn_slices = torch.squeeze(gnn_slices)
        gnn_edge_attributes = torch.unsqueeze(edge_attributes[gnn_slices], 1)
        gnn_edge_index = torch.cat(
            (
                torch.unsqueeze(edge_index[0][sgnn_slices], 0),
                torch.unsqueeze(edge_index[1][sgnn_slices], 0),
            ),
            dim=0,
        )

        # Get atom features
        x = self._batch_gbparameters

        # Do message passinge
        Bc = self.aggregate_information(
            x=x, edge_index=edge_index, edge_attributes=edge_attributes
        )  # B and charges
        # ADD small correction
        Bcn = torch.concat((Bc, x[:, 1].unsqueeze(1)), dim=1)
        Bcn = self.interaction1(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(
            edge_index=gnn_edge_index, x=Bcn, edge_attributes=gnn_edge_attributes
        )

        # Separate into polar and non-polar contributions
        c_scale = Bcn[:, 0]
        sa_scale = Bcn[:, 1]

        # Calculate SA term
        gamma = 0.00542  # kcal/(mol A^2)
        offset = 0.0195141
        radius = (x[:, 1] + offset).unsqueeze(1)
        sasa = (1 - 2 * self.sigmoid(sa_scale.unsqueeze(1))) * (
            radius + self._sa_probe_radius
        ) ** 2

        # Let the GNN decide on the scaling
        sa_energies = sasa * self._sa_scale

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:, 0].unsqueeze(1) * (
            self._fraction
            + self.sigmoid(c_scale.unsqueeze(1)) * (1 - self._fraction) * 2
        )

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn, Bc[:, 1].unsqueeze(1)), dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(
            x=Bc, edge_index=edge_index, edge_attributes=edge_attributes
        )

        if self._print_separate_energies:
            print("GBNeck+SA energies")
            print((energies + sa_energies).sum())
            print("SA energies")
            print(sa_energies.sum())

        return energies, sa_energies

    def forward(self, positions):
        energies, sa_energies = self._compute_energy_tensors(positions)
        return energies.sum() + sa_energies.sum()


class GNN3_scale_64_SA_flex_run_split(GNN3_scale_64_SA_flex_run):
    def forward(self, positions):
        energies, sa_energies = self._compute_energy_tensors(positions)
        return energies, sa_energies
