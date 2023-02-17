import torch
import numpy as np
from numba import njit
from torch_geometric.data import Data

def get_Graph_for_one_frame(data,atom_features,cutoff = 0.4,y=None):
    '''
    Calculate the Graph
    :param data:
    :param atom_features:
    :param cutoff:
    :param y:
    :return:
    '''
    data_tensor = torch.tensor(data)
    distanc_matrix = torch.cdist(data_tensor,data_tensor)

    ## Get Connection Graph
    # Adding Self Loops on the fly with >= 0
    connections = ((distanc_matrix >= 0) & (distanc_matrix < cutoff)).nonzero()
    con_from = connections[:, 0]
    con_to = connections[:, 1]
    edge_index = connections.swapaxes(0,1)

    ## Get Edge features
    distances = distanc_matrix[(con_from,con_to)]
    distances = distances.unsqueeze(1)
    edge_attributes = get_edge_features(distances)
    edge_attributes = edge_attributes.float()

    ## Get Node features
    node_features = torch.tensor(atom_features,dtype=torch.float)

    ## Check if target values exist
    if y is not None:
        y = torch.tensor(-1/y,dtype=torch.float)
        y = y.unsqueeze(1)

    # Create Data
    if y is not None:
        data = Data(x=node_features,edge_index=edge_index,edge_attributes=edge_attributes,y=y)
    else:
        data = Data(x=node_features, edge_index=edge_index, edge_attributes=edge_attributes)
    return data

def get_edge_features(distances,alpha=2,max_range=0.4,min_range=0.1,num_kernels=32):
    m = alpha * (max_range - min_range) / (num_kernels +1)
    lower_bound = min_range + m
    upper_bound = max_range - m
    centers = torch.linspace(lower_bound,upper_bound,num_kernels)
    k = distances - centers
    return torch.maximum(torch.tensor(0),torch.pow(1-(k/m)**2,3))