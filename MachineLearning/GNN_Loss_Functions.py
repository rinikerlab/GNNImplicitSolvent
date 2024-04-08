'''
File defining LosssFunctions for the training of GNNs
'''
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

def calculate_force_loss_only(pre_energy, pre_forces, ldata):
    loss = F.mse_loss(pre_forces, ldata.forces)
    return loss

def calculate_force_loss_per_molecule(pre_energy,pre_forces,ldata):
    loss = F.mse_loss(pre_forces, ldata.forces,reduction='none')
    loss = torch.sum(loss,dim=1).squeeze()
    individual_losses = torch.bincount(ldata.batch, weights=loss) / (torch.bincount(ldata.batch) * 3)
    return individual_losses