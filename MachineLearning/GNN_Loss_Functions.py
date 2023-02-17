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