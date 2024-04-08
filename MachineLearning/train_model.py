import sys
import os
sys.path.append('../')
import pandas as pd
from GNN_Trainer import Trainer
from GNN_Models import *
from GNN_Loss_Functions import *

import argparse

parser = argparse.ArgumentParser(description='Run Model Training')
parser.add_argument('-b','--batchsize',default=32,type=int,help='Batchsize')
parser.add_argument('-p','--per',default=0.8,type=float,help='fraction of training')
parser.add_argument('-f','--fra',default=0.1,type=float,help='scaling parameter')
parser.add_argument('-r','--random',default=161311,type=int,help='random seed')
parser.add_argument('-ra','--radius',default=0.6,type=float,help='radius')
parser.add_argument('-l','--lr',default=0.001,type=float,help='learning rate')
parser.add_argument('-fpt','--ptfile',default='.',type=str,help='Pytorch file with stored training data')
parser.add_argument('-e','--epochs',default=30,type=int,help='epochs to train for')
parser.add_argument('-m','--modelid',default=0,type=int,help='Model_architecture')
parser.add_argument('-n','--name',default='',type=str,help='name of model')
parser.add_argument('-c','--clip',default=0,type=float,help='norm clipping')
args = parser.parse_args()

assert args.modelid in [128,96,64,48,32]

# make string of commands
model_inf_string = ''
for arg in vars(args):
    if arg == 'folder':
        continue
    if arg == 'mapping_file':
        continue
    if arg == 'ptfile':
        continue
    if arg == 'npfile':
        continue
    model_inf_string += '_%s_%s' % (arg, getattr(args, arg))


ptfile = os.environ['TMPDIR']+'/%i.pt' % args.random
os.system("cp %s %s" % (args.ptfile,ptfile))

trainer = Trainer(verbose=False,name='GNN3_pub_' + model_inf_string,path='trained_models',force_mode=True,enable_tmp_dir=False,random_state=args.random)

device = 'cuda'
trainer.explicit = True
print('load data',flush=True)
gbneck_parameters, unique_radii = trainer.prepare_training_data_from_pt_file(ptfile)
print('data loaded',flush=True)

model_class_dict = {128 : GNN3_scale_128,
                    96 : GNN3_scale_96,
                    64 : GNN3_scale_64,
                    48 : GNN3_scale_48,
                    32 : GNN3_scale_32}

model_class = model_class_dict[args.modelid]
model = model_class(radius=args.radius,max_num_neighbors=10000,parameters=gbneck_parameters,device=device,fraction=args.fra,unique_radii=unique_radii)

trainer.model = model
trainer.initialize_optimizer(args.lr,'Exponential30')
trainer.set_lossfunction(calculate_force_loss_only)
trainer.train_model(args.epochs,args.batchsize,args.clip)
trainer.save_model()
