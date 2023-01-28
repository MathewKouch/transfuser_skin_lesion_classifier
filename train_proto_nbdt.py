#!/usr/bin/env python
# coding: utf-8

# Proto NBDT | Fusing NBDT and Prototypical Learning with Class Guided Metric

# Training model with prototypes decision tree classifier head

# Importing Dependencies

import os # directory
from PIL import ImageFile,Image # reading image
ImageFile.LOAD_TRUNCATED_IMAGES = True # Avoiding truncation OSError 
from datetime import datetime
import torch
import torch.nn as nn # For NN modules
import torch.nn.functional as F # For activations and utilies
import torch.optim as optim
import torchvision.models as models


import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from scipy.special import softmax
import math
import random
import pandas as pd
import imageio
from tqdm import tqdm # For progress bar
from collections import OrderedDict
import pickle
from IPython import display
import copy

from my_dataset2 import ClassDataset_CSV
from my_utilities2 import set_seed, get_cost_matrix
from augmentations import AllTransforms, MyRotationTransform_, MyZoomTransform_

# Baselines
from models_baseline import Transfuser, LateFusion, ScoreFusion

# Prototypes
from prototypical import DistortionLoss, PrototypicalNetwork

# Tree
from models_proto_nbdt import Hierarchy_prototype

# Prototypes and Tree
from proto_nbdt import PROTO_NBDT, train_proto_nbdt, eval_proto_nbdt, test_proto_nbdt

set_seed(42)
import argparse
import wandb
wandb.login()
print(f'Is cuda available? {torch.cuda.is_available()}')

# ## Constance and Global variables

parser = argparse.ArgumentParser(description='proto-nbdt')

parser.add_argument('--name', help='name of run. make this unique', required=True)
parser.add_argument('--model', help='type of model', required=True)
parser.add_argument('--nbdt', help='train with nbdt', default='False', required=True)
parser.add_argument('--l_metric', help='distortionloss weight', type=float, default=0.0, required=True)

parser.add_argument('--lm', help='ICD lambda weight', type=float, default=0.0, required=False)
parser.add_argument('--l_dc', help='distance correlation', type=float, default=0.0, required=False)

parser.add_argument('--nbdt_mode', help='train with nbdt', default='fixed', required=False)
parser.add_argument('--tloss_start', help='epoch to start training with tree supervision loss', type=int, default=20)
parser.add_argument('--ft_epochs', help='epoch to start perform fine tuning', type=int, default=50)
parser.add_argument('--pt_epochs', help='first n epochs to start perform pretraining. Default no pretraining', type=int, default=0)
parser.add_argument('--omega_t', help='max omega', type=float, default=1.0)

parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--lrt', help='learning rate for trained params', type=float, default=1e-4)

parser.add_argument('--wd', help='weight decay in Adam or AdamW', type=float, default=0.0)
parser.add_argument('--bs', help='batch size', type=int, default=72)
parser.add_argument('--hmac', help='mac hieght and width', type=int, default=256)
parser.add_argument('--hmic', help='mic height and width', type=int, default=320)
parser.add_argument('--n_emb', help='prototype embedding dimensions', type=int, default=512)

parser.add_argument('--mode', help='modality to use', default='MAC')
parser.add_argument('--amp', default='False', required=False)
parser.add_argument('--device', default='cuda:0', required=False)
parser.add_argument('--test', default='False', required=False)


args = parser.parse_args()
print(args)

# Device to train model on
DEVICE = args.device if torch.cuda.is_available() else 'cpu'

#DEVICE = 'cpu'
torch.cuda.device(DEVICE)

# To use auotmatic mixed precision
scaler = torch.cuda.amp.GradScaler() if args.amp=='True' else None

DATE = datetime.now()
YEAR = str(DATE.year)
MONTH = str(DATE.month)
DAY = str(DATE.day)
TODAY = DAY + '-' + MONTH + '-' + YEAR

torch.cuda.empty_cache()
print(f'Cache cleared. Training on GPU: {torch.cuda.current_device()}')
# Molemap folders for Data pipeline
MOLEMAPS = ['molemap_25k_ibukun','Molemap_Images_2020-02-11','Molemap_Images_2020-02-11_d2','Molemap_Images_2020-02-11_d3','Molemap_Images_2020-02-11_d4']

nw = 13 # K80 max: 12, T4 max: 6, A40 max: 13, T4 Heavy: 8

batch_size = args.bs
ROOT = f'../data/{MOLEMAPS[-1]}'

print(torch.version.cuda)

if args.model=='T':
    MODEL_FOLDER = 'Transfuser'
    
elif args.model=='LT':
    MODEL_FOLDER = 'LateFusion'
    
elif args.model=='SF':
    MODEL_FOLDER = 'ScoreFusion'
    
elif args.model=='SINGLE':
    MODEL_FOLDER = 'Resnet34'
else:
    MODEL_FOLDER = ''
    
if args.nbdt=='True':
    PROTO_OR_NBDT = 'proto-nbdt'
else:
    PROTO_OR_NBDT = 'proto'
    
SCRATCH = f'../../../scratch/oy30/nmkou3/models'

NAME = args.name

SAVE_DIR = os.path.join(SCRATCH, PROTO_OR_NBDT, MODEL_FOLDER, NAME)

try:
    os.mkdir(SAVE_DIR)
    print(f'SAVE_DIR: {SAVE_DIR} created.')

except OSError as error:
    print(f'SAVE_DIR: {SAVE_DIR} already exist.')
    
# ## Data
CLASSES = sorted(list(pd.read_csv('dataset/train.csv').label.value_counts().keys()))

RATIO = 1.0

H = 256 # 256rect with 160 is okay. 320rect with bs80 is okay. Base is 256sqr with bs256. 
W = int(H*RATIO)
W = W if W%2==0 else W+1
H0 = H+20

img_size_tag = f'-{args.hmac}-{args.hmic}-rect-' if RATIO>1.0 else f'-{args.hmac}-{args.hmic}-sqr-'

print(f'Input image resolution: {H} by {W} with {img_size_tag}')

l2_stats = {'train':{'mac':[ [0.6792, 0.5768, 0.5310], [0.1346, 0.1311, 0.1357] ], 
                     'mic':[ [0.7478, 0.6091, 0.5826], [0.0942, 0.0948, 0.0997]] },
            'val': {'mac':[ [0.6794, 0.5771, 0.5317], [0.1341, 0.1308, 0.1354] ], 
                    'mic':[ [0.7476, 0.6097, 0.5832], [0.0937, 0.0946, 0.0997] ]}, 
            'test': {'mac': [ [0.6795, 0.5773, 0.5317], [0.1342, 0.1309, 0.1356] ], 
                    'mic':[ [0.7478, 0.6093, 0.5826], [0.0944, 0.0952, 0.1001] ]}, 
            'eval':{'mac':[ [0.67945, 0.5772, 0.5317], [0.13415, 0.13085, 0.1355] ], 
             'mic':[[0.7477, 0.6095, 0.5829], [0.09405, 0.0949, 0.0999]]}, 
           }

t_stat = l2_stats['train']
v_stat = l2_stats['val']
te_stat = l2_stats['test']
eval_stat = l2_stats['eval']
    
  
tf_mac = transforms.Compose([
        transforms.Resize(args.hmac),
        transforms.CenterCrop((args.hmac, args.hmac)),        
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*t_stat['mac']),
        transforms.RandomApply([MyRotationTransform_(angles=[90,-90,180])],p=0.5),
        transforms.RandomApply([MyZoomTransform_(zoom=[0.99,0.95,0.9,0.85,0.80])],p=0.5),
])

tf_mic = transforms.Compose([
        transforms.Resize(args.hmic),
        transforms.CenterCrop((args.hmic, args.hmic)), 
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*t_stat['mic']),
        transforms.RandomApply([MyRotationTransform_(angles=[90,-90,180])],p=0.5),
        transforms.RandomApply([MyZoomTransform_(zoom=[0.99,0.95,0.9,0.85,0.80])],p=0.5),
])

tf  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(H),
        transforms.CenterCrop((H, W)),
])

tf_v_mac = transforms.Compose([
        transforms.Resize(args.hmac),
        transforms.CenterCrop((args.hmac, args.hmac)),
        transforms.ToTensor(),
        transforms.Normalize(*v_stat['mac'])
])

tf_v_mic = transforms.Compose([
        transforms.Resize(args.hmic),
        transforms.CenterCrop((args.hmic, args.hmic)),
        transforms.ToTensor(),
        transforms.Normalize(*v_stat['mic'])
])

train_tf = None 
val_tf = None 

train_ds = ClassDataset_CSV(ROOT, 'dataset/train.csv', CLASSES, transforms=[tf_mac, tf_mic] )
val_ds = ClassDataset_CSV(ROOT, 'dataset/val.csv', CLASSES, transforms=[tf_v_mac, tf_v_mic] )
test_ds = ClassDataset_CSV(ROOT, 'dataset/test.csv', CLASSES, transforms=[tf_v_mac, tf_v_mic] )

samples_ds = ClassDataset_CSV(ROOT,'dataset/samples.csv', CLASSES, transforms=[tf, tf] )


samples_dl = DataLoader(samples_ds,batch_size=batch_size,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)

train_dl = samples_dl if args.test=='True' else DataLoader(train_ds,batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
val_dl = samples_dl if args.test=='True' else DataLoader(val_ds,batch_size=batch_size,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)
test_dl = samples_dl if args.test=='True' else DataLoader(test_ds,batch_size=batch_size,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)

ds_tag = '-split_70_10_20-'

print(f'Train Dataset Size: {len(train_ds)}')
print(f'Val Dataset Size: {len(val_ds)}')
print(f'Test Dataset Size: {len(test_ds)}')
print(f'Samples Dataset Size: {len(samples_ds)}')

print(f'Train DataLoader Created with batch: {batch_size} Size: {len(train_dl)}')
print(f'Val DataLoader Created with batch: {batch_size} Size: {len(val_dl)}')
print(f'Test DataLoader Created with batch: {batch_size} Size: {len(test_dl)}')
print(f'Samples DataLoader Created with batch: {batch_size} Size: {len(samples_dl)}')
len(CLASSES)
D = torch.tensor(get_cost_matrix(CLASSES)).to(DEVICE)

# Choose model
T, LT, SF, SINGLE = False, False, False, False

if args.model=='T':
    T = True
elif args.model=='LT':
    LT = True
elif args.model=='SF':
    SF = True
elif args.model=='SINGLE':
    SINGLE = True

MODE = args.mode

# Choose how to train
SCRATCH = True

PRETRAINED = False # Training only classifier 

EPOCHS_0 = 0 # Starting epoch

PT_EPOCHS = args.pt_epochs # Epochs to only train classifier. LateFusion overfit easily after 4 epochs

TLOSS_START = args.tloss_start # Epoch to start training with tree supervision loss

FT_EPOCHS = args.ft_epochs # epochs to train end to end

EPOCHS = PT_EPOCHS + FT_EPOCHS # Total number of epochs trained.

patience = 100 # Patience
p = patience # patience countdown timer

# Hyper Params
lr = args.lr #0.0001 # untrained params. Transformers, classifiers, prototypes
lrt = args.lrt #0.0001 # Trained params/ CNN

# Proto Reg param
n_emb = args.n_emb # defualt 512
l_metric = args.l_metric #1.0
lm = args.lm #0.2 # ICD weights
l_dc = args.l_dc # 0.05 # alpha in paper. Page 

# NBDT hyper param
beta = 1 # Decaying constant for original loss_xe aka NN weight

#omega = 1 if args.nbdt=='True' else 0 # Growing Tree loss constant aka tree weight
omega = 0
omega_t = args.omega_t #1 # Max omega tree

if args.nbdt=='True' and args.nbdt_mode=='grow':
    omega = 0 # starts omega at 0, growing omegav every epoch
    betav = beta/(FT_EPOCHS - TLOSS_START) # decay increment for original loss_xe
    omegav = omega_t/(FT_EPOCHS - TLOSS_START) # growth increment for tree sup loss
else:
    betav, omegav = 0, 0
    
proto_2 = False # Proto_2 rebuilds tree every step with updated prototypes. e.g after optimizer.step()
    
best_val_loss = float('inf') 
best_val_acc = 0.55 if SINGLE else 0.62
best_val_nbdt_acc = 0.55 if SINGLE else 0.62

clustering_gif = [] # holds prototypes during training to make gif

if SCRATCH==True:
    
    if T==True:
        model = Transfuser(n_classes=n_emb,n_layers=[8,8,8,8], n_head=[4,4,4,4], fexpansion=4, 
                                    emb_pdrop=0, attn_pdrop=0, mlp_pdrop=0, pretrained=True, 
                                    cnns=['resnet34','resnet34'], ratio=RATIO, conv1d=False, fusion='cat').to(DEVICE)
        
    elif LT==True:
        model = LateFusion(num_class=n_emb).to(DEVICE)
        
    elif SF==True:
        model = ScoreFusion(num_class=n_emb).to(DEVICE)
        
    elif SINGLE==True:
        model = models.resnet34(pretrained=True).to(DEVICE)
        model.fc = nn.Linear(model.fc.in_features, n_emb).to(DEVICE)
    
    Prototypes = nn.Parameter(torch.rand((len(CLASSES), n_emb), device=torch.device(DEVICE))).requires_grad_(True) 

    tree = Hierarchy_prototype(Prototypes.data.clone().detach(), device=DEVICE).to(DEVICE)
        
    model_nbdt = PROTO_NBDT(model, tree, Prototypes, single=SINGLE, MODE=MODE).to(DEVICE)
   
    distortion_loss = DistortionLoss(D, DEVICE).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    params = [
              {'params': [p for n,p in model_nbdt.model.named_parameters() if 'encoder' not in n and 'Prototypes' not in n and p.requires_grad]},            
              {'params': [p for n,p in model_nbdt.model.named_parameters() if 'MAC' in n and p.requires_grad], 'lr':2*lrt},
              {'params': [p for n,p in model_nbdt.model.named_parameters() if 'MIC' in n and p.requires_grad], 'lr':lrt},
              {'params': model_nbdt.Prototypes, 'lr':lr}
             ]
    optimizer = optim.Adam(params, lr=lr) # default
    
    #schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1, verbose=True)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                      milestones=[12, 16],
                                                      gamma=0.05, verbose=False) # No schedular
    
    iterations = 0
    
    print('Pretrained model Loaded. Training pretrained model as NBDT.')

distortion_loss = DistortionLoss(D, DEVICE).to(DEVICE)

model_tag = model.__class__.__name__

# Freezing CNN BACKBONE
if PT_EPOCHS>0 and model.__class__.__name__ in ['Transfuser', 'ScoreFusion', 'LateFusion']:
    for param in model_nbdt.model.MIC_encoder.parameters():
        param.requires_grad = False
    for param in model_nbdt.model.MAC_encoder.parameters():
        param.requires_grad = False
    print('CNN backbone frozen')

    
DATE = datetime.now()
TODAY = f'{str(DATE.day)}-{str(DATE.month)}-22-{str(DATE.hour)}:{str(DATE.minute)}:{str(DATE.second)}'
        
RUN_NAME =f'{model_tag}-lr{lr}-lrt{lrt}-lm{l_metric}-lambda{lm}-ov{omegav:.2f}-{TODAY}' 

PROJECT_NAME = "Proto-NBDT"
run = wandb.init(
           project=PROJECT_NAME, 
           name = RUN_NAME,
           config={
               "batch_size":         batch_size,
               "transformer_layers": model.n_layers if model.__class__.__name__=='Transfuser' else None,
               "lr":                 lr,
               "n_epochs":           EPOCHS,
               "l_metric":           l_metric,
               "Hmac": args.hmac,
               "Hmic": args.hmic,
               'dataset': ds_tag,
               'img_size': img_size_tag,
               'beta': beta,
               'omega': omega,
               'lm': lm, # ICD loss weight
               'l_dc': l_dc, #alpha for correlation distance loss weight
               'omega_t': omega_t,
               'omegav': omegav,
               'optim':optimizer.__class__.__name__,
               'augmentation': 'AllTransforms',
               'NAME': NAME,
               'SAVE_DIR': SAVE_DIR,
               'model': model.__class__.__name__,
           })

SAVE_RUN_DIR = os.path.join(SAVE_DIR, RUN_NAME)

if os.path.exists(SAVE_RUN_DIR):
    print('Already exists')
    pass
else: 
    os.mkdir(SAVE_RUN_DIR)
    print(f'{SAVE_RUN_DIR} created')


print('Training Begins...')
    
for epoch in range(EPOCHS_0, EPOCHS):

    if epoch == PT_EPOCHS and PRETRAINED==True:
        for param in model_nbdt.model.parameters():
            param.requires_grad = True

    
    train_metrics = train_proto_nbdt(model_nbdt,
                               DEVICE,
                               train_dl,
                               optimizer,
                               criterion,
                                     
                               beta, 
                               omega,
                               l_metric,
                               lm,
                               l_dc,
                                     
                               epoch,
                               iterations,
                               train_tf,
                               CLASSES,
                               SAVE_RUN_DIR,
                               D,
                               distortion_loss,
                               clustering_gif,
                               single=SINGLE,
                               mode=MODE,
                               proto_2=proto_2,
                               scaler=scaler
                               )

    val_metrics = eval_proto_nbdt(model_nbdt,
                            DEVICE,
                            criterion,
                            val_dl, 
                            epoch,
                            val_tf,
                            CLASSES,
                            SAVE_RUN_DIR,
                            D,
                            distortion_loss,
                            clustering_gif,
                            single=SINGLE,
                            mode=MODE,
                            )
    
    if epoch==TLOSS_START:
        # Create new optimizer with learning rate reset
        params = [
              {'params': [p for n,p in model_nbdt.model.named_parameters() if 'encoder' not in n and 'Prototypes' not in n and p.requires_grad]},            
              {'params': [p for n,p in model_nbdt.model.named_parameters() if 'MAC' in n and p.requires_grad], 'lr':2*lrt},
              {'params': [p for n,p in model_nbdt.model.named_parameters() if 'MIC' in n and p.requires_grad], 'lr':lrt},
              {'params': model_nbdt.Prototypes, 'lr':lr}
             ]
        optimizer = optim.Adam(params, lr=lr) # default

        schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                      milestones=[12, 16],
                                                      gamma=0.05, verbose=True)
        if args.nbdt_mode=='fixed':
            omega = 1
            
    if epoch>=TLOSS_START:
        beta  = max(0, beta-betav)
        omega = min(omega_t, omega+omegav)
        if epoch<(EPOCHS-1): # Update the tree Hierarchy for all epochs except last epoch
            model_nbdt.tree = Hierarchy_prototype(model_nbdt.Prototypes.data.clone().detach(), device=DEVICE).to(DEVICE)
    print('VALIDATION DONE')
    iterations = train_metrics['iterations']
    
    schedular.step()
    
    metrics = {
        'train_metrics':train_metrics,
        'val_metrics':val_metrics,
        'best_val_loss':best_val_loss,
        'best_val_acc':best_val_acc,
    }

    check_point = {
        'model':model_nbdt,
        'epoch':epoch,
        'EPOCHS': EPOCHS,
        'optimizer':optimizer,
        'criterion': criterion,
        'initial_lr':lr,
        'batch_size':batch_size,
        'l_metric':l_metric,
        'trained_epochs':epoch,
        'next_epoch':epoch+1,
        'metrics':metrics,
        'omega': omega,
        'omegav': omegav,
        'omega_t': omega_t,
        'beta': beta,
        'betav': betav,
        'epochs': EPOCHS,
        'run_next_step': run.step + 1,
        'next_iterations': iterations + 1,
        }

    torch.save(check_point, os.path.join(SAVE_RUN_DIR,f'latest.pt'))  
    
    print(f'Latest Model CHECK POINTED!')
    if val_metrics['nbdt_acc'] > 0.63 or val_metrics['fc_acc'] > best_val_acc:
        best_val_nbdt_acc = val_metrics['nbdt_acc']
        
        best_val_acc = val_metrics['fc_acc']
        
        best_val_loss = val_metrics['loss']
        
        test_metrics = test_proto_nbdt(
                        model_nbdt,
                        DEVICE,
                        criterion,
                        test_dl, 
                        epoch,
                        val_tf,
                        CLASSES,
                        SAVE_DIR,
                        D,
                        distortion_loss,
                        single=SINGLE,
                        mode=MODE,
                        )
        
 
        torch.save(check_point, os.path.join(SAVE_RUN_DIR, f'best-cp.pt'))   
        
        print(f'BEST Model Checkpointed!')
        print(f'{epoch}/{EPOCHS} VAL NBDT acc:{val_metrics["nbdt_acc"]*100:.2f}%, loss:{val_metrics["loss"]}')
        print(f'{epoch}/{EPOCHS} TEST NBDT acc:{val_metrics["nbdt_acc"]*100:.2f}%, loss:{val_metrics["loss"]}')
    