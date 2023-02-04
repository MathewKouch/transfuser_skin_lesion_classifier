#!/usr/bin/env python
# coding: utf-8

#------------------------------- Importing Dependencies
import os # directory
from PIL import ImageFile,Image # reading image
ImageFile.LOAD_TRUNCATED_IMAGES = True # Avoiding truncation OSError 
from datetime import datetime
import torch
import torch.nn as nn # For NN modules
import torch.nn.functional as F # For activations and utilies
import torch.optim as optim
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

import numpy as np
import pandas as pd

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from scipy.special import softmax
import math

import random
from tqdm import tqdm # For progress bar

from my_utilities2 import train, evaluate, test, set_seed, get_cost_matrix
from my_dataset2 import ClassDataset_CSV
from models_baseline import ScoreFusion, LateFusion, Transfuser, TransfuserVIT
from augmentations import AllTransforms, MyRotationTransform_, MyZoomTransform_

from collections import OrderedDict
import pickle
from IPython import display
import wandb
import argparse
wandb.login()
set_seed(42)

#------------------------------- Args
parser = argparse.ArgumentParser(description='baseline argparse')
parser.add_argument('--model', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--mode', default='MAC', required=False)
parser.add_argument('--device', default='cuda:0', required=False)
parser.add_argument('--amp', default='True', required=False)
parser.add_argument('--hmac', type=int, default=256, required=False)
parser.add_argument('--hmic', type=int, default=512, required=False)
parser.add_argument('--lr', type=float, default=0.0001, required=False)
parser.add_argument('--lrt', type=float, default=0.0001, required=False)
parser.add_argument('--vit', default='vit_tiny_patch16_224', required=False)

args = parser.parse_args()
    
#------------------------------- Configurations
molemap = pd.read_csv('dataset/molemap.csv')
CLASSES = sorted(molemap.label.value_counts().keys())

# ## Constance and Global variables

# Device to train model on
DEVICE = args.device if torch.cuda.is_available() else 'cpu'

# To use auotmatic mixed precision
scaler = torch.cuda.amp.GradScaler() if args.amp=='True' else None
    
print(f'device {DEVICE}')
if torch.cuda.is_available():torch.cuda.set_device(DEVICE)

#------------------------------- Model Configurations

NAME = args.name

T, T_VIT, LT, SF, SINGLE = False, False, False, False, False

if args.model=='T':
    MODEL = 'Transfuser'
    T = True
elif args.model=='T_VIT':
    MODEL = 'TransfuserVIT'
    T_VIT = True
elif args.model=='LT':
    MODEL = 'LateFusion'
    LT = True
elif args.model=='SF':
    MODEL = 'ScoreFusion'
    SF = True
elif args.model=='SINGLE':
    MODEL = 'Resnet34'
    SINGLE = True

#------------------------------- Model and Data Configurations

SAVE_DIR = f'../../../scratch/oy30/nmkou3/models/baseline/{MODEL}/{NAME}'

try:
    os.mkdir(SAVE_DIR)
    print(f'SAVE_DIR: {SAVE_DIR} created.')

except OSError as error:
    print(f'SAVE_DIR: {SAVE_DIR} already exist.')
    
torch.cuda.empty_cache()
print(f'Cache cleared. Training on GPU: {torch.cuda.current_device()}')
# Molemap folders for Data pipeline
MOLEMAPS = ['molemap_25k_ibukun','Molemap_Images_2020-02-11','Molemap_Images_2020-02-11_d2','Molemap_Images_2020-02-11_d3','Molemap_Images_2020-02-11_d4']
batch_size = 72
nw = 13 # K80 max: 12, T4 max: 6, A40 max: 13, T4 Heavy: 8
ROOT = f'../data/{MOLEMAPS[-1]}'
print(torch.version.cuda)

#------------------------------- Image Augmentation 

RATIO_AVGPOOL = 1
RATIO = 1
H = args.hmac # 256rect with 160 is okay. 320rect with bs80 is okay. Base is 256sqr with bs256. 
W = int(H*RATIO)
W = W if W%2==0 else W+1

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

ds_tag = '-split-70-10-20-'


print(ds_tag)

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

#------------------------------- Data Configurations

molemap_df = pd.read_csv('dataset/molemap.csv')
CLASSES = sorted(molemap_df.label.value_counts().keys())

train_ds = ClassDataset_CSV(ROOT, 'dataset/train.csv', CLASSES, transforms=[tf_mac, tf_mic] )
val_ds = ClassDataset_CSV(ROOT, 'dataset/val.csv', CLASSES, transforms=[tf_v_mac, tf_v_mic] )
test_ds = ClassDataset_CSV(ROOT, 'dataset/test.csv', CLASSES, transforms=[tf_v_mac, tf_v_mic] )

sample_ds = ClassDataset_CSV(ROOT, 'dataset/samples.csv', CLASSES, transforms=[tf, tf] )

D = torch.tensor(get_cost_matrix(CLASSES)) # Cost Distance matrix to calculate severity of mistakes

train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)
test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)

print(f'Train Dataset Size: {len(train_ds)}')
print(f'Val Dataset Size: {len(val_ds)}')
print(f'Test Dataset Size: {len(test_ds)}')

print(f'Train DataLoader Created with batch: {batch_size} Size: {len(train_dl)}')
print(f'Val DataLoader Created with batch: {batch_size} Size: {len(val_dl)}')
print(f'Test DataLoader Created with batch: {batch_size} Size: {len(test_dl)}')

#------------------------------- Model Training Configurations

MODE = args.mode

SCRATCH = True

REWIND = False

PRETRAINED = False
    
# Set Hyper param
lr = args.lr # for untrained transformers
lrt = args.lrt # for pretrained params e.g resnet

l_metric = 1

# Conditions for checkpointing
best_val_loss = float('inf') 
best_val_acc = 0.62 if SINGLE==False else 0.55
best_val_epoch = 0

# Training time
patience = 100
p = patience # patience countdown timer

PT_EPOCHS = 3 if PRETRAINED else 0
FT_EPOCHS = 30 # set 100 for ReduceLROnPlateau
EPOCHS = PT_EPOCHS + FT_EPOCHS 
EPOCHS_START = 0

DECAY = False
BN_FREEZE = False
weight_decay = 0.01 if DECAY else 0
dec = f'-d{weight_decay}-' if DECAY else ''
bn = '-BN-' if BN_FREEZE else ''

#------------------------------- Model Creation

if SCRATCH==True:

    if T==True:

        model = Transfuser(n_classes=len(CLASSES), n_layers=[8,8,8,8], n_head=[4,4,4,4], fexpansion=4, 
                            emb_pdrop=0.1, attn_pdrop=0.1, mlp_pdrop=0.1, pretrained=True, 
                            cnns=['resnet34','resnet34'], ratio=RATIO_AVGPOOL, conv1d=False, fusion='cat').to(DEVICE)
        
        model_tag = f'XL{model.n_layers[0]}H{model.n_heads[0]}-{model.cnns[0][-2:]}{model.cnns[1][-2:]}'

    if T_VIT==True:
        model = TransfuserVIT(n_classes=len(CLASSES), vit=args.vit, n_layers=12, pretrained=True, cnns=['resnet34','resnet34'], 
            ratio=1.0, fusion='cat', classifier_pdrop=0.1).to(DEVICE)

        model_tag = f'T_VIT{model.n_layers}{args.vit}-{model.cnns[0][-2:]}{model.cnns[1][-2:]}'

    if LT==True:
        model = LateFusion(cnns=['resnet34','resnet34'], num_class=len(CLASSES)).to(DEVICE) 

        model_tag = f"LT{model.cnns[0][-2:]}{model.cnns[1][-2:]}"
        
    if SF==True:
        model = ScoreFusion(cnns=['resnet34','resnet34'], num_class=len(CLASSES)).to(DEVICE) 
        
        model_tag = f"SF{model.cnns[0][-2:]}{model.cnns[1][-2:]}"
        
    if SINGLE==True:
        resnet_models = {'resnet18':models.resnet18(pretrained=True),
                         'resnet34':models.resnet18(pretrained=True),
                         'resnet50':models.resnet18(pretrained=True)}
        
        choice = 'resnet34'
        model = resnet_models[choice].to(DEVICE) # Resnet18 with MIC images
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES)).to(DEVICE) #lr = 0.001 # For Resnet18
        
        model_tag = f"{choice}-{args.mode}"

        if PRETRAINED==True and SINGLE==True:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        
        del resnet_models

    if PRETRAINED == True:
        for param in model.MAC_encoder.parameters():
            param.requires_grad = False
        for param in model.MIC_encoder.parameters():
            param.requires_grad = False
        print('Pretraining TRUE. Model CNN frozen')

    # Optimizer and lr scheduler
    if SINGLE==False:                      
        param_dicts = [{"params": [p for n, p in model.named_parameters() if "MAC" in n and p.requires_grad], 'lr': 2*lrt},
                       {"params": [p for n, p in model.named_parameters() if "MIC" in n and p.requires_grad], 'lr': lrt},
                       {"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad], "lr": lr}
                      ]
    else:
        param_dicts = [{'params':[p for n,p in model.named_parameters() if 'fc' in n and p.requires_grad]},
             {'params': [p for n,p in model.named_parameters() if 'fc' not in n and p.requires_grad], 'lr':lrt},
            ] 
                             
    optimizer = torch.optim.Adam(param_dicts, lr=lr)

    if T==True or LT==True or SF==True or SINGLE==True:                         
        schedular1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14,20], gamma=1, verbose=True)
        schedular2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12,18], gamma=0.05, verbose=True)

    if T_VIT==True: 
        schedular1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14,20], gamma=1, verbose=True)
        schedular2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12,24], gamma=0.1, verbose=True)
        
    print(f'Training {model_tag} model from SCRATCH')
    total_params = sum(params.numel() for params in model.parameters() if params.requires_grad)
    print(f'Total trainable param {total_params}')

criterion = nn.CrossEntropyLoss().to(DEVICE)
criterion_eval = nn.CrossEntropyLoss().to(DEVICE)

#------------------------------- Weight and Biase Project Run Configurations

DATE = datetime.now()
TODAY = '{}-{}-22-{}:{}:{}'.format(
          str(DATE.day),
          str(DATE.month),
          str(DATE.hour),
          str(DATE.minute),
          str(DATE.second)
        )

pt_tag = f'-P{PT_EPOCHS}F{FT_EPOCHS}-' if SCRATCH==True else ''
RUN_NAME =f'{model_tag}--lr{lr}-lrt{lrt}-bs{batch_size}-{TODAY}'

wandb.init(project="FT-Baselines",
          name = RUN_NAME,
          settings=wandb.Settings(start_method="fork"),
          config={
              "batch_size":         batch_size,
              "MAC_cnn": model.cnns[0] if SINGLE!=True else choice,
              "MIC_cnn": model.cnns[1] if SINGLE!=True else choice,
              "transformer_layers": model.n_layers if T else 'NA',
              "transformer_heads": model.n_heads if T else 'NA',
              "lr":                 lr,
              "n_epochs":           EPOCHS,
              "l_metric":           l_metric,
              'optim':optimizer.__class__.__name__,
              'loss': criterion.__class__.__name__,
              'ds': ds_tag,
              'height':H,
              'width':W,
              'ratio':RATIO,
              'SAVE_DIR': SAVE_DIR,
              'RUN_NAME': RUN_NAME,
              'img_size_tag': img_size_tag,
              'NAME': NAME,
              'params': total_params,
          })

#------------------------------- Training and Validation

print(f'------{RUN_NAME}------------')
print('##################### BASELINE ####################')
print(f'{model_tag} Fine-Tuning Begins from {EPOCHS_START}')

for epoch in range(EPOCHS_START, EPOCHS):

    if (epoch == PT_EPOCHS) and (PRETRAINED == True):
        for param in model.parameters():
            param.requires_grad = True
    
    train_metrics = train(model,
                          DEVICE,
                          train_dl,
                          optimizer,
                          criterion,
                          epoch,
                          D,
                          CLASSES,
                          SAVE_DIR,
                          train_tf,
                          single=SINGLE,
                          MODE=MODE,
                          focal_loss=False,
                          scaler=scaler,
                         )

    val_metrics = evaluate(model, 
                           DEVICE,
                           val_dl,
                           criterion_eval,
                           epoch,
                           D,
                           CLASSES,
                           SAVE_DIR,
                           val_tf,
                           single=SINGLE,
                           MODE=MODE
                           )

    schedular1.step()  
    schedular2.step() 
    
    metrics = {
        'train_metrics':train_metrics,
        'val_metrics':val_metrics,
        'best_val_loss':best_val_loss,
        'best_val_acc':best_val_acc,
    }


    check_point = {
        'model':model,
        'optimizer':optimizer,
        'initial_lr':lr,
        'batch_size':batch_size,
        'l_metric':l_metric,
        'trained_epochs':epoch,
        'metrics':metrics,
        'EPOCHS': EPOCHS,
        'model_tag': model_tag,
        'val_acc': val_metrics['acc'],
        'val_loss': val_metrics['loss']
        }
    
    # Create new folder for this wandb run only when first epoch is successful
    SAVE_RUN_DIR = os.path.join(SAVE_DIR, TODAY)
    
    if os.path.exists(SAVE_RUN_DIR):
        pass
    else: 
        os.mkdir(SAVE_RUN_DIR)
        
    torch.save(check_point, os.path.join(SAVE_RUN_DIR,f'latest-cp.pt'))    
    print(f'Latest Model CHECK POINTED!')

    if val_metrics['acc'] > best_val_acc:
        p = patience # resets patience
        
        best_val_acc = val_metrics['acc']
        best_val_epoch = epoch
        
        torch.save(check_point, os.path.join(SAVE_RUN_DIR,f'best-cp.pt'))   
        torch.save(check_point, os.path.join(SAVE_RUN_DIR,f'best-epoch-{epoch}-cp.pt'))    

        print(f'New best at {epoch}/{EPOCHS} CHECK POINTED!') 
        
        print('Now testing')
        test_metrics = test(model, DEVICE, test_dl, criterion, epoch, D, CLASSES, SAVE_DIR, 
                            tf=val_tf, single=SINGLE, MODE=MODE, status='test')
            

    else:
        p -= 1 # reduce patience for Early Stopping
        
    if p <= 0: # No more patience! Exit training now!
        print(f'EARLY STOPPING! Patience: {patience}, and best val loss: {best_val_loss}')
        break 
    if optimizer.param_groups[0]['lr']<1e-8:
        print(f'STOPPING! Learning rate below 1e-8')
        break
        
    print(f'Patience: {p}, Best val_loss: {best_val_loss:.4f} , val_acc: {best_val_acc:.4f} at {best_val_epoch}')



