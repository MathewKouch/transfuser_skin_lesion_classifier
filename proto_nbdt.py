# Contains prototype and decision tree classifier (proto-nbdt) wrappers, 
# and proto-nbdt model training and validaiton loops

import os # directory
from PIL import ImageFile,Image # reading image
ImageFile.LOAD_TRUNCATED_IMAGES = True # Avoiding truncation OSError 
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
from scipy.special import softmax
import math
from tqdm import tqdm # For progress bar
import wandb
from sklearn.metrics import classification_report, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns # for confusion matrix visualisation
from pylab import savefig # to create and save sns figure 
from my_utilities2 import wandb_log, log_confusion_matrix
from prototypical import get_dist, Distortion, ScaleFreeDistortion, proto_vis, ICD, Loss_DC # to calculate the Euclidean distances between prototypes

from torch.distributions import Categorical
import imageio

import random
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(RANDOM_SEED)
random.seed(RANDOM_SEED)

class PROTO_NBDT(nn.Module):
    '''
    Joins and wrap the backbone model with the induced hierarchy tree.
    '''
    def __init__(self, model, tree, P, single=False, MODE='MAC'):
        super(PROTO_NBDT, self).__init__()
                
        self.MODE = MODE
        
        self.single = single
                        
        self.model = model # Any model that outputs a representation
        
        self.Prototypes = P
        
        self.tree = tree
        
        #assert self.model.classifier[-1].out_features==self.Prototypes.shape[-1], 'Model output dim must match prototype dim'
        
    def update_tree(self):
        
        self.tree.load_weights(self.Prototypes.clone().detach())

    def forward(self, MAC, MIC):
        if self.single==True:
            
            img = MAC if self.MODE=='MAC' else MIC
            emb = self.model(img) 
   
        else:
            emb = self.model(MAC, MIC) 
            
        # distance between representation to every Prototypes                
        dist = -torch.norm(emb[:, None, :] - self.Prototypes[None, :, :], dim=-1)             

        prob, _ = self.tree.get_leaf_prob(emb) # tree leaf path prob, all filled_prob

        return dist, prob, emb
        
    
def train_proto_nbdt(model, DEVICE, data_loader, optimizer, criterion, beta, omega, l_metric, lm, l_dc,
               epoch, iterations, tf, CLASSES, SAVE_DIR, D, distortion_loss, clustering_gif, 
                     single=False, mode='MAC', proto_2=False, gif_iterations=500, scaler=None):
    '''
    Train the prototypical network.
    - model (nn.Module) is the pretrained FX backbone NN.
    - DEVICE (str) cuda/cpu
    - data_loader
    - optimizer
    - epoch (current epoch)
    - tf (list of transforms) one per MAC and MIC
    - proto_2 (bool) is NBDT tree that updates weight every step by loading new prototypes without re-initit
    '''
    
    count = 0
    fc_correct = 0
    nbdt_correct = 0
    avg_loss = 0
    avg_loss_xe = 0
    avg_loss_tree = 0
    avg_dist_loss = 0
    
    # For wandb metrics logging
    all_prob = []
    all_pred = []
    
    all_prob_tree = []
    all_pred_tree = []
    
    all_labels = []
    
    model.train()
    
    distortion = Distortion(D)
    icd = ICD() # intra class distance
    sf_distortion = ScaleFreeDistortion(D)
    corr_loss = Loss_DC()
    
    avg_icd_loss = 0
    avg_sf_distortion = 0
    avg_loss_dc = 0
    
    loader = tqdm(data_loader)
    for batch_idx, (labels,_,_,MAC_img,MIC_img) in enumerate(loader):
        labels = labels.to(DEVICE) 
        if tf is not None and len(tf)>1:
            MAC_img = tf[0](MAC_img)
            MIC_img = tf[1](MIC_img)
        elif tf is not None and len(tf)==1:
            MAC_img, MIC_img = tf[0](MAC_img, MIC_img, labels)
        
        optimizer.zero_grad()
        if scaler is None:
            dists, prob, features = model(MAC_img.to(DEVICE), MIC_img.to(DEVICE)) # output of fx, tree, feature vector sample

            loss_xe = criterion(dists, labels) # aka Loss _data
            dist_loss = distortion_loss(model.Prototypes) # distortion_loss loss of prototypes with respect to class hierarchy
            loss_tree = F.cross_entropy(prob, labels) # tree supervision loss

            icd_loss = icd(features, model.Prototypes[labels])

            loss_dc = corr_loss(model.Prototypes, D)

            loss = beta*loss_xe + l_metric*dist_loss + omega*loss_tree + lm*icd_loss +l_dc*loss_dc
            
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                dists, prob, features = model(MAC_img.to(DEVICE), MIC_img.to(DEVICE)) # output of fx, tree, feature vector sample

                loss_xe = criterion(dists, labels) # aka Loss _data
                dist_loss = distortion_loss(model.Prototypes) # distortion_loss loss of prototypes with respect to class hierarchy
                loss_tree = F.cross_entropy(prob, labels) # tree supervision loss

                icd_loss = icd(features, model.Prototypes[labels])

                loss_dc = corr_loss(model.Prototypes, D)

                loss = beta*loss_xe + l_metric*dist_loss + omega*loss_tree + lm*icd_loss +l_dc*loss_dc
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
        # Update hierarchy node weights every update step
        if proto_2==True:
            model.update_tree()
        
        with torch.no_grad():
            
            # Class predictions from dist between feature and prototypes
            count += len(labels)
            pred = torch.argmax(dists.detach(),1).detach()
            pred_path = torch.argmax(prob,1).detach()
            
            fc_correct += (pred==labels).sum()
            nbdt_correct += (pred_path==labels).sum()
            
            avg_loss += loss.item()
            avg_icd_loss += icd_loss.item()
            avg_loss_xe += loss_xe.item()
            avg_loss_tree += loss_tree.item()
            avg_dist_loss += dist_loss.item()
            avg_loss_dc += loss_dc.item()

            distortion_ = distortion(model.Prototypes.clone().detach()) # distortion   
            sf_distortion_ = sf_distortion(model.Prototypes.clone().detach()) # SF distortion
            avg_sf_distortion += sf_distortion_
            avg_sf_distortion /= (batch_idx+1)
            
            fc_acc = fc_correct/count
            nbdt_acc = nbdt_correct/count
            
            all_pred += pred.cpu()
            all_labels += labels.cpu()
            all_prob += F.softmax(dists.cpu(), -1)

            all_pred_tree += pred_path
            all_prob_tree += prob
            
            if batch_idx%10==0 or ((batch_idx+1)==len(loader)):
                
                proto_dis = get_dist(model.Prototypes.clone().detach())
                proto_dis = proto_dis/np.max(proto_dis) # normalise the distance
                proto_image = wandb.Image(proto_dis)
                
                batches = batch_idx + 1
                wandb.log({"train_running_loss":avg_loss/batches, 
                           "train_running_icd_loss": avg_icd_loss/batches,
                           "train_running_loss_dc": avg_loss_dc/batches,
                          "train_running_acc":fc_acc,
                          "train_run_nbdt_acc":nbdt_acc,
                          "train_run_fc_loss":avg_loss_xe/batches,
                          "train_run_nbdt_loss":avg_loss_tree/batches,
                          "train_dist_loss": avg_dist_loss/batches,
                          "iterations": iterations,
                          "proto_dist": proto_image,
                          'train_distortion': distortion_,
                           'train_scale_free_distortion': sf_distortion_
                         })
                loader.set_description(f'TRAIN | {epoch} | FC_ACC: {fc_acc*100:.2f}%, FC_LOSS: {avg_loss_xe/(batch_idx+1):.5f} | NBDT_ACC: {nbdt_acc*100:.2f}%, Tree Loss: {avg_loss_tree/(batch_idx+1):.5f} | LOSS: {avg_loss/(batch_idx+1):.5f}')
                
            if iterations%gif_iterations==0:
                B = features.shape[0]
                fx = features.clone().detach().cpu().reshape(B,-1)
                proto_vis(model.Prototypes.clone().detach().cpu(), 
                          fx, 
                          labels.cpu(), CLASSES, clustering_gif, epoch,
                          iterations, SAVE_DIR, dim=2, n_iter=10000, 
                          alpha=0.9, size=16, save_fig=True, make_gif=True, status='train')    
                
            iterations += 1
            
    with torch.no_grad(): 
        metrics = {
                   'acc':fc_correct/count,
                   'nbdt_acc':nbdt_acc,
                   'nbdt_loss':avg_loss_tree/len(data_loader),
                   'loss':avg_loss/len(data_loader),
                   'iterations': iterations,
                  }
        wandb.define_metric('train_acc', summary='max')
        wandb.define_metric('train_loss', summary='min')
        wandb.define_metric('train_AHC', summary='min')
    
        wandb.log({"train_loss":avg_loss/batches, 
                  "train_fc_acc":fc_correct/count,
                  "train_acc":fc_correct/count,
                  "train_fc_loss": avg_loss_xe/batches,
                  "train_NBDT_acc":nbdt_correct/count,
                  "train_NBDT_loss":avg_loss_tree/batches,
                  'train_avg_distortion': avg_sf_distortion,
                  "epoch":epoch,
                  "epochs":epoch,
                  'omega':omega,
                  'beta':beta,
                  "lr":optimizer.param_groups[0]['lr'],
                 })
        
        wandb_log(
            all_prob,
            all_pred,
            all_labels,
            [nbdt_acc,avg_loss_tree/len(loader)],
            epoch, 
            CLASSES,
            SAVE_DIR,
            D,
            status='train',
            nbdt_outputs=[all_prob_tree, all_pred_tree],
        )
       
        return metrics
    
def eval_proto_nbdt(model, DEVICE, criterion,data_loader, epoch, tf, 
              CLASSES, SAVE_DIR, D, distortion_loss, clustering_gif, single=False, mode='MAC'):
    '''
    model is the trained FX NN
    '''
    
    count = 0
    fc_correct = 0
    nbdt_correct = 0
    avg_loss = 0
    avg_loss_xe = 0
    avg_loss_tree = 0
    avg_dist_loss = 0
    avg_dc_loss = 0
    # For wandb metrics logging
    all_prob = []
    all_pred = []
    
    all_prob_tree = []
    all_pred_tree = []
    
    all_labels = []
    
    features = []

    distortion = Distortion(D)
    corr_loss = Loss_DC()
    icd = ICD()
    
    model.eval()
    with torch.no_grad():
        
        distortion_ = distortion(model.Prototypes.clone().detach()).item() # distortion
    
        loader = tqdm(data_loader)
        for batch_idx, (labels,_,_,MAC_img,MIC_img) in enumerate(loader):
            labels = labels.to(DEVICE)
 
            if tf is not None and len(tf)>1:
                MAC_img = tf[0](MAC_img).to(DEVICE)
                MIC_img = tf[1](MIC_img).to(DEVICE)
            elif tf is not None and len(tf)==1:
                MAC_img, MIC_img = tf[0](MAC_img, MIC_img, labels)
            
            dists, prob, fx = model(MAC_img.to(DEVICE), MIC_img.to(DEVICE)) # output of fx, tree, feature vector sample
                        
            loss_xe = criterion(dists,labels) # aka Loss _data
            dist_loss = distortion_loss(model.Prototypes)
            loss_dc = corr_loss(model.Prototypes, D)
            icd_loss = icd(fx, model.Prototypes[labels])
            
            loss_tree = F.cross_entropy(prob, labels)
            
            loss = loss_xe #beta*loss_xe + l_metric*dist_loss + omega*loss_tree + l_dc*loss_dc + lm*icd_loss
            
            with torch.no_grad():
                if len(fx.shape)==3:
                    B, r, c = fx.shape
                    features.append( fx.detach().cpu().view(B*r, c) )
                elif len(fx.shape)==2:
                    features.append(fx.detach().cpu())
                                
                pred = torch.argmax(dists.detach(),1).detach()
                pred_path = torch.argmax(prob,1).detach()
                
                nbdt_correct += (pred_path==labels).sum()
                fc_correct += (pred==labels).sum()
                
                avg_loss += loss.item()
                avg_loss_xe += loss_xe.item()
                avg_loss_tree += loss_tree.item()
                avg_dist_loss += dist_loss.item()
                avg_dc_loss += loss_dc.item()
                
                count += len(labels)
                
                fc_acc = fc_correct/count
                nbdt_acc = nbdt_correct/count

                #LOSS_meter.add(loss.item())
                all_pred += pred.cpu()
                all_labels += labels.cpu()
                all_prob += F.softmax(dists.cpu(), -1)

                all_pred_tree += pred_path
                all_prob_tree += prob.detach()
                # loader.set_description(f'EVAL|{epoch}| FC_ACC: {100*fc_acc:.2f}% | NBDT_ACC: {nbdt_acc*100:.2f}% | Loss: {loss.item():0.5f}')
                loader.set_description(f'EVAL | {epoch} | FC_ACC: {fc_acc*100:.2f}%, FC_LOSS: {avg_loss_xe/(batch_idx+1):.5f} | NBDT_ACC: {nbdt_acc*100:.2f}%, Tree Loss: {avg_loss_tree/(batch_idx+1):.5f} | LOSS: {avg_loss/(batch_idx+1):.5f}')
            
    with torch.no_grad():
        
        last_batch = features[-1]
        B = last_batch.shape[0]
        lb = last_batch.squeeze().view(B,-1)
        features = torch.stack(features[:-1]) # still 3D
        if len(features.shape)==3:
            a,b,c = features.shape
            features = features.view(a*b, c)
            
        features = torch.cat([features, lb], dim=0)
        
        all_labels_vis = torch.stack(all_labels, dim=0).cpu()
        
        proto_vis(model.Prototypes.clone().detach().cpu(), 
                  features.cpu(), 
                  all_labels_vis,
                  CLASSES, clustering_gif, 
                  epoch,0, SAVE_DIR, dim=2, n_iter=10000, alpha=0.9, 
                  size=200, save_fig=True, make_gif=True, status='val')    
        
        wandb.define_metric('val_acc', summary='max')
        wandb.define_metric('val_loss', summary='min')
        wandb.define_metric('val_AHC', summary='min')
        
        wandb.define_metric('val_fc_acc', summary='max')
        wandb.define_metric('val_fc_loss', summary='min')
        
        wandb.define_metric('val_NBDT_acc', summary='max')
        wandb.define_metric('val_NBDT_loss', summary='min')
        wandb.define_metric('val_nbdt_AHC', summary='min')
        
        wandb.define_metric('val_nbdt_acc', summary='max')
        wandb.define_metric('val_nbdt_loss', summary='min')
        
        wandb.log({"val_loss":avg_loss/len(loader), 
                  "val_fc_acc":fc_correct/count,
                  "val_fc_loss": avg_loss_xe/len(loader),
                  "val_NBDT_acc":nbdt_correct/count,
                  "val_NBDT_loss":avg_loss_tree/len(loader),
                  "val dist_loss": avg_dist_loss/len(loader),
                   "val_loss_dc": avg_dc_loss/len(loader),
                  "epoch":epoch,
                  "epochs":epoch,
                  "val_distortion": distortion_
                 })
        
        metrics = {
                   'acc':fc_correct/count,
                   'fc_acc':fc_acc,
                   'nbdt_acc':nbdt_acc,
                   'nbdt_loss':avg_loss_tree/(avg_loss_tree/len(loader)),
                   'loss':avg_loss/len(data_loader),
                  }
        
        # #print(f'EVAL | {epoch}:{batch_idx+1}/{len(data_loader)}| FC_ACC: {100*fc_acc:.2f}% | NBDT_ACC: {nbdt_acc*100:.2f}% |')
        wandb_log(
            all_prob,
            all_pred,
            all_labels,
            [nbdt_acc, avg_loss_tree/len(loader)],
            epoch, 
            CLASSES, 
            SAVE_DIR,
            D,
            status='val',
            nbdt_outputs=[all_prob_tree, all_pred_tree],
        )
        return metrics
    
    
def test_proto_nbdt(model, DEVICE, criterion,data_loader, epoch, tf, \
                    CLASSES, SAVE_DIR, D,distortion_loss, single=False, mode='MAC'):
    '''
    model is the trained FX NN
    '''
    
    count = 0
    fc_correct = 0
    nbdt_correct = 0
    avg_loss = 0
    avg_loss_xe = 0
    avg_loss_tree = 0
    avg_dist_loss = 0
    
    # For wandb metrics logging
    all_prob = []
    all_pred = []
    
    all_prob_tree = []
    all_pred_tree = []
    
    all_labels = []
    
    distortion = Distortion(D)
    corr_loss = Loss_DC()
    icd = ICD()
    
    model.eval()
    with torch.no_grad():

        distortion_ = distortion(model.Prototypes.clone().detach()) # distortion
        
        loader = tqdm(data_loader)
        for batch_idx, (labels,_,_,MAC_img,MIC_img) in enumerate(loader):
            labels = labels.to(DEVICE)
 
            if tf is not None and len(tf)>1:
                MAC_img = tf[0](MAC_img).to(DEVICE)
                MIC_img = tf[1](MIC_img).to(DEVICE)
            elif tf is not None and len(tf)==1:
                MAC_img, MIC_img = tf[0](MAC_img, MIC_img, labels)
            
            dists, prob, fx = model(MAC_img.to(DEVICE), MIC_img.to(DEVICE)) # output of fx, tree, feature vector sample
                                 
            loss_xe = criterion(dists,labels) # aka Loss _data
            dist_loss = distortion_loss(model.tree.leaves)
            loss_dc = corr_loss(model.Prototypes, D)
            icd_loss = icd(fx, model.Prototypes[labels])
            loss_tree = F.cross_entropy(prob, labels)
            
            loss = loss_xe #beta*loss_xe + l_metric*dist_loss + omega*loss_tree + l_dc*loss_dc + lm*icd_loss
            
            with torch.no_grad():
                
                pred = torch.argmax(dists,1).detach()
                pred_path = torch.argmax(prob,1).detach()
                
                nbdt_correct += (pred_path==labels).sum()
                fc_correct += (pred==labels).sum()
                
                avg_loss += loss.item()
                avg_loss_xe += loss_xe.item()
                avg_loss_tree += loss_tree.item()
                avg_dist_loss += dist_loss.item()
                
                count += len(labels)
                
                fc_acc = fc_correct/count
                nbdt_acc = nbdt_correct/count

                #LOSS_meter.add(loss.item())
                all_pred += pred.cpu()
                all_labels += labels.cpu()
                all_prob += F.softmax(dists.cpu(), -1)

                all_pred_tree += pred_path
                all_prob_tree += prob.detach()
                # loader.set_description(f'EVAL|{epoch}| FC_ACC: {100*fc_acc:.2f}% | NBDT_ACC: {nbdt_acc*100:.2f}% | Loss: {loss.item():0.5f}')
                loader.set_description(f'TESTING | {epoch} | FC_ACC: {fc_acc*100:.2f}%, FC_LOSS: {avg_loss_xe/(batch_idx+1):.5f} | NBDT_ACC: {nbdt_acc*100:.2f}%, Tree Loss: {avg_loss_tree/(batch_idx+1):.5f} | LOSS: {avg_loss/(batch_idx+1):.5f}')
            
    with torch.no_grad():
        
        wandb.define_metric('test_acc', summary='max')
        wandb.define_metric('test_loss', summary='min')
        wandb.define_metric('test_AHC', summary='min')
        
        wandb.define_metric('test_fc_acc', summary='max')
        wandb.define_metric('test_fc_loss', summary='min')
        
        wandb.define_metric('test_NBDT_acc', summary='max')
        wandb.define_metric('test_NBDT_loss', summary='min')
        wandb.define_metric('test_nbdt_AHC', summary='min')
        
        wandb.define_metric('test_nbdt_acc', summary='max')
        wandb.define_metric('test_nbdt_loss', summary='min')
        
        wandb.log({"test_loss":avg_loss/len(loader), 
                  "test_fc_acc":fc_correct/count,
                  "test_fc_loss": avg_loss_xe/len(loader),
                  "test_NBDT_acc":nbdt_correct/count,
                  "test_NBDT_loss":avg_loss_tree/len(loader),
                  "test_dist_loss": avg_dist_loss/len(loader),
                  "epoch":epoch,
                  "epochs":epoch,
                  'test_distortion': distortion_
                 })
        
        metrics = {
                   'acc':fc_correct/count,
                   'fc_acc':fc_acc,
                   'nbdt_acc':nbdt_acc,
                   'nbdt_loss':avg_loss_tree/(avg_loss_tree/len(loader)),
                   'loss':avg_loss/len(data_loader),
                  }
        
        #print(f'EVAL | {epoch}:{batch_idx+1}/{len(data_loader)}| FC_ACC: {100*fc_acc:.2f}% | NBDT_ACC: {nbdt_acc*100:.2f}% |')
        wandb_log(
            all_prob,
            all_pred,
            all_labels,
            [nbdt_acc, avg_loss_tree/len(loader)],
            epoch, 
            CLASSES, 
            SAVE_DIR,
            D,
            status='test',
            nbdt_outputs=[all_prob_tree, all_pred_tree],
        )
        return metrics

