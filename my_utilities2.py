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
import matplotlib.pyplot as plt
from scipy.special import softmax
import math
from tqdm import tqdm # For progress bar
import wandb
from torchnet.meter import ClassErrorMeter, AverageValueMeter
import seaborn as sns # for confusion matrix visualisation
from pylab import savefig # to create and save sns figure 
import random
import itertools

import umap

from prototypical import proto_vis

# For metrics
from sklearn.metrics import classification_report, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import top_k_accuracy_score

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

def set_seed(RANDOM_SEED=42):
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print('random seed ', RANDOM_SEED, ' set.')

def get_accuracy(score,label):
    correct = (torch.argmax(score,1)==label).sum()
    return correct/len(label)

def train(model, DEVICE, data_loader, optimizer, criterion, epoch, D, 
          CLASSES, SAVE_DIR, tf=None, single=False, UR_CLASSES=None, MODE='MAC', focal_loss=False, scaler=None):
    
    losses = []
    accuracies = []
    correct = 0
    count = 0
    
    # TO calc ROC-AUC
    all_prob = [] # contains all output class prob
    all_pred = [] # contains all ouput class predictions
    all_labels = [] # contains all true labels
    
    wandb.log({
        'epochs': epoch,
        'lr':optimizer.param_groups[0]['lr'],
    })
    
    model.train()
    loader = tqdm(data_loader)
    for batch_idx, (labels,_,_,MAC_img,MIC_img) in enumerate(loader):
        
        labels = labels.to(DEVICE)
            
        if tf!=None and len(tf)==1:
            MAC_img, MIC_img = tf[0](MAC_img, MIC_img, labels)
        elif tf!=None and len(tf)==2:
            MAC_img, MIC_img = tf[0](MAC_img), tf[1](MIC_img)
            
            
        if single:
            if MODE=='MAC':
                img = MAC_img.to(DEVICE)
            elif MODE=='MIC':
                img = MIC_img.to(DEVICE)
                             
        else:

            MAC_img = MAC_img.to(DEVICE)

            MIC_img = MIC_img.to(DEVICE)
        
        optimizer.zero_grad()
        
        if scaler is None:
            
            if single:
                score = model(img).to(DEVICE)
            else:
                score = model(MAC_img, MIC_img).to(DEVICE)
                
            loss = criterion(score,labels) # Move loss to Device for back prop
        
            if focal_loss==True:
                gamma = 2
                pt = torch.exp(-loss)
                loss = (((1.0-pt)**gamma)*loss).mean() # focal Loss

            loss.backward()
            optimizer.step()
            
        else:
            with torch.cuda.amp.autocast():
                
                if single:
                    score = model(img).to(DEVICE)
                else:
                    score = model(MAC_img, MIC_img).to(DEVICE)
    
                loss = criterion(score,labels) # Move loss to Device for back prop

                if focal_loss==True:
                    gamma = 2
                    pt = torch.exp(-loss)
                    loss = (((1.0-pt)**gamma)*loss).mean() # focal Loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
        
        with torch.no_grad():
            
            pred = torch.argmax(score,1).detach()
            
            all_pred += pred.cpu()
            all_labels += labels.cpu()
            all_prob += F.softmax(score.detach().cpu().to(dtype=torch.float32), dim=-1)
            
            count += len(labels)
            correct += (pred==labels).sum()
            acc = correct/count
            
            accuracies.append(acc)
            losses.append(loss.item())			
			
            if batch_idx%10==0:
                AVG_loss = torch.tensor(losses).mean()
                loader.set_description(f'|EPOCH {epoch} | ACC: {acc*100:.2f}% | LOSS: {AVG_loss:.5f} |')
                wandb.log({'train_running_acc': acc,
                           'train_running_loss': AVG_loss,
                           'epochs': epoch
                          })
    
    with torch.no_grad():
        AVG_loss = torch.tensor(losses).mean()
        AVG_acc = correct/count

        wandb.define_metric('train_acc', summary='max')
        wandb.define_metric('train_loss', summary='min')
        wandb.define_metric('train_AHC', summary='min')
        
        
        wandb_log(all_prob, all_pred, all_labels, [AVG_acc, AVG_loss], epoch, 
                CLASSES, SAVE_DIR=SAVE_DIR, D=D, status='train', nbdt_outputs=False)

 
        
        metrics = {'loss': AVG_loss,
                'acc': AVG_acc,
                'losses': losses,
                'accuracies': accuracies,
                'all_pred': all_pred,
                'all_labels': all_labels,
                'iterations':(epoch+1)*(batch_idx+1),
                }
        
    return metrics

def evaluate(model, DEVICE,data_loader, criterion, epoch, D, 
             CLASSES, SAVE_DIR, tf=None, single=False, MODE='MAC', status='val', clustering_gif=None, MODEL=False):
    
    losses = []
    accuracies = []
    count = 0
    correct = 0

    # TO calc ROC-AUC
    all_prob = [] # contains all output class prob
    all_pred = [] # contains all ouput class predictions
    all_labels = [] # contains all true labels
    
    features = [] # holds feature vectors
    
    model.eval()
    with torch.no_grad():
        loader = tqdm(data_loader)
        for batch_idx, (labels,_,_,MAC_img,MIC_img) in enumerate(loader):
            
            labels = labels.to(DEVICE)
            
            if tf!=None and len(tf)==1:
                MAC_img, MIC_img = tf[0](MAC_img, MIC_img, labels)
            elif tf!=None and len(tf)==2:
                MAC_img, MIC_img = tf[0](MAC_img), tf[1](MIC_img)
            
               
            if single:
                if MODE=='MAC':
                    img = MAC_img.to(DEVICE)
                elif MODE=='MIC':
                    img = MIC_img.to(DEVICE)

                score = model(img).to(DEVICE)
                
            else:
                
                MAC_img = MAC_img.to(DEVICE)
                
                MIC_img = MIC_img.to(DEVICE)

                score = model(MAC_img, MIC_img) # Move score to cpu as labels are also in cpu
            
            fx = score.detach()
            
            loss = criterion(score,labels)

            with torch.no_grad():

                pred = torch.argmax(score,1).detach()
                count += len(labels)
                correct += (pred==labels).sum()
                acc = correct/count
                
                all_pred += pred.cpu()
                all_labels += labels.cpu()
                all_prob += F.softmax(score.cpu(), -1)
                
                accuracies.append(acc)
                losses.append(loss.item())
                AVG_loss = torch.tensor(losses).mean()
                
                # Appending features
                if len(fx.shape)==3:
                    B, r, c = fx.shape
                    features.append( fx.detach().cpu().view(B*r, c) )
                elif len(fx.shape)==2:
                    features.append(fx.detach().cpu())
                    
                loader.set_description(f'|EPOCH {epoch}| ACC: {(correct/count)*100:.2f}% | LOSS: {AVG_loss:.5f} |')
    
    with torch.no_grad():
        
        AVG_loss = torch.tensor(losses).mean()
        AVG_acc = correct/count

        wandb.define_metric(f'{status}_acc', summary='max')
        wandb.define_metric(f'{status}_loss', summary='min')
        wandb.define_metric(f'{status}_AHC', summary='min')

        wandb_log(all_prob, all_pred, all_labels, [AVG_acc, AVG_loss], epoch, 
                CLASSES, SAVE_DIR=SAVE_DIR, D=D, status=status, nbdt_outputs=False)

        metrics = {'loss': AVG_loss,
                'acc': AVG_acc,
                'losses': losses,
                'accuracies': accuracies,
                'all_pred': all_pred,
                'all_labels': all_labels,
                }
        
    return metrics

def test(model, DEVICE,data_loader, criterion, epoch, D, 
             CLASSES, SAVE_DIR, tf=None, single=False, MODE='MAC', status='test'):
    
    losses = []
    accuracies = []
    count = 0
    correct = 0

    # TO calc ROC-AUC
    all_prob = [] # contains all output class prob
    all_pred = [] # contains all ouput class predictions
    all_labels = [] # contains all true labels
    
    model.eval()
    with torch.no_grad():
        loader = tqdm(data_loader)
        for batch_idx, (labels,_,_,MAC_img,MIC_img) in enumerate(loader):
            
            labels = labels.to(DEVICE)
            
            if tf!=None and len(tf)==1:
                MAC_img, MIC_img = tf[0](MAC_img, MIC_img, labels)
            elif tf!=None and len(tf)==2:
                MAC_img, MIC_img = tf[0](MAC_img), tf[1](MIC_img)
            
            
            if single:
                if MODE=='MAC':
                    img = MAC_img.to(DEVICE)
                elif MODE=='MIC':
                    img = MIC_img.to(DEVICE)

                score = model(img).to(DEVICE)
                
            else:
                
                MAC_img = MAC_img.to(DEVICE)
                
                MIC_img = MIC_img.to(DEVICE)

                score = model(MAC_img, MIC_img) # Move score to cpu as labels are also in cpu
            
            loss = criterion(score,labels)

            with torch.no_grad():

                pred = torch.argmax(score,1).detach()
                count += len(labels)
                correct += (pred==labels).sum()
                acc = correct/count
                
                all_pred += pred.cpu()
                all_labels += labels.cpu()
                all_prob += F.softmax(score.cpu(), -1)
                
                accuracies.append(acc)
                losses.append(loss.item())
                AVG_loss = torch.tensor(losses).mean()
                loader.set_description(f'| TESTING | ACC: {(correct/count)*100:.2f}% | LOSS: {AVG_loss:.5f} |')
    
    with torch.no_grad():

        AVG_loss = torch.tensor(losses).mean()
        AVG_acc = correct/count

        wandb.define_metric(f'{status}_acc', summary='max')
        wandb.define_metric(f'{status}_loss', summary='min')
        wandb.define_metric(f'{status}_AHC', summary='min')

        wandb_log(all_prob, all_pred, all_labels, [AVG_acc, AVG_loss], epoch, 
                CLASSES, SAVE_DIR=SAVE_DIR, D=D, status=status, nbdt_outputs=False)

        metrics = {'loss': AVG_loss,
                'acc': AVG_acc,
                'losses': losses,
                'accuracies': accuracies,
                'all_pred': all_pred,
                'all_labels': all_labels,
                }
        
    return metrics

   
def get_cost_matrix(CLASSES):
    ''' Generate class cost matrix from CLASSES as list with class name as: c1_c2_c3'''
    COST_MATRIX = np.zeros((len(CLASSES),len(CLASSES)))
    for row, a in enumerate(CLASSES):
        for col, b in enumerate(CLASSES):
            c = a.split('_')
            d = b.split('_')
            dist = 3
            for idx, e in enumerate(c):
                if e==d[idx]:
                    dist -= 1
                else:
                    break
            COST_MATRIX[row,col] = dist/3
            
    return COST_MATRIX


def get_AHC(D,pred,true, macro=False):
    ''' Clculates the Average Hierarchy Cost.
    Args: - D (2D matrix) of class distance aka cost matrix
          - pred (list or tensor of integers) of predicted class idx
          - true (list or tensor of integers) of true class idx
    '''
    if macro==False:
        N = len(true) # N samples
        dist = 0
        for idx, p in enumerate(pred):
            dist += D[p,true[idx]]
            AHC = dist/N
            AHC = AHC.item()
        return AHC # The AHC
    else:
        AHC_macro = {k:[] for k in set([t.item() for t in true ])}
        N = len(true) # N samples
        dist = 0
        for idx, p in enumerate(pred):
            dist += D[p,true[idx]]
            AHC = dist/N
            AHC = AHC.item()
            AHC_macro[true[idx]].append(AHC) # Calc and saves AHC for every prediction
        AHC_macro_avg = 0
        for k,v in AHC_macro.items():
            AHC_macro[k] = torch.tensor(AHC_macro[k]).mean() # Calc the mean AHC for every class
            AHC_macro_avg+=AHC_macro[k]
        return AHC_macro, AHC_macro_avg/len(AHC_macro) # The AHC



def wandb_log(all_prob, all_pred, all_labels, stats, epoch, 
              CLASSES, SAVE_DIR=None, D=None, status='train', nbdt_outputs=False):
    '''
    Create performance metrics and log them to wandb
    args:
    all_prob: a list of tensor prob output from NN backbone, tensors size (N x k) N is batch size, k is number of classes
              of the softmax of the output of the model, per batch
    
    nbdt_ooutputs: a list containing: (default is False)
              1) a list of tensor prob output from tree of NBDT, size (N x k) N is batch size, k is number of classes
              of the softmax of the output of the model, per batch. 
              2) a list of tensors of pred output of nbdt
              
    all_pred: a list of tensors of the predictions per bacth, (Nx1)
    all_labels: a list of tensors of all the labels per batch, (Nx1)
    epoch: (int) tracks the epoch, required at first epoch to count number of samples
    CLASSES: list of strings of class names
    status: train or eval or test of this run default is train
    
    '''
    all_probs_ = torch.stack(all_prob).cpu()
    all_pred = torch.stack(all_pred).cpu()
    all_labels = torch.stack(all_labels).cpu()
    
    # Appending class index label in their name for easier referencing
    # CLASSES_ = list()
    # for idx, cls_ in enumerate(CLASSES):
    #     CLASSES_.append(f'[{idx}] {cls_}')
        
    # Define Top2 and Top3 Accuracy


    CLASSES_ = CLASSES
    with torch.no_grad():

        ACC = stats[0]
        LOSS = stats[1]
        
        roc_auc_ovr_weighted = roc_auc_score(all_labels.cpu(), all_probs_.cpu(), multi_class='ovr', average='weighted')
        roc_auc_ovr_macro = roc_auc_score(all_labels.cpu(), all_probs_.cpu(), multi_class='ovr', average='macro')
        roc_auc_ovr_micro = roc_auc_score(all_labels.cpu(), all_probs_.cpu(), multi_class='ovr', average='micro')

        # Precision, Recall, F1-score
        prf1_report_weighted = precision_recall_fscore_support(all_labels.cpu(), all_pred.cpu(), average='weighted', zero_division=0)
        prf1_report_macro = precision_recall_fscore_support(all_labels.cpu(), all_pred.cpu(), average='macro', zero_division=0)

        prf1_report_micro = precision_recall_fscore_support(all_labels.cpu(), all_pred.cpu(), average='micro', zero_division=0)

            
        precision_weighted = prf1_report_weighted[0]
        recall_weighted = prf1_report_weighted[1]
        f1_weighted = prf1_report_weighted[2]

        precision_macro = prf1_report_macro[0]
        recall_macro= prf1_report_macro[1]
        f1_macro = prf1_report_macro[2]

        precision_micro = prf1_report_micro[0]
        recall_micro= prf1_report_micro[1]
        f1_micro = prf1_report_micro[2]
        
        # Class precision, recall, f1 REPORT
        report = classification_report(all_labels.cpu(),
                                       all_pred.cpu(),
                                       labels=np.arange(len(CLASSES)), 
                                       target_names=CLASSES_, 
                                       sample_weight=None, 
                                       digits=2, 
                                       output_dict=True, 
                                       zero_division=0)
        if epoch==0:
            class_count = []
            for c in CLASSES_:
                class_count.append(report[c]['support'])
            data = [[label, val] for (label, val) in zip(CLASSES_, class_count)]
            table = wandb.Table(data=data, columns = ["classes", "count"])
            wandb.log({f"{status}_class_count": wandb.plot.bar(table, "classes", "count", title=f"{status.title()} Class Count")})
    
        class_count = []
        for c in CLASSES_:
            class_count.append(report[c]['f1-score'])
        data = [[label, val] for (label, val) in zip(CLASSES_, class_count)]
        table = wandb.Table(data=data, columns = ["classes", "f1-score"])
        wandb.log({f"{status} F score": wandb.plot.bar(table, "classes", "f1-score", title=f"{status.title()} F score"),
                   'epoch': epoch,
                   'epochs': epoch,
                  })

        class_count = []
        for c in CLASSES_:
            class_count.append(report[c]['precision'])
        data = [[label, val] for (label, val) in zip(CLASSES_, class_count)]
        table = wandb.Table(data=data, columns = ["classes", "precision"])
        wandb.log({f"{status} Precision": wandb.plot.bar(table, "classes", "precision", title=f"{status.title()} Precision"),
                  'epoch': epoch,
                  'epochs': epoch,
                  })

        class_count = []
        for c in CLASSES_:
            class_count.append(report[c]['recall'])
        data = [[label, val] for (label, val) in zip(CLASSES_, class_count)]
        table = wandb.Table(data=data, columns = ["classes", "recall"])
        wandb.log({f"{status} Recall": wandb.plot.bar(table, "classes", "recall", title=f"{status.title()} Recall"),
                  'epoch': epoch,
                  'epochs': epoch,
                  })

        # Average AHC of all prediction
        AHC = get_AHC(D, all_pred, all_labels)
        
        log_confusion_matrix(all_labels, all_pred, stats=stats, epoch=epoch, 
                             CLASSES=CLASSES, SAVE_DIR=SAVE_DIR, SIZE=(30,20), 
                             DPI=128, xrot=0, status=f'{status}', cmap='Reds')

        level0_acc, level1_acc = get_level_accuracy(all_pred, all_labels, D)

        wandb.log({
            'epochs': epoch,
            'epoch': epoch,
            f'{status}_roc_auc_ovr_weighted': roc_auc_ovr_weighted,
            f'{status}_roc_auc_ovr_macro': roc_auc_ovr_macro,
            f'{status}_roc_auc_ovr_micro': roc_auc_ovr_micro,
            
            f'{status}_precision_weighted': precision_weighted,
            f'{status}_recall_weighted': recall_weighted,
            f'{status}_f1_weighted' : f1_weighted,
            
            f'{status}_precision_macro': precision_macro,
            f'{status}_recall_macro': recall_macro,
            f'{status}_f1_macro' : f1_macro,
            
            f'{status}_precision_micro': precision_micro,
            f'{status}_recall_micro': recall_micro,
            f'{status}_f1_micro' : f1_micro,
            
            f"{status} top-1 accuracy": top_1_accuracy(all_labels, all_probs_, CLASSES_),
            f"{status} top-2 accuracy": top_2_accuracy(all_labels, all_probs_, CLASSES_),
            f"{status} top-3 accuracy": top_3_accuracy(all_labels, all_probs_, CLASSES_),
            f"{status} level0_acc": level0_acc,
            f"{status} level1_acc": level1_acc,

            f'{status}_AHC': AHC,
            f'{status}_acc': ACC,
            f'{status}_loss': LOSS,

            #f'{status}_AHC_macro': AHC_macro.item(),
        })
        
    # Log metrics for nbdt output    
    if nbdt_outputs:
        with torch.no_grad():
            all_probs_ = torch.stack(nbdt_outputs[0]).cpu()
            all_pred = torch.stack(nbdt_outputs[1]).cpu()
            
            #all_probs_ = torch.stack(all_probs).cpu()
            roc_auc_ovr_weighted = roc_auc_score(all_labels, all_probs_, multi_class='ovr', average='weighted')
            roc_auc_ovr_macro = roc_auc_score(all_labels, all_probs_, multi_class='ovr', average='macro')
            roc_auc_ovr_micro = roc_auc_score(all_labels, all_probs_, multi_class='ovr', average='micro')

            # Precision, Recall, F1-score
            prf1_report_weighted = precision_recall_fscore_support(all_labels, all_pred, average='weighted', zero_division=0)
            prf1_report_macro = precision_recall_fscore_support(all_labels, all_pred, average='macro', zero_division=0)
            prf1_report_micro = precision_recall_fscore_support(all_labels, all_pred, average='micro', zero_division=0)

            precision_weighted = prf1_report_weighted[0]
            recall_weighted = prf1_report_weighted[1]
            f1_weighted = prf1_report_weighted[2]

            precision_macro = prf1_report_macro[0]
            recall_macro= prf1_report_macro[1]
            f1_macro = prf1_report_macro[2]

            precision_micro = prf1_report_micro[0]
            recall_micro= prf1_report_micro[1]
            f1_micro = prf1_report_micro[2]
            
            # Class precision, recall, f1 REPORT
            report = classification_report(all_labels,
                                           all_pred,
                                           labels=np.arange(len(CLASSES_)), 
                                           target_names=CLASSES_, 
                                           sample_weight=None, 
                                           digits=2, 
                                           output_dict=True, 
                                           zero_division=0)
            
            class_count = []
            for c in CLASSES_:
                class_count.append(report[c]['f1-score'])
            data = [[label, val] for (label, val) in zip(CLASSES_, class_count)]
            table = wandb.Table(data=data, columns = ["classes", "f1-score"])
            wandb.log({f"{status} NBDT F score": wandb.plot.bar(table, "classes", "f1-score", title=f"{status.title()} NBDT F score"),
                       'epoch':epoch,
                       'epochs':epoch,
                      })

            class_count = []
            for c in CLASSES_:
                class_count.append(report[c]['precision'])
            data = [[label, val] for (label, val) in zip(CLASSES_, class_count)]
            table = wandb.Table(data=data, columns = ["classes", "precision"])
            wandb.log({f"{status} NBDT Precision": wandb.plot.bar(table, "classes", "precision", title=f"{status.title()} NBDT Precision"),
                       'epoch':epoch,
                       'epochs':epoch,
                      })

            class_count = []
            for c in CLASSES_:
                class_count.append(report[c]['recall'])
            data = [[label, val] for (label, val) in zip(CLASSES_, class_count)]
            table = wandb.Table(data=data, columns = ["classes", "recall"])
            wandb.log({f"{status} NBDT Recall": wandb.plot.bar(table, "classes", "recall", title=f"{status.title()} NBDT Recall"),
                      'epoch': epoch,
                      'epochs': epoch,
                      })

            AHC = get_AHC(D, all_pred, all_labels)
       
            log_confusion_matrix(all_labels, all_pred, stats=stats, epoch=epoch, 
                             CLASSES=CLASSES, SAVE_DIR=SAVE_DIR, SIZE=(30,20), 
                             DPI=128, xrot=0, status=f'{status} NBDT', cmap='Reds')    

            level0_acc, level1_acc = get_level_accuracy(all_pred, all_labels, D)

            wandb.log({
                'epochs': epoch,
                'epoch': epoch,
                f'{status}_nbdt_roc_auc_ovr_weighted': roc_auc_ovr_weighted,
                f'{status}_nbdt_roc_auc_ovr_macro': roc_auc_ovr_macro,
                f'{status}_nbdt_roc_auc_ovr_micro': roc_auc_ovr_micro,

                f'{status}_nbdt_precision_weighted': precision_weighted,
                f'{status}_nbdt_recall_weighted': recall_weighted,
                f'{status}_nbdt_f1_weighted' : f1_weighted,
                
                f'{status}_nbdt_precision_macro': precision_macro,
                f'{status}_ndbt_recall_macro': recall_macro,
                f'{status}_nbdt_f1_macro' : f1_macro,
                
                f'{status}_nbdt_precision_micro': precision_micro,
                f'{status}_ndbt_recall_micro': recall_micro,
                f'{status}_nbdt_f1_micro' : f1_micro,
                
                f"{status}_nbdt top-1 accuracy": top_1_accuracy(all_labels, all_probs_, CLASSES_),                
                f"{status}_nbdt top-2 accuracy": top_2_accuracy(all_labels, all_probs_, CLASSES_),
                f"{status}_nbdt top-3 accuracy": top_3_accuracy(all_labels, all_probs_, CLASSES_),
                f"{status} level0_nbdt_acc": level0_acc,
                f"{status} level1_nbdt_acc": level1_acc,

                f'{status}_nbdt_AHC': AHC,
                f'{status}_acc': ACC,
                f'{status}_loss': LOSS,
                #f'{status}_nbdt_AHC_macro': AHC_macro.item(),
            })
            
            
def get_level_accuracy(pred, real, D):
    ''' Calculates level0 (malignancy condition) and level1 (super class) accuracy of predictions
        D (tensor): 65 x 65, the cost distance tensor used to derive the accuracy matrix level0 and level1
        pred (tensor): N x 1, N =  number of predictions
        real (tensor): N x 1, N = number of real
        CLASSES (list): contains 65 skin lesion classes, string
    '''

    level0 = torch.zeros_like(D) # 2D tensor of 1s = correct or 0s = incorrect prediction for level0  
    level1 = torch.zeros_like(D) # 2D tensor of 1s = correct or 0s = incorrect prediction for level1 

    for row, col in itertools.product(range(D.shape[0]), range(D.shape[1])):
        level0[row][col] = 1 if D[row][col]<=2/3 else 0
 
    for row, col in itertools.product(range(D.shape[0]), range(D.shape[1])):
        level1[row][col] = 1 if D[row][col]<=1/3 else 0

    level0_acc = level0[pred, real].mean()
    level1_acc = level1[pred, real].mean()
    
    return level0_acc, level1_acc

def log_confusion_matrix(actual, pred, stats, epoch,CLASSES, SAVE_DIR, normalize='true',
                         SIZE=(30,20), DPI=128,xrot=0,status='train',cmap='Reds', save_fig=False):
    
    SIZE = (30,20) # Good ratio with dpi 64 to 128 gives under 300KB
    DPI = 80
    
    acc = stats[0] if stats!= None else ''
    loss = stats[1] if stats!=None else ''

    from matplotlib.pyplot import figure

    #### For plotting graphs/ confusion matrix
    ylabels = list()
    xlabels = list()

    for idx, cls_ in enumerate(CLASSES):
        ylabels.append(f'[{idx}] {cls_}')
        xlabels.append(f'{idx}')
        
    plt.figure(figsize=SIZE)
    
    conf = confusion_matrix(actual,pred, normalize=normalize)
    cfs = conf.copy()
    for r in range(len(conf[0])):
        for l in range(len(conf[1])):
            cfs[r,l] = np.round(conf[r,l], decimals=2) if conf[r,l]>1e-2 else 0

    ax = sns.heatmap(cfs, annot=True, cmap=cmap)

    ax.set_title(f'{status.title()} Confusion Matrix at {epoch}, acc: {acc*100:.2f}%, loss: {loss:.4f}');
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(xlabels)
    ax.yaxis.set_ticklabels(ylabels)
    ax.set_yticklabels(ax.get_yticklabels(),rotation = 0)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = xrot)


    figure = ax.get_figure()
    
    fn = f'{status} conf_matrix_{SIZE}_{DPI}_epoch_{epoch}.png'
    
    if save_fig==True:
        SAVE_DIR = os.path.join(SAVE_DIR, 'confusion_matrix')
    
        if os.path.exists(SAVE_DIR):
            pass
        else: 
            os.mkdir(SAVE_DIR)
            
        figure.savefig(os.path.join(SAVE_DIR, fn), dpi=DPI)
        
    plt.close(figure) # ensures closing all plot/figure windows
    
    wandb.log({
        'epoch':epoch,
        f'{status.title()} Confusion Matrix Image': wandb.Image(figure),
    })

def top_3_accuracy(labels, probs, CLASSES):
    return top_k_accuracy_score(y_true=labels, y_score=probs, k=3)#, labels=CLASSES)

def top_2_accuracy(labels, probs, CLASSES):
    return top_k_accuracy_score(y_true=labels, y_score=probs, k=2)#, labels=CLASSES)

def top_1_accuracy(labels, probs, CLASSES):
    return top_k_accuracy_score(y_true=labels, y_score=probs, k=1)#, labels=CLASSES)

def set_bn_momentum(model, momentum):
    for mn, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            print(f'Before {mn} momentum: {m.momentum}')
            m.momentum = momentum # set to 0.1 for large batch size, else 0.0001 for anything smaller, smaller is best
            print(f'After {mn} momentum: {m.momentum}')
    print('Model momentum successfully frozen')

    return model

def freeze_bn(model):
    for mn, m in model.named_modules():
        for pn, param in m.named_parameters():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.0001
                print(f'Before {mn} requires_grad? {param.requires_grad}')

                param.requires_grad=False
                print(f'After {mn} requires_grad? {param.requires_grad} \n')

    print('Model BatchNorm2d layers Frozen')

    return model

def freeze_input_layers(model, input_layers=None):
    assert input_layers is not None, 'You MUST input input_layers to FREEZE!'
    if input_layers is None:
        input_layers = [
            # Resnets
            'conv1', 'bn1', 'maxpool',

            # Densenet121
            'conv0', 'pool0'
            
            ]
    
    layers_frozen = [] # Records which layers were frozen

    if isinstance(model, (TransFuser3, TransFuser4, DenseFuser)):
        print('TransFuser detected')
        input_layers_MAC = [f'MAC_encoder.resnet.{l}' for l in input_layers]
        input_layers_MIC = [f'MIC_encoder.resnet.{l}' for l in input_layers]
        
        input_layers = input_layers_MAC + input_layers_MIC
        
        for mn, m in model.named_modules():
            for pn, param in m.named_parameters():
                if mn in input_layers:

                    print(f'Before {mn}.{pn} requires_grad? {param.requires_grad}')
                    param.requires_grad=False
                    print(f'After {mn}.{pn} requires_grad? {param.requires_grad} \n')
                    layers_frozen.append(f'{mn}.{pn}')

        print(f'TransFuser input layers {layers_frozen} frozen')

    
    else:
        for mn, m in model.named_modules():
            for pn, param in m.named_parameters():
                if mn in input_layers:

                    print(f'Before {mn}.{pn} requires_grad? {param.requires_grad}')
                    param.requires_grad=False
                    print(f'After {mn}.{pn} requires_grad? {param.requires_grad} \n')
                    layers_frozen.append(f'{mn}.{pn}')

        print(f'Input layers {layers_frozen} frozen')

    return model   

def fine_tuning(model, fc_layers=None):
    ''' Freezes all layers except the MLP head, defined by fc_layers'''
    assert fc_layers is not None, 'You must input which fc_layers to ALLOW for training!'
    if fc_layers is None:
        fc_layers = [
                    # Resnets
                    'fc', 
                    # Densenet, TransFusers
                    'classifier'
                    ]
    layers_frozen = [] # Records which layers were frozen
    for p in model.parameters():
        #Freezes all parameters first
        p.requires_grad = False
    
    # Then only unfreeze last fc layers
    for mn, m in model.named_modules():
        for pn, param in m.named_parameters():
            for frozen_layer in fc_layers:
                if pn.startswith(frozen_layer):

                    print(f'Before {mn}.{pn} requires_grad? {param.requires_grad}')
                    param.requires_grad = True
                    print(f'After {mn}.{pn} requires_grad? {param.requires_grad} \n')
                    layers_frozen.append(f'{mn}.{pn}')

    print(f'On {layers_frozen} are trained!')

    return model   

def pretraining(model, pt_layers=None):
    ''' Freezes layers in pt_layers for model pretraining 
        Args:
        model - nn.Module, the model where layers need to be freezed
        pt_layers - list of strings for the names of the layers to train
        Return:
        model with layers freezed.
    '''
    assert pt_layers is not None, 'You must input which layer to FREEZE for pretrianing!'

    if pt_layers is None:
        pt_layers = [
                    # Resnets
                    'MAC_encoder',
                    'MIC_encoder' 
                    # Densenet, TransFusers
                    
                    ]
    layers_frozen = [] # Records which layers were frozen

    for mn, m in model.named_modules():
        for pn, param in m.named_parameters():
            for ptl in pt_layers:
                if pn.startswith(ptl):
                    
                    #print(f'Before {mn}.{pn} requires_grad? {param.requires_grad}')
                    param.requires_grad=False
                    #print(f'After {mn}.{pn} requires_grad? {param.requires_grad} \n')
                    layers_frozen.append(f'{mn}.{pn}')

    print(f'Layers {layers_frozen} frozen')

    return model   

def get_model_size(model):
    '''
    Counts number of trainable params in model
    Arg:
    model - nn.Module
    Return:
    total_param - float, total parameters 
    trainable_param - float, total trainable parameters
    '''
    total_param = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Model has {total_param} total parameters, of which {trainable_params} are trainable.')
    return total_param, trainable_params


  

def decay_weights(DECAY=True, model, lr):
    ''' creates an optimizer with weight decay to specific parameters in model
        Args:
        model - nn.Module 
        lr - float, the learning rate
        Return:
        optimizer 
    '''
    # Applying weight decay
    if DECAY == True:
        decay = set()# List of parameters that will have weight decay
        no_decay = set() # List of parameters that wil NOT have weight decay
        all_params = set(model.parameters())
        module_with_decay = (torch.nn.Linear, torch.nn.Conv2d)
        module_with_no_decay = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)

        for mod_name, mod in model.named_modules():
            for param_name, param in mod.named_parameters():
                fpn = f'{mod_name}.{param_name}' if mod_name else param_name

                if param_name.endswith('bias'):
                    no_decay.add(fpn)
                elif param_name.endswith('weight') and isinstance(mod, module_with_decay):
                    decay.add(fpn)
                elif param_name.endswith('weight') and isinstance(mod, module_with_no_decay):
                    no_decay.add(fpn)

                if isinstance(mod, Transformer_New):
        # no_decay = all_params - decay
                    no_decay.add(mod_name + '.positional_emb')

        param_dict = {param_name: param for param_name, param in model.named_parameters()}
        optimizer_parameters = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))], 'weight_decay': weight_decay },
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))], 'weight_decay': 0.00},
        ]    

        optimizer = optim.Adam(optimizer_parameters, lr=lr)

    return optimizer

class MeanModel(nn.Module):
    def __init__(self):
        super(MeanModel, self).__init__()
        '''
        Calculates the mean of the dataset. Need to do this first
        '''

    def forward(self,MAC, MIC):
        ''' MAC and MIC are rgb tensor'''
        MAC = MAC.squeeze(0)
        MIC = MIC.squeeze(0)
        mac_sum = MAC.sum(axis=(1,2))
        mic_sum = MIC.sum(axis=(1,2))
        
        return mac_sum, mic_sum

class STDModel(nn.Module):
    def __init__(self, mac_mean, mic_mean, DEVICE):
        super(STDModel, self).__init__()
        '''
        mac_mean is the mean of the ds. Need to calculate the means first.
        returns the mac and mic std
        '''
        
        self.mac_mean = mac_mean.reshape((3,1,1)).to(DEVICE)
        self.mic_mean = mic_mean.reshape((3,1,1)).to(DEVICE)
        
    def forward(self,MAC, MIC):
        ''' MAC and MIC are rgb tensor'''
        MAC = MAC.squeeze(0)
        MIC = MIC.squeeze(0)
        macstd = (MAC**2 - (self.mac_mean**2)).sum(axis=(1,2))**0.5
        micstd = (MIC**2 - (self.mic_mean**2)).sum(axis=(1,2))**0.5
#         print(f'MAC shape is {MAC.shape}, mac_mean.shape is {self.mac_mean.shape}, macsq.shape is {macsq.shape}')

        return macstd, micstd

def get_molemap():
    MOLEMAP_NAME= 'Molemap_Images_2020-02-11_d4'
    MOLEMAP_DIR = f'../data/{MOLEMAP_NAME}'
    MOLEMAP_CLASSES = os.listdir(MOLEMAP_DIR)
    MOLEMAP_HIERARCHY = {}
    MOLEMAP_UNIQUE_CLASSES = {}
    for c in MOLEMAP_CLASSES:
        c1 = c.split('_')[0] # super class: benign or malignant - 2 classes
        c2 = c.split('_')[1] # subclass: has 4 subclasses in each c1 superclass
        c3 = c.split('_')[2] # name of class: has 65 in MoleMap
    #     print(c1,c2,c3)
        if c1 not in MOLEMAP_HIERARCHY:
            MOLEMAP_HIERARCHY[c1] = {}
        else:
            if c2 not in MOLEMAP_HIERARCHY[c1]:
                MOLEMAP_HIERARCHY[c1][c2] = {}
            else:
                if c3 not in MOLEMAP_HIERARCHY[c1][c2]:
                    MOLEMAP_HIERARCHY[c1][c2][c3]=1
        class_name = c1 + "_" + c2 + "_" + c3
        if class_name not in MOLEMAP_UNIQUE_CLASSES:
            MOLEMAP_UNIQUE_CLASSES[class_name] = 0

    for c in MOLEMAP_CLASSES:
        c1 = c.split('_')[0] # super class: benign or malignant - 2 classes
        c2 = c.split('_')[1] # subclass: has 4 subclasses in each c1 superclass
        c3 = c.split('_')[2] # name of class: has 65 in MoleMap
        class_name = c1 + "_" + c2 + "_" + c3
        MAC_size = len(os.listdir(os.path.join(MOLEMAP_DIR,c,'MAC'))) # Number of MAC images in this class folder
        MIC_size = len(os.listdir(os.path.join(MOLEMAP_DIR,c,'MIC.POL'))) # Number of MIC images in this class folder
        MOLEMAP_UNIQUE_CLASSES[class_name] += min(MAC_size, MIC_size)

    #     MOLEMAP_UNIQUE_CLASSES['malignant_keratinocytic*_actinic']=MOLEMAP_UNIQUE_CLASSES['malignant_keratinocytic_actinic']
    # del MOLEMAP_UNIQUE_CLASSES['malignant_keratinocytic_actinic']
    MOLEMAP_UNIQUE_CLASSES = OrderedDict(sorted(MOLEMAP_UNIQUE_CLASSES.items())) # classes and count in molemap
    CLASSES= list(sorted(MOLEMAP_UNIQUE_CLASSES.keys())) # Alphabetically sorted classes in molemap



    MOLEMAP_SORTED_VALUES = OrderedDict(sorted(MOLEMAP_UNIQUE_CLASSES.items(), key=lambda kv:(kv[1], kv[0])))
    CLASS_NAME_SORTED_VALUES = list(MOLEMAP_SORTED_VALUES.keys())
    CLASS_FREQ_SORTED_VALUES = list(MOLEMAP_SORTED_VALUES.values())
    
    return MOLEMAP_UNIQUE_CLASSES, CLASSES, MOLEMAP_SORTED_VALUES

def get_cost_matrix(CLASSES, dist=3):
    ''' Generate class cost matrix from CLASSES as list with class name as: c1_c2_c3'''
    COST_MATRIX = np.zeros((len(CLASSES),len(CLASSES)))
    for row, a in enumerate(CLASSES):
        for col, b in enumerate(CLASSES):
            c = a.split('_')
            d = b.split('_')
            dist0 = dist
            for idx, e in enumerate(c):
                if e==d[idx]:
                    dist0 -= 1
                else:
                    break
            if row==col:
                COST_MATRIX[row,col] = 0
            else:
                COST_MATRIX[row,col] = dist0/dist
            
    return COST_MATRIX

def get_class_count(l,CLASSES):
    ''' Output panda DF of class count in dataset
    Arg: - ds (subset of Dataset)
        - l (list) the list of class index lables in each dataset
        -CLASSES (list) list of the 65 classes alphabetically sorted
    return: - class_count (pd.df)
    '''
    class_count = {}
    for cl in CLASSES:
        class_count[cl]=0
        
    for c in l:
        cl = CLASSES[c]
        class_count[cl] += 1

    # Return alphabetically sorted class
    return OrderedDict(sorted(class_count.items(), key = lambda kv : (kv[0],kv[1])))
                
def plot_class_count(MOLEMAP_SORTED_VALUES, save=False, sort=True):
    
    if sort==True: # sort by value ascending
        classes = OrderedDict(sorted(MOLEMAP_SORTED_VALUES.items(), key=lambda kv: (kv[1],kv[0])))
    else: # sort by class name
        classes = OrderedDict(sorted(MOLEMAP_SORTED_VALUES.items(), key=lambda kv: (kv[0],kv[1])))
        
    classes = list(MOLEMAP_SORTED_VALUES.keys())
    count = list(MOLEMAP_SORTED_VALUES.values())

    y_pos = np.arange(len(classes))
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

    fig, ax = plt.subplots(figsize=(12,15))
    my_cmap = plt.get_cmap("summer")
    bars = ax.barh(y_pos, count, align='center', color=my_cmap(rescale(count)))
    ax.bar_label(bars)
    ax.invert_yaxis()  # labels read top-to-bottom

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Count')
    ax.set_ylabel('Class')
    if save==True:
        title_name = input('Name of plot? ')
    title_name = 'Class Plot'
    ax.set_title(f'{title_name}')
    
    if save==True:
        plot_name = input('Name of this class count? ')
        plt.savefig(f'{plot_name}.png')
        print(f'{plot_name}.png plot is saved!')


def plot_cost_matrix(CLASSES, save=False,size=(20,20)):
    cs = get_cost_matrix(CLASSES)
    fig, ax = plt.subplots(figsize=size)
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=90)
    # ax.xaxis.tick_top()
    # ax.xaxis.set_ticks_position('both')
    plt.imshow(cs)
    ax.set_title('Skin Lesion Cost Matrix')
    if save != False:
        plt.savefig('cost_matrix.png')

def get_AHC(D,pred,true):
    ''' Clculates the Average Hierarchy Cost.
    Args: - D (2D matrix) of class distance aka cost matrix
          - pred (list or tensor of integers) of predicted class idx
          - true (list or tensor of integers) of true class idx
    '''
    N = len(true) # N samples
    dist = 0
    for idx, p in enumerate(pred):
        dist += D[p,true[idx]]
        AHC = dist/N
        AHC = AHC.item()
    return  AHC # The AHC