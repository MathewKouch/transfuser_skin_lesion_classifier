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
from sklearn.metrics import classification_report, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from my_utilities2 import wandb_log
import copy
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

# Visualisaing prototypes
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.lines as mlines
import umap

class PrototypicalNetwork(nn.Module):
    def __init__(self, model,n_proto,n_emb,cost_matrix,device, single=False):
        '''
        A NN feature extractor in conjunction with prototype.
        Args:
        - model = (nn.Module) the NN FX
        - num_parameteres = (int) N number of prototypes/class or 65 in skin lesion molemap
        - emb_size = (int) size of the feature vectors extracted from model
        - cost matrix = (tensor) of shape N x N 2D matrix of class distance and class hierarchy
        - device = (str) cuda or cpu the device to be trained on
        '''

        super(PrototypicalNetwork, self).__init__()
        self.model = model.to(device)
        
        assert self.model.classifier[-1].out_features==n_emb, f'Model fc out_features {self.model.classifier[-1].out_features} does not must match n_emb of prototype'
        
        self.D = cost_matrix.to(device)
        self.n_proto = n_proto
        self.n_emb = n_emb
        self.Prototypes = nn.Parameter(torch.rand((n_proto, n_emb), device=torch.device(device))).requires_grad_(True) 
        self.device = device
        self.single = single

    def EUC_DISTS(self, p, emb):
        return torch.norm(p[:, None, :] - emb[None, :, :], dim=-1)
    
    def forward(self,MAC,MIC):
        '''
        Performs forward pass of skin lesion images, btain embedding
        Calc the Euc distance of each emb to all prototypes,
        spits out a distance
        '''

        if self.single:
            emb = self.model(MIC).to(self.device)
        else:
            emb = self.model(MAC,MIC).to(self.device) # (B x 1 x 65)
        
        dist = -EUC_DISTS(self.Prototypes, emb)
        
class DistortionLoss(nn.Module):
    """Scale-free squared distortion regularizer"""

    def __init__(self, D, scale_free=True):
        super(DistortionLoss, self).__init__()
        self.D = D
        self.scale_free = scale_free
        self.dist = Eucl_Mat()

    def forward(self, mapping, idxs=None):
        d = self.dist(mapping) # distance between prototypes. 

        a = d / (self.D + torch.eye(self.D.shape[0], device=self.D.device))
        scaling = a.sum() / torch.pow(a, 2).sum()


        d = (scaling * d - self.D) ** 2 / (
            self.D + torch.eye(self.D.shape[0], device=self.D.device)
        ) ** 2
        d = d.sum() / (d.shape[0] ** 2 - d.shape[0])
        
        return d

    
class Distortion(nn.Module):
    """Distortion measure of the embedding of finite metric given by matrix D into another metric space"""

    def __init__(self, D):
        """
        Args:
            D (tensor): 2D cost matrix of the finite metric, shape (NxN)
            dist: Distance to use in the target embedding space (euclidean or cosine)
        """
        super(Distortion, self).__init__()
        self.D = D
 

    def forward(self, mapping, idxs=None):
        """
        mapping (tensor):  Tensor of shape (N x Embedding_dimension) giving the mapping to the target metric space
        """
        d = torch.norm(mapping[:, None, :] - mapping[None, :, :], dim=-1)
        d = (d - self.D).abs() / (
            self.D + torch.eye(self.D.shape[0], device=self.D.device)
        )
        d = d.sum() / (d.shape[0] ** 2 - d.shape[0])
        return d
    
class Eucl_Mat(nn.Module):
    """Pairwise Euclidean distance"""

    def __init_(self):
        super(Eucl_Mat, self).__init__()

    def forward(self, mapping):
        """
        Args:
            mapping (tensor): Tensor of shape N_vectors x Embedding_dimension
        Returns:
            distances: Tensor of shape N_vectors x N_vectors giving the pairwise Euclidean distances

        """
        return torch.norm(mapping[:, None, :] - mapping[None, :, :], dim=-1)
    
class ScaleFreeDistortion(nn.Module):
    def __init__(self, D):
        super(ScaleFreeDistortion, self).__init__()
        self.D = D
        self.disto = Distortion(D)
        self.em = Eucl_Mat()

    def forward(self, prototypes):
        # Compute distance ratios
        d = self.em(prototypes)
        d = d / (self.D + torch.eye(self.D.shape[0], device=self.D.device))

        # Get sorted list of ratios
        alpha = d[d > 0].detach().cpu().numpy()
        alpha = np.sort(alpha)

        # Find optimal scaling
        cumul = np.cumsum(alpha)
        a_i = alpha[np.where(cumul >= alpha.sum() - cumul)[0].min()]
        scale = 1 / a_i

        return self.disto(scale * prototypes)
    
class ICD(nn.Module):
    """Intra Class Distance"""

    def __init_(self):
        super(ICD, self).__init__()

    def forward(self, mapping, p):
        """
        Args:
            mapping (tensor): Tensor of shape N_vectors x Embedding_dimension
            p (tensor):prototypes of Tensor of shape N_vectors x Embedding_dimension

        Returns:
            distances: Tensor of shape N_vectors x N_vectors giving the pairwise Euclidean distances

        """
        dist = -torch.norm(mapping-p, dim=-1, keepdim=True)
        dist = torch.exp(dist)+1
        dist = torch.sum(dist, dim=-1)
        dist = torch.log(dist)
        dist = torch.mean(dist)
        return dist
    
class Loss_DC(nn.Module):
    def __init__(self):
        super(Loss_DC, self).__init__()

    def Distance_Correlation(self, prototypes, distance_matrix=None):      
        matrix_a = torch.sqrt(torch.sum(torch.square(prototypes.unsqueeze(0) - prototypes.unsqueeze(1)), dim = -1) + 1e-12)

        matrix_b = distance_matrix

        matrix_A = (matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - 
                    torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a))
        
        matrix_B = (matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - 
                    torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b))

        Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

        correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r

    def forward(self, prototypes, distance_matrix=None):
        # maximize the correlation
        dc_loss = -self.Distance_Correlation(prototypes, distance_matrix)

        return dc_loss  

def get_accuracy(score,label):
    acc = (torch.argmax(score,1)==label).type(torch.DoubleTensor).mean()
    return acc

def get_dist(p):
    '''
    Calculates the Euc Norm of the distances between all prototypes
    arg: -p (torch tensor) the K x emb matrix of prototypes
    return (numpy) of the prototype class distance symmetric matrix
    '''
    return torch.norm(p[:, None, :] - p[None, :, :], dim=-1).detach().cpu().numpy()
    
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
    return dist/N # The AHC

def train_prototype(model,l_metric, l_dc, DEVICE, data_loader, optimizer, criterion, distortion, epoch,tf,D, CLASSES, SAVE_DIR, SINGLE=False):
    '''
    Train the prototypical network.
    - model (nn.Module) is the prototypical nextwork, with NN FX backbone, and randomly intiated prototypes.
    - l_metrix (float) setting it to 0 means no metric guidance.
    - DEVICE (str) cuda/cpu
    - data_loader
    - optimizer
    - epoch (current epoch)
    - tf (list of transforms) one per MAC and MIC
    ''' 
    losses = []
    accuracies = []
    count = 0
    correct = 0
    
    all_prob = []
    all_pred = []
    all_labels = []
    
    corr_loss = Loss_DC() #correlation loss b/w prototypes and cost distance
    
    model.train()
    loader = tqdm(data_loader)
    for batch_idx, (labels,_,_,MAC_img,MIC_img) in enumerate(loader):
        
        labels = labels.to(DEVICE) # labels are prob of 
        
        optimizer.zero_grad()
        
        if len(tf)>1:
            
            
            MAC_img = tf[0](MAC_img).to(DEVICE)
            MIC_img = tf[1](MIC_img).to(DEVICE)
            emb, dists = model(MAC_img,MIC_img) #logits are now distances, want least negative
        else:
            MAC_img, MIC_img = tf[0](MAC_img, MIC_img, labels)
            
            MAC_img = MAC_img.to(DEVICE)
            MIC_img = MIC_img.to(DEVICE)
            
            if SINGLE==False:
                emb, dists = model(MAC_img ,MIC_img) #logits are now distances, want least negative
            else:
                emb, dists = model(MIC_img)
                
        loss_data = criterion(dists,labels)
        distortion_loss = distortion(model.Prototypes) 
        loss = loss_data + l_metric*distortion_loss
        
        loss = loss + l_dc*corr_los(model.Prototypes, D)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            dists = dists.detach()
            # Class predictions from dist between feature and prototypes
            pred = torch.argmax(dists,1).detach()
            count += len(labels)
            correct += (pred==labels).sum()
            acc = correct/count # the cumulative accuracy
            accuracies.append(acc)
            losses.append(loss.item())
            
            all_pred += pred.cpu()
            all_labels += labels.cpu()
            all_prob += F.softmax(dists.cpu(), -1)


            if batch_idx%10==0 or ((batch_idx+1)==len(loader)):
                prototypes = model.Prototypes.clone().detach()
                proto_dis = get_dist(prototypes)
                proto_dis = proto_dis/np.max(proto_dis) # normalise the distances

                proto_image = wandb.Image(proto_dis)
                
                # Visualising prototypes
                if model.Prototypes.data.shape[-1]==2:
                    data = [[x, y] for (x, y) in zip(prototypes[:,0], prototypes[:,1])]
                    clustering = wandb.Table(data=data, columns = ["x", "y"])
                    wandb.log({"2D Prototype Clustering" : wandb.plot.scatter(clustering, "x", "y")})
                        
                AVG_loss = torch.tensor(losses).mean()
                wandb.log({"train_running_loss": AVG_loss,
                          "train_running_acc":acc,
                          'proto_dist': proto_image,
                           'train_distortion_loss': distortion_loss.item(),
                          }) 
                
                loader.set_description(f'TRAIN | epoch: {epoch} | ACC: {acc*100:.2f}% | LOSS: {AVG_loss:.4f}')

    with torch.no_grad():

        # AUC-ROC
        all_probs_ = torch.stack(all_prob)
        roc_auc_ovr_weighted = roc_auc_score(all_labels, all_probs_, multi_class='ovr', average='weighted')
        roc_auc_ovr_macro = roc_auc_score(all_labels, all_probs_, multi_class='ovr', average='macro')

        # Precision, Recall, F1-score
        prf1_report_weighted = precision_recall_fscore_support(all_labels, all_pred, average='weighted', zero_division=0)
        prf1_report_macro = precision_recall_fscore_support(all_labels, all_pred, average='macro', zero_division=0)

        precision_weighted = prf1_report_weighted[0]
        recall_weighted = prf1_report_weighted[1]
        f1_weighted = prf1_report_weighted[2]

        precision_macro = prf1_report_macro[0]
        recall_macro= prf1_report_macro[1]
        f1_macro = prf1_report_macro[2]

        # Class precision, recall, f1 REPORT
        report = classification_report(all_labels,
                                       all_pred,
                                       labels=np.arange(len(CLASSES)), 
                                       target_names=CLASSES, 
                                       sample_weight=None, 
                                       digits=2, 
                                       output_dict=True, 
                                       zero_division=0)
        AHC = get_AHC(D, all_pred, all_labels)

        # Metrics
        if epoch==0:
            class_count = []
            for c in CLASSES:
                class_count.append(report[c]['support'])
            data = [[label, val] for (label, val) in zip(CLASSES, class_count)]
            table = wandb.Table(data=data, columns = ["classes", "count"])
            wandb.log({"train_class_count": wandb.plot.bar(table, "classes", "count", title="Train Class Count")})
        
        wandb.define_metric('train_acc', summary='max')
        wandb.define_metric('train_loss', summary='min')
        wandb.define_metric('train_AHC', summary='min')
    
        wandb.log({
            'epochs': epoch,
            'epoch': epoch,
            'train_acc': acc,
            'train_loss': AVG_loss,
            'train_roc_auc_ovr_weighted': roc_auc_ovr_weighted,
            'train_roc_auc_ovr_macro': roc_auc_ovr_macro,
            'train_precision_weighted': precision_weighted,
            'train_recall_weighted': recall_weighted,
            'train_f1_weighted' : f1_weighted,
            'train_precision_macro': precision_macro,
            'train_recall_macro': recall_macro,
            'train_f1_macro' : f1_macro,
            #'train_report': report,
            'train_AHC': AHC,
            'train_distortion_loss': distortion_loss.item(),

        })
        
        wandb_log(
            all_prob,
            all_pred,
            all_labels,
            [acc, AVG_loss],
            epoch, 
            CLASSES,
            SAVE_DIR,
            D,
            status='train',
        )
        
        prototypes = model.Prototypes.clone().detach().cpu()

        metrics = {'loss': AVG_loss,
                   'acc': acc,
                   'losses': losses,
                   'accuracies': accuracies,
                   'loss_meter':losses, 
                   'loss': AVG_loss,
                   'prototypes':prototypes
                  }

        return metrics
    
def evaluate_prototype(model,l_metric,l_dc,DEVICE,data_loader, criterion,distortion,epoch,tf, D, CLASSES, SAVE_DIR, SINGLE=False):

    losses = []
    accuracies = []
    
    count = 0
    correct = 0

    # TO calc ROC-AUC
    all_prob = [] # contains all output class prob
    all_pred = [] # contains all ouput class predictions
    all_labels = [] # contains all true labels
    
    corr_loss = Loss_DC()
    
    model.eval()
    with torch.no_grad():
        loader = tqdm(data_loader)
        for batch_idx, (labels,_,_,MAC_img,MIC_img) in enumerate(loader):
            labels = labels.to(DEVICE)
            if len(tf)>1:
                MAC_img = tf[0](MAC_img).to(DEVICE)
                MIC_img = tf[1](MIC_img).to(DEVICE)
                
                emb, dists = model(MAC_img,MIC_img) #logits are now distances, want least negative
            else:
                MAC_img, MIC_img = tf[0](MAC_img, MIC_img, labels)

                MAC_img = MAC_img.to(DEVICE)
                MIC_img = MIC_img.to(DEVICE)

                if SINGLE==False:
                    emb, dists = model(MAC_img,MIC_img) #logits are now distances, want least negative
                else:
                    emb, dists = model(MIC_img)

            loss_data = criterion(dists,labels)
            distortion_loss = distortion(model.Prototypes) 
            loss_dc = corr_loss(model.Prototypes, D)
            
            loss = loss_data + l_metric*distortion_loss + l_dc*loss_dc

            with torch.no_grad():
                
                distortion_loss = distortion(model.Prototypes)
                loss = criterion(dists, labels) + l_metric*distortion_loss

                pred = torch.argmax(dists ,1).detach()
                correct += (pred==labels).sum()
                count += len(labels)
                acc = correct/count
                losses.append(loss.item())
                accuracies.append(acc)
                AVG_loss = torch.tensor(losses).mean()
                
                all_pred += pred.cpu()
                all_labels += labels.cpu()
                all_prob += F.softmax(dists.cpu(), -1)
            
                # Log prototypes distances as image every 10 epochs
                prototypes = model.Prototypes.clone().detach()
                proto_dis = get_dist(prototypes)
                proto_dis = proto_dis/np.max(proto_dis) # normalise the distances
                eval_proto_image = wandb.Image(proto_dis)
                
                
                loader.set_description(f'EVAL | epoch: {epoch} | ACC: {100*acc:.2f}% | Loss: {AVG_loss:0.5f}')
            
         # Class precision, recall, f1 REPORT
        report = classification_report(all_labels,
                                       all_pred,
                                       labels=np.arange(len(CLASSES)), 
                                       target_names=CLASSES, 
                                       sample_weight=None, 
                                       digits=2, 
                                       output_dict=True, 
                                       zero_division=0)

        # Metrics
        if epoch==0:
            class_count = []
            for c in CLASSES:
                class_count.append(report[c]['support'])
            data = [[label, val] for (label, val) in zip(CLASSES, class_count)]
            table = wandb.Table(data=data, columns = ["classes", "count"])
            wandb.log({"val_class_count" : wandb.plot.bar(table, "classes", "count", title="Eval Class Count")})

        # ROC_AUC
        all_probs_ = torch.stack(all_prob)
        roc_auc_ovr_weighted = roc_auc_score(all_labels, all_probs_, multi_class='ovr', average='weighted')
        roc_auc_ovr_macro = roc_auc_score(all_labels, all_probs_, multi_class='ovr', average='macro')
      
        # Precision, Recall, F1-score
        prf1_report_weighted = precision_recall_fscore_support(all_labels, all_pred, average='weighted', zero_division=0)
        prf1_report_macro = precision_recall_fscore_support(all_labels, all_pred, average='macro', zero_division=0)

        precision_weighted = prf1_report_weighted[0]
        recall_weighted = prf1_report_weighted[1]
        f1_weighted = prf1_report_weighted[2]

        precision_macro = prf1_report_macro[0]
        recall_macro= prf1_report_macro[1]
        f1_macro = prf1_report_macro[2]

            # Class Hierarhy
        AHC = get_AHC(D, all_pred, all_labels)
        
        prototypes = model.Prototypes.clone().detach().cpu()

        metrics = {'loss': AVG_loss,
                   'acc': acc,
                   'losses': losses,
                   'accuracies': accuracies,
                   'ahc': AHC,
                   'prototypes': prototypes,
                   'distortion_loss': distortion_loss.item(),
                  }

        wandb.define_metric('val_acc', summary='max')
        wandb.define_metric('val_loss', summary='min')
        wandb.define_metric('eval_AHC', summary='min')
    
        wandb.log({
            'val_acc': acc,
            'eval_acc': acc,
            'eval_loss': AVG_loss,
            'val_loss': AVG_loss,
            'epochs': epoch,
            'epoch': epoch,
            'eval_roc_auc_ovr_weighted': roc_auc_ovr_weighted,
            'eval_roc_auc_ovr_macro': roc_auc_ovr_macro,
            'eval_precision_weighted': precision_weighted,
            'eval_recall_weighted': recall_weighted,
            'eval_f1_weighted' : f1_weighted,
            'eval_precision_macro': precision_macro,
            'eval_recall_macro': recall_macro,
            'eval_f1_macro' : f1_macro,
            'eval_proto': eval_proto_image,
            'eval_AHC': AHC,
            'eval_distortion_loss': distortion_loss.item(),
        })

        wandb_log(
            all_prob,
            all_pred,
            all_labels,
            [acc, AVG_loss],
            epoch, 
            CLASSES,
            SAVE_DIR,
            D,
            status='eval',
        )
        return metrics


def class_idx(CLASSES):
    ben = [] # list of benign labels

    # benign super classes
    bk = [] 
    bm = []
    bo = []
    bv = []

    mal = [] # list of malignant labels

    # malignant super classes
    mb = []
    mk = []
    mm = []
    ms = []

    for idx, cls_ in enumerate(CLASSES):
        splits = cls_.split('_')

        cond = splits[0]
        supercls = splits[1]
        class_ = splits[2]

        # Condition
        if cond=='benign':
            ben.append(idx)
            if supercls=='keratinocytic':
                bk.append(idx)
            elif supercls=='melanocytic':
                bm.append(idx)
            elif supercls=='other':
                bo.append(idx)
            elif supercls=='vascular':
                bv.append(idx)

        elif cond=='malignant':
            mal.append(idx)
            if supercls=='bcc':
                mb.append(idx)
            elif supercls=='keratinocytic':
                mk.append(idx)
            elif supercls=='melanoma':
                mm.append(idx)
            elif supercls=='scc':
                ms.append(idx)

    out = {'ben':ben,
        'bk':bk,
        'bm':bm,
        'bo':bo,
        'bv':bv,
        'mal':mal,
        'mb':mb,
        'mk':mk,
        'mm':mm,
        'ms':ms,}

    return out
    
def proto_vis(p, fx, label, CLASSES, clustering_gif, epoch, iterations, SAVE_DIR, dim=2, n_iter=10000, alpha=1, size=20, save_fig=False, show_legend=False, make_gif=False, method='umap', status='status'):
    '''
    Performs dimension reductions of prototypes (p), fx(features), and log to wandb, create gif, or save images.
    '''
    #from visualization import ANN
    n = p.shape[0] # number of prototypes
    ndim = p.shape[-1] # dimension of each prototypes
    class_segmented = class_idx(CLASSES)

    cc = cm.get_cmap('rainbow_r', 2048)

    cmap_proto = np.asarray([cc((i+1)/65) for i in range(len(CLASSES))])
              
    v = copy.deepcopy(p).cpu().detach()
    
    if fx is not None and label is not None:
        cmap_fx = cmap_proto[label]
        
        reducer = umap.UMAP(n_neighbors=65)
        fx_r = reducer.fit_transform(fx)
        
        reducer = umap.UMAP(n_neighbors=65)
        p_r = reducer.fit_transform(copy.deepcopy(p).cpu().detach())
        
        v = torch.cat([v, fx], dim=0)
        
    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(dim, verbose=0, n_iter=n_iter)
    emb_tsne = tsne.fit_transform(v)
    
    # Reducing proto and fx seprately
    reducer = umap.UMAP(n_neighbors=65)
    emb = reducer.fit_transform(v)    
    
    
    fig, ax = plt.subplots(figsize=(8,8), dpi=(128))
    num_categories = v.shape[0]
    
    # Plot prototypes on top of features         
    ax.scatter(emb_tsne[:n,0], emb_tsne[:n,1], c=cmap_proto, label = CLASSES ,alpha=alpha, marker='o', s=10)          
    for idx, c in enumerate(CLASSES):
        ax.text(emb_tsne[idx,0], emb_tsne[idx,1], str(idx), color='black', fontsize=6, 
        ha="center", va="center",
        bbox=dict(boxstyle="round", ec='black', fc=cmap_proto[idx], alpha=0.90))
                
    fig.set(facecolor="black")         
    ax.axis('off')
    figure_tsne = fig.get_figure()
    plt.close(figure_tsne) # ensures closing all plot/figure windows

    fig, ax = plt.subplots(figsize=(8,8), dpi=(128))
    
    # Plot features          
    if fx is not None:
        fx = emb[n:] # features
        ax.scatter(emb[n:,0],emb[n:,1],c=cmap_fx, marker='.', s=1)
    
    # Plot prototypes on top of features         
    ax.scatter(emb[:n,0], emb[:n,1], c=cmap_proto, label = CLASSES ,alpha=alpha, marker='o', s=10)          
    for idx, c in enumerate(CLASSES):
        ax.text(emb[idx,0], emb[idx,1], str(idx), color='black', fontsize=6, 
        ha="center", va="center",
        bbox=dict(boxstyle="round",
                    ec='black',
                    fc=cmap_proto[idx],
                    alpha=0.90,
                )
        )

    fig.set(facecolor="black")
              
    ax.axis('off')

    figure = fig.get_figure()
    # Reducing proto and fx seprately
    
    fig, ax = plt.subplots(figsize=(8,8), dpi=(128))
    
    if fx is not None:
        # Plot features          
        ax.scatter(fx_r[:,0], fx_r[:,1], c=cmap_fx, marker='.', s=1)
  
        # Plot prototypes on top of features         
        ax.scatter(p_r[:,0], p_r[:,1],c=cmap_proto, label = CLASSES ,alpha=alpha, marker='o', s=10)          
        fig.set(facecolor="black")
        figure2 = fig.get_figure()
        clustering_gif.append(figure2)
        ax.axis('off')
        plt.close(figure2) # ensures closing all plot/figure windows
        
    # Plot features          
    if fx is not None:
        fig, ax = plt.subplots(figsize=(8,8), dpi=(128))
        ax.scatter(fx_r[:,0], fx_r[:,1], c=cmap_fx, marker='.', s=1)   
        fig.set(facecolor="black")
        figure3 = fig.get_figure()
        ax.axis('off')
        
        plt.close(figure3) # ensures closing all plot/figure windows
        
    fig, ax = plt.subplots(figsize=(8,8), dpi=(128))
    ax.scatter(emb[:n,0], emb[:n,1],c=cmap_proto, label = CLASSES ,alpha=alpha, marker='o', s=10)          
    fig.set(facecolor="black")
    figure4 = fig.get_figure()
    ax.axis('off')
    plt.close(figure4) # ensures closing all plot/figure windows
       
    SAVE_DIR = os.path.join(SAVE_DIR, 'prototype_clustering') # folder to save figure

    if os.path.exists(SAVE_DIR):
        pass
    else:
        os.mkdir(SAVE_DIR)
        
    if save_fig:
        figure.savefig(os.path.join(SAVE_DIR, f'prototypes_epoch_{epoch}_{iterations}.png'), dpi=128)
     
    if make_gif:
        
        assert os.path.exists(SAVE_DIR), 'prototype_clustering folder not created. Need to create with images first.'
        
        SAVE_DIR_GIF = os.path.join(SAVE_DIR, 'prototype_clustering.gif') # name of clustering gif

        if os.path.exists(SAVE_DIR_GIF):
            pass
        else: 
            os.mkdir(SAVE_DIR_GIF)
            
        images = []
        
        for img_file in os.listdir(SAVE_DIR):
            if img_file.endswith('.png'):
                img = Image.open(os.path.join(SAVE_DIR, img_file))# PIL image
                img = transforms.ToTensor()(img)
            imgs_np = (img.numpy().transpose((1, 2, 0))*255).astype(np.uint8)
                                 
            images.append(imgs_np)
        
        imageio.mimsave(os.path.join(SAVE_DIR_GIF, 'prototype_clustering.gif'), images, fps=1)
        
     # ensures closing all plot/figure windows
    plt.close(figure)    

    if fx is None:
        wandb.log({
            'epoch':epoch,
            'epochs': epoch,
            f'{status} prototypes clustering': wandb.Image(figure),
            f'{status} prototypes': wandb.Image(figure4),

        })
    else:
        wandb.log({
            'epoch':epoch,
            'epochs': epoch,
            f'prototypes and {status} features': wandb.Image(figure),
            f'{status} prototypes and {status} features overlay': wandb.Image(figure2),
            f'{status} features': wandb.Image(figure3),
            f'{status} prototypes': wandb.Image(figure4),
            f'{status} prototypes tsne': wandb.Image(figure_tsne),
        })
       
