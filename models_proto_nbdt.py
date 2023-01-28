import os # directory

import torch
import torch.nn as nn # For NN modules
import torch.nn.functional as F # For activations and utilies

from sklearn.cluster import AgglomerativeClustering # to build tree
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from scipy.special import softmax
import math
import networkx as nx
from prototypical import get_dist

class Hierarchy_prototype(nn.Module):
    '''
    Build hierarchy using prototypes that are jointly learnt with network backbone end to end.
    '''
    def __init__(self, W, draw_tree=False, device='cpu'):
        '''
        Builds the hierarchy from randomly initiated prototypes, W.
        '''
        super(Hierarchy_prototype,self).__init__()
        self.W = W

        self.draw_tree = draw_tree # flag to record node prob as graph attributes during training
        self.leaves_prob = None # to hold the calculated leaves path prob after pasing samples
        self.node_prob = None # to hold the calculated nodes prob
        self.DEVICE = device
        
        ############# Setting weights of all nodes #############
   
        self.leaves = self.W

        self.weights = torch.zeros((76,self.leaves.shape[-1]), device=self.DEVICE)
        
        # setting leave weights
        self.weights[0:65,:] = self.leaves

        # RBK
        self.weights[65,:] = self.leaves[0:7,:].mean(dim=0)
        # RBM
        self.weights[66,:] = self.leaves[7:30,:].mean(dim=0)
        # RBO
        self.weights[67,:] = self.leaves[30:48,:].mean(dim=0)
        # RBV
        self.weights[68,:] = self.leaves[48:53,:].mean(dim=0)
        
        # RMB
        self.weights[69,:] = self.leaves[53:58,:].mean(dim=0)
        # RMK
        self.weights[70,:] = self.leaves[58,:].mean(dim=0)
        # RMM
        self.weights[71,:] = self.leaves[59:62,:].mean(dim=0)
        # RMS
        self.weights[72,:] = self.leaves[62:65,:].mean(dim=0)
        
        # Benign
        self.weights[73,:] = self.leaves[0:53,:].mean(dim=0)
        
        # Malignant
        self.weights[74,:] = self.leaves[53:65,:].mean(dim=0)
        
        # Root
        self.weights[75,:] = self.leaves.mean(dim=0)

        #################### Root to Leaves Pathways #########################    
        self.leaf_paths = {}
        # Root to leaf path template for every class in the 8 super classes
        RBK = [75,73,65,0]
        RBM = [75,73,66,0]
        RBO = [75,73,67,0]
        RBV = [75,73,68,0]
        RMB = [75,74,69,0]
        RMK = [75,74,70,0]
        RMM = [75,74,71,0]
        RMS = [75,74,72,0]
        
        for leaf in range(65):
            if leaf in np.arange(7): 
                _path = RBK.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(7,30):
                _path = RBM.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(30,48):
                _path = RBO.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(48,53):
                _path = RBV.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(53,58):
                _path = RMB.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf == 58:
                _path = RMK.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(59,62):
                _path = RMM.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(62,65):
                _path = RMS.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
    
    
        ########### Initialising Graph #############################
        self.G = nx.DiGraph() # Directed graph to visualise node weights, path probs, leaf path prob
        
        # Adding 65 leaf nodes, 8 super classe nodes, 2 conditional nodes, 1 root node
        for leaf in range(65+8+2+1):
            self.G.add_node(leaf)
        
        self.G.add_edge(75,73) # Root to Benign
        self.G.add_edge(75,74) # Root to Malignant
        
        for bs in range(65,69): # Benign to superclasses
            self.G.add_edge(73,bs)
        
        for ms in range(69,73): # Maligant to superclasses
            self.G.add_edge(74,ms)

        for leaf in range(65): # Super Classes to Leaves
            if leaf in np.arange(7):
                self.G.add_edge(65, leaf)
            elif leaf in np.arange(7,30):
                self.G.add_edge(66, leaf)
            elif leaf in np.arange(30,48):
                self.G.add_edge(67, leaf)
            elif leaf in np.arange(48,53):
                self.G.add_edge(68, leaf)
            elif leaf in np.arange(53,58):
                self.G.add_edge(69, leaf)
            elif leaf == 58:
                self.G.add_edge(70, leaf)
            elif leaf in np.arange(59,62):
                self.G.add_edge(71, leaf)
            elif leaf in np.arange(62,65):
                self.G.add_edge(72, leaf)

        # Dict to fill node weights as attributes in graph
        attr = {node: self.weights[node] for node in range(76)}    
        # Fill the weights in each node 'w' attribute    
        nx.set_node_attributes(self.G, attr, name='w') 
        
        #self.weights.requires_grad = True

        print('Inner nodes and leaves weights loaded. Hierarchy tree built!')
        
    def traverse(self, x, node_probs, parent=75):
        '''
        Pass x, the featurised sample of backbone into tree.
        Calculate node prob for all nodes.
        Returns a dictionary of all node and their path prob from their parent node.
        '''

        children = list(self.G.succ[parent].keys())
        # Dot prod of samples with children weights is Nx1 tensor
        prods = []
        for child in children:
            # Euclidean distance from every fx sample to a prototype
            # Multiply dist by -1 because we want to maximise the softmax score. 
            # i.e smaller negative dist gives biggest prob
            p = self.weights[child].view(1,-1) # Make prototype 2D
            dist_ = -torch.norm(p[:, None, :] - x[None, :, :], dim=-1) 
            prods.append(dist_)
        
        # prods is now Nx2 tensor column concatenated as [prod with child0, prod with child1]
        prods = torch.stack(prods, dim=-1)
        # softmax along every row (a samples) dot product. Result is N x 2, the children prob per sample
        probs = F.softmax(prods, dim = -1).view(x.shape[0],len(children))
    
        node_probs[:,children] = probs

        for child in children:
            if child>64:
                _ = self.traverse(x, node_probs, parent=child) # Not bug, function seeas node_probs in all calls
        return node_probs
    
    def node_path(self, Gr, start, root=75):
        '''
        Returns a list representing the path from root node to the start leaf.

        Gr = reverse view of original graph G.^
        start = the start leaf
        root = node label or index of root

        ^ It is much easier to traverse from leaf to root via the reverse view 
        of the DiGraph as there is only one succesor from every node (except root - with none) in reverse 
        direction and not two in original direction.
        '''

        if start == root: # Base case. Exitting out of recursion
            return []
        parent = list(Gr.succ[start].keys())[0]
        next_path = self.node_path(Gr, parent)
        next_path.append(start)

        return sorted(next_path, reverse=True)
    
    def get_leaf_prob(self, output):
        '''
        output is a batch of tensors from NN, the logits or featurised samples.
        return all the path probabilities of every leaf, for every tensor in batch.
        '''
        N, emb_dim = output.shape # Number of samples, and size of each sample vector

        # Tensor of all node probs, each row is the node prob of one sample in the batch
        node_probs = torch.zeros((N, 76), device=self.DEVICE) # output.shape[0] is the batch dimension
        node_probs[:,75] = 1.0
        
        # Tensor of all node probs, each row is the node prob of one sample in the batch,
        # columns are node index.
        node_probs = self.traverse(output, node_probs)

        leaf_path_probs = torch.zeros((N,65), device=self.DEVICE)# prefilled leaf path prob, N rows per samples
        
        for i in range(65): # Get path prob for every leaf
            leaf_path_probs[:,i] = torch.prod(node_probs[:, self.leaf_paths[i]],dim=1)
        
        if self.draw_tree==True:
            nx.set_node_attributes(self.G, node_probs, name='node_prob') # THEN add the node prob as node atrributes 
            self.leaves_prob = leaf_path_probs.clone().detach()
            self.node_prob = node_probs.clone().detach()
        
        return leaf_path_probs, node_probs
    
       
class Hierarchy_molemap_logits(nn.Module):
    '''
    Use logits instead as product of weights and f.
    Use logits of children to workout prob via softmax, and to calc root to leaf path prob
    '''
    def __init__(self, draw_tree=False, device='cpu'):
        '''
        Hierarchy of molemap is fixed.       
        '''
        super(Hierarchy_molemap_logits, self).__init__()
        self.DEVICE = device
        
        #################### Root to Leaves Pathways #########################    
        self.leaf_paths = {}
        # Root to leaf path template for every class in the 8 super classes
        RBK = [75,73,65,0]
        RBM = [75,73,66,0]
        RBO = [75,73,67,0]
        RBV = [75,73,68,0]
        RMB = [75,74,69,0]
        RMK = [75,74,70,0]
        RMM = [75,74,71,0]
        RMS = [75,74,72,0]
        
        for leaf in range(65):
            if leaf in np.arange(7): 
                _path = RBK.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(7,30):
                _path = RBM.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(30,48):
                _path = RBO.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(48,53):
                _path = RBV.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(53,58):
                _path = RMB.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf == 58:
                _path = RMK.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(59,62):
                _path = RMM.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(62,65):
                _path = RMS.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
    
        ##################### The leaves of internal nodes#####################
        self.node_leaves = {} # Only holds internal nodes key values of internal nodes index 65-76
        
        for inode in range(65,76):
            leaves = list()
            for node, root_to_leaf in self.leaf_paths.items():
                if inode in root_to_leaf:
                    leaves.append(root_to_leaf[-1])
                    
            self.node_leaves[inode] = leaves
            
        
        ########### Initialising Graph #############################
        self.G = nx.DiGraph() # Directed graph to visualise node weights, path probs, leaf path prob
        
        # Adding 65 leaf nodes, 8 super classe nodes, 2 conditional nodes, 1 root node
        for leaf in range(65+8+2+1):
            self.G.add_node(leaf)
        
        self.G.add_edge(75,73) # Root to Benign
        self.G.add_edge(75,74) # Root to Malignant
        
        for bs in range(65,69): # Benign to superclasses
            self.G.add_edge(73,bs)
        
        for ms in range(69,73): # Maligant to superclasses
            self.G.add_edge(74,ms)

        for leaf in range(65): # Super Classes to Leaves
            if leaf in np.arange(7):
                self.G.add_edge(65, leaf)
            elif leaf in np.arange(7,30):
                self.G.add_edge(66, leaf)
            elif leaf in np.arange(30,48):
                self.G.add_edge(67, leaf)
            elif leaf in np.arange(48,53):
                self.G.add_edge(68, leaf)
            elif leaf in np.arange(53,58):
                self.G.add_edge(69, leaf)
            elif leaf == 58:
                self.G.add_edge(70, leaf)
            elif leaf in np.arange(59,62):
                self.G.add_edge(71, leaf)
            elif leaf in np.arange(62,65):
                self.G.add_edge(72, leaf)

        print('Inner nodes and leaves weights loaded. Hierarchy tree built!')
        
    def traverse(self, x, node_probs, parent=75):
        '''
        x = 1D tensor of logit,
        node_probs = tensor of children prob.
        Via Soft decision rule.
        Returns a dictionary of all node and their path prob from their parent node.
        '''
        x = x.view(-1,65)
        N, embd = x.shape
        
        children = list(self.G.succ[parent].keys())
        # Dot prod of samples with children weights is Nx1 tensor
        prods = []
        for child in children:
            # logit of internal node as mean of its leaves logits
            if child>64:
                child_mean_logit = torch.mean(x[:,self.node_leaves[child]], dim=-1).view(-1,1)  
            else:
                child_mean_logit = x[:,child].view(-1,1)
                
            prods.append(child_mean_logit)
        
        # prods is a tensor of N x number_of_children 
        prods = torch.stack(prods, dim=-1)
        # softmax along every row (a samples) dot product. Result is N x 2, the children prob per sample
        probs = F.softmax(prods, dim = -1).view(N,len(children))
    
        node_probs[:,children] = probs

        for child in children:
            if child>64:
                node_probs = self.traverse(x, node_probs, parent=child) # Not bug, function seeas node_probs in all calls
        return node_probs
    
    def node_path(self, Gr, start, root=75):
        '''
        Returns a list representing the path from root node to the start leaf.

        Gr = reverse view of original graph G.^
        start = the start leaf
        root = node label or index of root

        ^ It is much easier to traverse from leaf to root via the reverse view 
        of the DiGraph as there is only one succesor from every node (except root - with none) in reverse 
        direction and not two in original direction.
        '''

        if start == root: # Base case. Exitting out of recursion
            return []
        parent = list(Gr.succ[start].keys())[0]
        next_path = self.node_path(Gr, parent)
        next_path.append(start)

        return sorted(next_path, reverse=True)
    
    def get_leaf_prob(self, output):
        '''
        output is a batch logits NN
        leaf_path_prob = B x 65 root to leaf path prob for every sample in B 
        node_probs = B x 76 tensor of node probability (edge)
        '''
        N, emb_dim = output.shape # Number of samples, and size of each sample vector

        # Tensor of all node probs, each row is the node prob of one sample in the batch
        node_probs = torch.zeros((N, 76), device=self.DEVICE) # output.shape[0] is the batch dimension
        node_probs[:,75] = 1.0
        
        # Tensor of all node probs, each row is the node prob of one sample in the batch,
        # columns are node index.
        node_probs = self.traverse(output, node_probs)

        leaf_path_probs = torch.zeros((N,65), device=self.DEVICE)# prefilled leaf path prob, N rows per samples
        
        for i in range(65):
            leaf_path_probs[:,i] = torch.prod(node_probs[:, self.leaf_paths[i]],dim=1)
        
        return leaf_path_probs, node_probs
    


class Hierarchy_molemap(nn.Module):
    '''
    Build Hierarcy from trained weights of final fully connected layer (W).
    Returns leaf prob given an input tensor or batch.
    
    ! All tensors should be on CUDA except for clustering and children
    '''
    def __init__(self, W, draw_tree=False, device='cpu'):
        '''
        Builds the Hierarchy upon init from pretrained fc weights, W. 
        W will not be updated.
        W will need to be a copy of fc weights
        '''
        super(Hierarchy_molemap,self).__init__()
        self.draw_tree = draw_tree # flag to record node prob as graph attributes during training
        self.leaves_prob = None # to hold the calculated leaves path prob after pasing samples
        self.node_prob = None # to hold the calculated nodes prob
        self.DEVICE = device
        
        ############# Setting weights of all nodes #############
        
        # normalised weights for leaves
        self.leaves = W/(torch.linalg.norm(W, dim=-1).reshape(65,1)) 
        self.leaves = self.leaves.to(self.DEVICE) # 65 x 512
       
        self.weights = torch.zeros((76,512), device=self.DEVICE)
        
        # setting leave weights
        self.weights[0:65,:] = self.leaves

        # RBK
        self.weights[65,:] = self.leaves[0:7,:].mean(dim=0)
        # RBM
        self.weights[66,:] = self.leaves[7:30,:].mean(dim=0)
        # RBO
        self.weights[67,:] = self.leaves[30:48,:].mean(dim=0)
        # RBV
        self.weights[68,:] = self.leaves[48:53,:].mean(dim=0)
        
        # RMB
        self.weights[69,:] = self.leaves[53:58,:].mean(dim=0)
        # RMK
        self.weights[70,:] = self.leaves[58,:].mean(dim=0)
        # RMM
        self.weights[71,:] = self.leaves[59:62,:].mean(dim=0)
        # RMS
        self.weights[72,:] = self.leaves[62:65,:].mean(dim=0)
        
        # Benign
        self.weights[73,:] = self.leaves[0:53,:].mean(dim=0)
        
        # Malignant
        self.weights[74,:] = self.leaves[53:65,:].mean(dim=0)
        
        # Root
        self.weights[75,:] = self.leaves.mean(dim=0)

        #################### Root to Leaves Pathways #########################    
        self.leaf_paths = {}
        # Root to leaf path template for every class in the 8 super classes
        RBK = [75,73,65,0]
        RBM = [75,73,66,0]
        RBO = [75,73,67,0]
        RBV = [75,73,68,0]
        RMB = [75,74,69,0]
        RMK = [75,74,70,0]
        RMM = [75,74,71,0]
        RMS = [75,74,72,0]
        
        for leaf in range(65):
            if leaf in np.arange(7): 
                _path = RBK.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(7,30):
                _path = RBM.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(30,48):
                _path = RBO.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(48,53):
                _path = RBV.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(53,58):
                _path = RMB.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf == 58:
                _path = RMK.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(59,62):
                _path = RMM.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
            elif leaf in np.arange(62,65):
                _path = RMS.copy()
                _path[-1] = leaf
                self.leaf_paths[leaf] = _path
    
    
        ########### Initialising Graph #############################
        self.G = nx.DiGraph() # Directed graph to visualise node weights, path probs, leaf path prob
        
        # Adding 65 leaf nodes, 8 super classe nodes, 2 conditional nodes, 1 root node
        for leaf in range(65+8+2+1):
            self.G.add_node(leaf)
        
        self.G.add_edge(75,73) # Root to Benign
        self.G.add_edge(75,74) # Root to Malignant
        
        for bs in range(65,69): # Benign to superclasses
            self.G.add_edge(73,bs)
        
        for ms in range(69,73): # Maligant to superclasses
            self.G.add_edge(74,ms)

        for leaf in range(65): # Super Classes to Leaves
            if leaf in np.arange(7):
                self.G.add_edge(65, leaf)
            elif leaf in np.arange(7,30):
                self.G.add_edge(66, leaf)
            elif leaf in np.arange(30,48):
                self.G.add_edge(67, leaf)
            elif leaf in np.arange(48,53):
                self.G.add_edge(68, leaf)
            elif leaf in np.arange(53,58):
                self.G.add_edge(69, leaf)
            elif leaf == 58:
                self.G.add_edge(70, leaf)
            elif leaf in np.arange(59,62):
                self.G.add_edge(71, leaf)
            elif leaf in np.arange(62,65):
                self.G.add_edge(72, leaf)

        # Dict to fill node weights as attributes in graph
        attr = {node: self.weights[node] for node in range(76)}    
        # Fill the weights in each node 'w' attribute    
        nx.set_node_attributes(self.G, attr, name='w') 
        
        #self.weights.requires_grad = True

        print('Inner nodes and leaves weights loaded. Hierarchy tree built!')
        
    def traverse(self, x, node_probs, parent=75):
        '''
        Pass x, the featurised sample or logits of NN into tree.
        Via Soft decision rule.
        Returns a dictionary of all node and their path prob from their parent node.
        '''

        children = list(self.G.succ[parent].keys())
        # Dot prod of samples with children weights is Nx1 tensor
        prods = []
        for child in children:
            prod = torch.bmm(x.view(1, x.shape[0], x.shape[1]), self.weights[child].view(1, x.shape[1], 1)).squeeze(0)
            prods.append(prod)
        
        # prods is now Nx2 tensor column concatenated as [prod with child0, prod with child1]
        prods = torch.stack(prods, dim=-1)
        # softmax along every row (a samples) dot product. Result is N x 2, the children prob per sample
        probs = F.softmax(prods, dim = -1).view(x.shape[0],len(children))
    
        node_probs[:,children] = probs

        for child in children:
            if child>64:
                _ = self.traverse(x, node_probs, parent=child) # Not bug, function seeas node_probs in all calls
        return node_probs
    
    def node_path(self, Gr, start, root=75):
        '''
        Returns a list representing the path from root node to the start leaf.

        Gr = reverse view of original graph G.^
        start = the start leaf
        root = node label or index of root

        ^ It is much easier to traverse from leaf to root via the reverse view 
        of the DiGraph as there is only one succesor from every node (except root - with none) in reverse 
        direction and not two in original direction.
        '''

        if start == root: # Base case. Exitting out of recursion
            return []
        parent = list(Gr.succ[start].keys())[0]
        next_path = self.node_path(Gr, parent)
        next_path.append(start)

        return sorted(next_path, reverse=True)
    
    def get_leaf_prob(self, output):
        '''
        output is a batch of tensors from NN, the logits or featurised samples.
        return all the path probabilities of every leaf, for every tensor in batch.
        '''
        N, emb_dim = output.shape # Number of samples, and size of each sample vector

        # Tensor of all node probs, each row is the node prob of one sample in the batch
        node_probs = torch.zeros((N, 76), device=self.DEVICE) # output.shape[0] is the batch dimension
        node_probs[:,75] = 1.0
        
        # Tensor of all node probs, each row is the node prob of one sample in the batch,
        # columns are node index.
        node_probs = self.traverse(output, node_probs)

        leaf_path_probs = torch.zeros((N,65), device=self.DEVICE)# prefilled leaf path prob, N rows per samples
        
        for i in range(65):
            leaf_path_probs[:,i] = torch.prod(node_probs[:, self.leaf_paths[i]],dim=1)
        
        if self.draw_tree==True:
            nx.set_node_attributes(self.G, node_probs, name='node_prob') # THEN add the node prob as node atrributes 
            self.leaves_prob = leaf_path_probs.clone().detach()
            self.node_prob = node_probs.clone().detach()
        
        return leaf_path_probs, node_probs
    