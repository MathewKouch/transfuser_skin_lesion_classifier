#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ngoun (Mathew) Kouch
"""

import os # directory
from PIL import ImageFile,Image # reading image
ImageFile.LOAD_TRUNCATED_IMAGES = True # Avoiding truncation OSError 
import torch
import torch.nn as nn # For NN modules
import torch.nn.functional as F # For activations and utilies
import torchvision.models as models # For pretrained resnets
import timm # for pretrained ViT

class Transfuser(nn.Module):
    '''
    The transfuser model with transformer block 
    bridging between the two CNN layers.
    Args:
    - n_classes: (required) number of classes. Default None
    - n_layers: (requried) list of four ints, number of transformer encoder layers. Default [8, 8, 8, 8]
    - n_head: (requried) list of four ints, number of head per transfomer. Default [4, 4, 4, 4]
    - fexpansion: int, mulitplier for transformer block feed forward block after MHA. Default 4
    - emb_pdrop: float, prob of dropping embedding into each transformer encoder layer. a.k.a residual drop. Default 0.1
    - attn_pdrop: float, prob of dropping attention weights. Default 0.1
    - mlp_pdrop: float, prob of drop out of output of mlp block in Transformer Block. Default 0.1
    - classifier_pdrop: float, prob of mlp classifier on the flattened feature vector of CNN/Transformer backbone. Default 0.1
    - pretrained: bool, flag to use pretrained weights. Default True
    - cnns: list, of string to choose backbone CNN models. Default [resnet34. resnet34]
    - ratio: float, image width to hieght ratio. either 1.0 for square or 4/3 for rectangle. Default 1.0
    - fusion: string, how to fuse the final representation/feature vector from both CNN branch before classification. Default 'sum'
        'sum' performs element summation
        'cat' performs concetentation.
    - conv1s: bool, (epxerimental), to use 1 x 1 conv to increase/decrease feature map channels for transformer. Default False
    
    * Architecture inspired by https://ap229997.github.io/projects/transfuser/ of 
    'Multi-Modal Fusion Transformer for End-to-End Autonomous Driving' paper.
    '''
    def __init__(self,n_classes=None, n_layers=[8,8,8,8], n_head=[4,4,4,4], fexpansion=4, emb_pdrop=0.1, attn_pdrop=0.1, 
                 mlp_pdrop=0.1, classifier_pdrop=0.1, pretrained=True, cnns=['resnet34','resnet34'], ratio=1.0, fusion='sum', conv1d=False):
        super(Transfuser,self).__init__()
        
        assert n_classes is not None, 'You must input n_classes!'
        assert len(n_layers)==4, 'Insert a list of four ints for number of layers for the four transformers.'
        assert len(n_head)==4, 'Insert a list of four ints for number of heads for the four transformers.'
        assert fusion=='sum' or fusion=='cat', 'Fusion mode invalid or missing. It must be either sum or cat.'

        self.n_classes = n_classes
        # self.device = device
        self.fusion = fusion
        self.cnns = cnns

        # # Down samples feature maps as transformer input to reduce computation
        self.avgps = [8,8,8,8] # average pool sizes

        if ratio>=(4/3):
            self.ratio = 4/3
            self.Hpool = 9
            self.Wpool = 12

        if ratio==1.0: # DEFAULT
            self.ratio = ratio
            self.Hpool = 8
            self.Wpool = 8

        self.avgpool = nn.AdaptiveAvgPool2d((self.Hpool,self.Wpool))

        ########################### Experimental ####################################
        self.CONV1D = conv1d # FLag to perform 1x1 convd for transformer encoders
        ## To increase channels of 8x8 avgpooled feature maps to 512 for all transformers
        if self.CONV1D == True:
            # For MAC
            self.mac_up_1 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1, padding=0)
            self.mac_down_1 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, padding=0)

            self.mac_up_2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, padding=0)
            self.mac_down_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0)

            self.mac_up_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0)
            self.mac_down_3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)

            # For MIC
            self.mic_up_1 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1, padding=0)
            self.mic_down_1 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, padding=0)

            self.mic_up_2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, padding=0)
            self.mic_down_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0)

            self.mic_up_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0)
            self.mic_down_3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        
        ##########################################################################################

        # Transformer embdedding dims
        self.n_embs = [64*4,128*4,256*4,512*4] if cnns[0] in ['resnet50'] else [64,128,256,512]

        self.n_layers = n_layers
        self.n_heads = n_head

        self.MAC_encoder = MAC_CNN(mac_model=cnns[0], pretrained=pretrained)
        self.MIC_encoder = MIC_CNN(mic_model=cnns[1], pretrained=pretrained)

        self.ofm = 1 if cnns[0] in ['resnet18', 'resnet34'] else 4 # out feature multiplier.
        
        self.transformer1 = Transformer(n_layers=self.n_layers[0], n_seq=2*self.Hpool*self.Wpool, 
                                            n_emb=self.n_embs[0], n_head=self.n_heads[0], 
                                            fexpansion=fexpansion, emb_pdrop=emb_pdrop, attn_pdrop=attn_pdrop, 
                                            mlp_pdrop=mlp_pdrop)
        
        self.transformer2 = Transformer(n_layers=self.n_layers[1], n_seq=2*self.Hpool*self.Wpool, 
                                            n_emb=self.n_embs[1], n_head=self.n_heads[1], 
                                            fexpansion=fexpansion, emb_pdrop=emb_pdrop, attn_pdrop=attn_pdrop, 
                                            mlp_pdrop=mlp_pdrop)
        
        self.transformer3 = Transformer(n_layers=self.n_layers[2], n_seq=2*self.Hpool*self.Wpool, 
                                            n_emb=self.n_embs[2], n_head=self.n_heads[2], 
                                            fexpansion=fexpansion, emb_pdrop=emb_pdrop, attn_pdrop=attn_pdrop, 
                                            mlp_pdrop=mlp_pdrop)
        
        self.transformer4 = Transformer(n_layers=self.n_layers[3], n_seq=2*self.Hpool*self.Wpool, 
                                            n_emb=self.n_embs[3], n_head=self.n_heads[3], 
                                            fexpansion=fexpansion, emb_pdrop=emb_pdrop, attn_pdrop=attn_pdrop, 
                                            mlp_pdrop=mlp_pdrop)
        self.classifier_in_features = 512 if cnns[0] in ['resnet34','resnet18'] else 512*4 # resnet50 and densenet121 has 2048dim

        if self.fusion=='cat':
            self.classifier_in_features *= 2

        self.classifier_pdrop = classifier_pdrop
        self.classifier = nn.Sequential(
                            nn.Dropout(p=self.classifier_pdrop),
                            nn.Linear(self.classifier_in_features, self.n_classes),
                            )
                           
        print(f'Transfuser with {cnns} backbone, XL{self.n_layers}H{self.n_heads} made with {self.n_embs} embedding dim for transformers. Fusion via: {self.fusion} ')
        print(f'AvgPool ratio is {self.ratio:.2f} with height {self.Hpool} width {self.Wpool} total {self.Hpool*self.Wpool} embedding tokens for transformers.')

    def forward(self,MAC,MIC):
        BS, C, H, W = MAC.shape
        
        x_mac = self.MAC_encoder.resnet.conv1(MAC)
        x_mac = self.MAC_encoder.resnet.bn1(x_mac)
        x_mac = self.MAC_encoder.resnet.relu(x_mac)
        x_mac = self.MAC_encoder.resnet.maxpool(x_mac)
        
        x_mic = self.MIC_encoder.resnet.conv1(MIC)
        x_mic = self.MIC_encoder.resnet.bn1(x_mic)
        x_mic = self.MIC_encoder.resnet.relu(x_mic)
        x_mic = self.MIC_encoder.resnet.maxpool(x_mic)
        

        
        x_mac_1 = self.MAC_encoder.resnet.layer1(x_mac)
        x_mic_1 = self.MIC_encoder.resnet.layer1(x_mic)

        # Downsamples Layer 1 feature maps to (8x8)
        if self.CONV1D == True:
            x_mac_down1 = self.mac_up_1(self.avgpool(x_mac_1))
            x_mic_down1 = self.mic_up_1(self.avgpool(x_mic_1))
        else:
            x_mac_down1 = self.avgpool(x_mac_1)
            x_mic_down1 = self.avgpool(x_mic_1)
            
        x_mac_t1, x_mic_t1= self.transformer1(x_mac_down1, x_mic_down1)
        #print('x_mac.shape: ',x_mac.shape, 'x_mac_t1.shape: ',x_mac_t1.shape)
        uf = x_mac_1.shape[-1]/x_mac_t1.shape[-1] # upscale factor
        hmac,wmac = x_mac_1.shape[-2],x_mac_1.shape[-1]
        hmic,wmic = x_mic_1.shape[-2],x_mic_1.shape[-1]
        if self.CONV1D == True:
            x_mac_t1 = self.mac_down_1(x_mac_t1)
            x_mic_t1 = self.mic_down_1(x_mic_t1)

        #x_mac_up1 = F.interpolate(x_mac_t1, size=(h,w),=scale_factor=uf, mode='bilinear')
        #x_mic_up1 = F.interpolate(x_mic_t1, scale_factor=uf, mode='bilinear')
        hmac,wmac = x_mac_1.shape[-2],x_mac_1.shape[-1]
        hmic,wmic = x_mic_1.shape[-2],x_mic_1.shape[-1]
        x_mac_up1 = F.interpolate(x_mac_t1, size=(hmac,wmac), mode='bilinear')
        x_mic_up1 = F.interpolate(x_mic_t1, size=(hmic,wmic), mode='bilinear')
        
        #print(f'x_mac_up1.shape: {x_mac_up1.shape}')
        x_mac_s1 = x_mac_1 + x_mac_up1
        x_mic_s1 = x_mic_1 + x_mic_up1
        
        x_mac_2 = self.MAC_encoder.resnet.layer2(x_mac_s1)
        x_mic_2 = self.MIC_encoder.resnet.layer2(x_mic_s1)
        
        # Downsamples Layer 1 feature maps to (8x8) and conv1D to 512 channels for transformers 
        if self.CONV1D == True:
            x_mac_down2 = self.mac_up_2(self.avgpool(x_mac_2))
            x_mic_down2 = self.mic_up_2(self.avgpool(x_mic_2))
        else:
            x_mac_down2 = self.avgpool(x_mac_2)
            x_mic_down2 = self.avgpool(x_mic_2)
            
        x_mac_t2, x_mic_t2= self.transformer2(x_mac_down2, x_mic_down2)
       # print('x_mac.shape: ',x_mac.shape, 'x_mac_t1.shape: ',x_mac_t1.shape)
        uf = x_mac_2.shape[-1]/x_mac_t2.shape[-1] # upscale factor
        #print(f'uf2 = {uf}')
        
        if self.CONV1D == True:
            x_mac_t2 = self.mac_down_2(x_mac_t2)
            x_mic_t2 = self.mic_down_2(x_mic_t2)
            
        # x_mac_up2 = F.interpolate(x_mac_t2, scale_factor=uf, mode='bilinear')
        # x_mic_up2 = F.interpolate(x_mic_t2, scale_factor=uf, mode='bilinear')
        hmac, wmac = x_mac_2.shape[-2],x_mac_2.shape[-1]
        hmic, wmic = x_mic_2.shape[-2],x_mic_2.shape[-1]
        x_mac_up2 = F.interpolate(x_mac_t2, size=(hmac,wmac), mode='bilinear')
        x_mic_up2 = F.interpolate(x_mic_t2, size=(hmic,wmic), mode='bilinear')
        
        #print(f'x_mac_up1.shape: {x_mac_up1.shape}')
        x_mac_s2 = x_mac_2 + x_mac_up2
        x_mic_s2 = x_mic_2 + x_mic_up2
        
       # print(f'x_mac_s2 shape = {x_mac_s2.shape}')

        x_mac_3 = self.MAC_encoder.resnet.layer3(x_mac_s2)
        x_mic_3 = self.MIC_encoder.resnet.layer3(x_mic_s2)

        # Downsamples Layer 1 feature maps to (8x8) 
        if self.CONV1D == True:
            x_mac_down3 = self.mac_up_3(self.avgpool(x_mac_3))
            x_mic_down3 = self.mic_up_3(self.avgpool(x_mic_3))
        else:
            x_mac_down3 = self.avgpool(x_mac_3)
            x_mic_down3 = self.avgpool(x_mic_3)
            
        x_mac_t3, x_mic_t3 = self.transformer3(x_mac_down3, x_mic_down3)
        #print('x_mac.shape: ',x_mac.shape, 'x_mac_t1.shape: ',x_mac_t1.shape)
        if self.CONV1D == True:
            x_mac_t3 = self.mac_down_3(x_mac_t3)
            x_mic_t3 = self.mic_down_3(x_mic_t3)
            
        uf = x_mac_3.shape[-1]/x_mac_t3.shape[-1] # upscale factor

        # x_mac_up3 = F.interpolate(x_mac_t3, scale_factor=uf, mode='bilinear')
        # x_mic_up3 = F.interpolate(x_mic_t3, scale_factor=uf, mode='bilinear')
        hmac, wmac = x_mac_3.shape[-2],x_mac_3.shape[-1]
        hmic, wmic = x_mic_3.shape[-2],x_mic_3.shape[-1]
        x_mac_up3 = F.interpolate(x_mac_t3, size=(hmac,wmac), mode='bilinear')
        x_mic_up3 = F.interpolate(x_mic_t3, size=(hmic,wmic), mode='bilinear')
        
        #print(f'x_mac_up1.shape: {x_mac_up1.shape}')
        x_mac_s3 = x_mac_3 + x_mac_up3
        x_mic_s3 = x_mic_3 + x_mic_up3
        
        x_mac_4 = self.MAC_encoder.resnet.layer4(x_mac_s3)
        x_mic_4 = self.MIC_encoder.resnet.layer4(x_mic_s3)

        # Downsamples Layer 1 feature maps to (8x8) 
        x_mac_down4 = self.avgpool(x_mac_4)
        x_mic_down4 = self.avgpool(x_mic_4)
        
        x_mac_t4, x_mic_t4 = self.transformer4(x_mac_down4, x_mic_down4)
        #print('x_mac.shape: ',x_mac.shape, 'x_mac_t1.shape: ',x_mac_t1.shape)
        #uf = x_mac_4.shape[-1]/x_mac_t4.shape[-1] # upscale factor

        # x_mac_up4 = F.interpolate(x_mac_t4, scale_factor=uf, mode='bilinear')
        # x_mic_up4 = F.interpolate(x_mic_t4, scale_factor=uf, mode='bilinear')
        hmac, wmac = x_mac_4.shape[-2],x_mac_4.shape[-1]
        hmic, wmic = x_mic_4.shape[-2],x_mic_4.shape[-1]
        
        x_mac_up4 = F.interpolate(x_mac_t4, size=(hmac,wmac), mode='bilinear')
        x_mic_up4 = F.interpolate(x_mic_t4, size=(hmic,wmic), mode='bilinear')
        
        #print(f'x_mac_up1.shape: {x_mac_up1.shape}')
        x_mac_s4 = x_mac_4 + x_mac_up4
        x_mic_s4 = x_mic_4 + x_mic_up4
        
        x_mac_avgpool = self.MAC_encoder.resnet.avgpool(x_mac_s4)
        x_mic_avgpool = self.MIC_encoder.resnet.avgpool(x_mic_s4)
        
        x_mac_flatten = torch.flatten(x_mac_avgpool,start_dim=1)
        x_mac_view = x_mac_flatten.view(BS,1,-1) # Vectorise all mac feature maps 
        x_mic_flatten = torch.flatten(x_mic_avgpool,start_dim=1)
        x_mic_view = x_mic_flatten.view(BS,1,-1) # Vectorise all mic feature maps
        
        if self.fusion=='sum':
            fusion_cat = torch.cat([x_mac_view, x_mic_view], dim=1)
            fusion_sum = torch.sum(fusion_cat, dim=1)
            
            return self.classifier(fusion_sum)

        if self.fusion=='cat':
            fusion_cat = torch.cat([x_mac_view, x_mic_view], dim=-1)
            fusion_cat = torch.flatten(fusion_cat, start_dim=1)
            #print(f'fusion cat shape {fusion_cat.shape}')

            return self.classifier(fusion_cat)
        

class LateFusion(nn.Module):
    '''
    An ensemble of Resnet18 and Resnet34 with MLP classifier 
    '''
    def __init__(self, cnns=['resnet34','resnet34'], num_class=None, classifier_drop=0.1):
        super(LateFusion,self).__init__()
        assert num_class!=None, 'You must insert num_class for LateFusion!'

        self.cnns = cnns
        self.MAC_encoder = MAC_CNN(mac_model=cnns[0])
        self.MIC_encoder = MIC_CNN(mic_model=cnns[1])
        
        self.in_features = self.MAC_encoder.in_features
        self.num_class = num_class
        self.classifier_drop = classifier_drop
        
        self.classifier = nn.Sequential(
            nn.Dropout(self.classifier_drop),
            nn.Linear(self.in_features, self.num_class),
        )
    def forward(self,MAC,MIC):
           
        x_mac = self.MAC_encoder(MAC)
        x_mic = self.MIC_encoder(MIC)
        
        x_mac_flatten = torch.flatten(x_mac,start_dim=1)
        x_mic_flatten = torch.flatten(x_mic,start_dim=1)
        
        return self.classifier(x_mac_flatten+x_mic_flatten)
    
class ScoreFusion(nn.Module):
    '''
    An ensemble of Resnet18 and Resnet34 with MLP classifier 
    '''
    def __init__(self, cnns=['resnet34','resnet34'], num_class=None, classifier_drop=0.1):
        super(ScoreFusion,self).__init__()
        
        assert num_class!=None, 'You must insert num_class for ScoreFusion!'
        self.num_class = num_class

        self.cnns = cnns
        self.MAC_encoder = MAC_CNN(mac_model=cnns[0])
        self.MIC_encoder = MIC_CNN(mic_model=cnns[1])
        
        # Adding back the Linear layer removed and missing in MAC_CNN and MIC_CNN
        self.MAC_encoder.resnet.fc = nn.Linear(self.MAC_encoder.in_features, self.num_class)
        self.MIC_encoder.resnet.fc = nn.Linear(self.MIC_encoder.in_features, self.num_class)

        self.classifier_drop = classifier_drop
        self.dropout = nn.Dropout(self.classifier_drop)
        
    def forward(self,MAC,MIC):

        # logits output   
        x_mac = self.MAC_encoder(MAC)
        #print(x_mac)
        x_mic = self.MIC_encoder(MIC)
       #print(x_mic)
        out = x_mac+x_mic
        out = self.dropout(out)
        
        return out

################## Transfuser Building Blocks ####################################

class MAC_CNN(nn.Module):
    '''
    The Resnet34 image feature extractor for clinical images (MAC)
    '''
    def __init__(self, mac_model='resnet34', pretrained=True):
        super(MAC_CNN, self).__init__()
        
        if mac_model == 'resnet34':
            self.resnet = models.resnet34(
                pretrained=pretrained
                #weights='ResNet34_Weights.DEFAULT'
            )
        if mac_model == 'resnet18':
            self.resnet = models.resnet18(
                pretrained=pretrained
                #weights='ResNet18_Weights.DEFAULT'
            )
        if mac_model == 'resnet50':
            self.resnet = models.resnet50(
                pretrained=pretrained
                #weights='ResNet50_Weights.DEFAULT',
            )
        
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
            
    def forward(self,x):
        x = self.resnet(x)
        return x
        
        
class MIC_CNN(nn.Module):
    '''
    The Resnet34 image feature extractor for dermascopic images (MIC).
    '''
    def __init__(self, mic_model='resnet34', pretrained=True):
        super(MIC_CNN, self).__init__()

        if mic_model == 'resnet34':
            self.resnet = models.resnet34(
                pretrained=pretrained
                #weights='ResNet34_Weights.DEFAULT'
            )
        if mic_model == 'resnet18':
            self.resnet = models.resnet18(
                pretrained=pretrained
                #weights='ResNet18_Weights.DEFAULT'
            )
        if mic_model == 'resnet50':
            self.resnet = models.resnet50(
                pretrained=pretrained
                #weights='ResNet50_Weights.DEFAULT',
            )
            
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
    def forward(self,x):
        x = self.resnet(x)
        return x
    
class MHA(nn.Module):
    '''
    My implmentation of the attention block in transformer
    Args: 
    n_emb = dimension of embedding. which is number of CHANNELS in feature map
    n_head = number of attention head
    fexpansion = forward expansion of tensosr in feed forward block
    attn_pdrop = drop out percentage of weights and feature output
    res
    '''
    def __init__(self,n_emb, n_head=4, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        
        assert n_emb % n_head == 0 # Making sure even split of emb to each head

        self.n_emb = n_emb
        self.n_head = n_head
        
        # key, query, value projections for all heads
        self.K = nn.Linear(n_emb, n_emb, bias=True)
        self.Q = nn.Linear(n_emb, n_emb, bias=True)
        self.V = nn.Linear(n_emb, n_emb, bias=True)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.W0 = nn.Linear(n_emb, n_emb)

    def forward(self, x):
        
        # HEAD COMBINED - Works! 
        BS, T, C = x.size() # Batch x seq length x emb length
                
        # Project embedding to get quality, key, and value vectors.
        # Then split emb to each head along feature dim (channel)
        query = self.Q(x).view(BS, T, self.n_head, C//self.n_head) # B x T x head x k/n_head
        key = self.K(x).view(BS, T, self.n_head, C//self.n_head)# B x T x head x k/n_head
        value = self.V(x).view(BS, T, self.n_head, C//self.n_head) # B x T x head x k/n_head
        
        # Make every smaller sequence a layer for every head, then bundle batch and sequence dimensino for 3D torch.bmm()
        query = query.transpose(1,2).contiguous().view(BS*self.n_head, T, C//self.n_head)# B*n_head x [T x n_head_dimension]
        key = key.transpose(1,2).contiguous().view(BS*self.n_head, T, C//self.n_head)# B*n_head x [T x n_head_dimension]
        value = value.transpose(1,2).contiguous().view(BS*self.n_head, T, C//self.n_head)# B*n_head x [T x n_head_dimension]
        
        # Calculating self-attention
        #D_k = math.sqrt(key.shape[-1])**-1 # the inverted scalar sqrt of the length of smaller key vector = k/n_head
        key_T = key.transpose(1,2).contiguous() # key transpose for matrix dot product with quality. B*n_head x [n_head_dim x T]
        
        #print('shapes', quality.shape, key.shape, key_T.shape, D_k)
        
        # B*n_head x [T x n_head_dimension] dot B*n_head x [n_head_dimension x T] = B*n_head x [T x T]
        dot_product = torch.bmm(query, key_T) * ((C/self.n_head)**(-1/2)) # B*n_head x [T x T] Scaled dot product of a token to every tokens in sequence
        
        attention_weights = self.attn_drop(F.softmax(dot_product,2).contiguous()) # attention weights for every token wrt every token in sequence
        
        # (B*n_head, [T, T]) x (B*n_head, [T, n_head_dim]) = (B, n_head, [T, n_head_dim])
        attention = torch.bmm(attention_weights, value) # B*n_head x T x n_head_dim. Weighted sum of all value vectors by attention weights
        
        # (B*n_heads x T x n_head_dim) -> (B x n_heads x T x n_head_dim) -> (B x T x n_heads x n_head_dim) 
        attention = attention.view(BS, self.n_head, T, C//self.n_head).transpose(1,2).contiguous()
        
        # Concatenating all parts of attention along the heads together to get the full attention vector for every embedding in a sequence, for entire batch.
        attention = attention.view(BS,T,C) # (B, T, k=(n_heads*n_head_dim))
                
        # Projecting attention down to same dimension as embedding, then dropout
        # W0 reshapes attention vectors back to k.
        attention = self.resid_drop(self.W0(attention)) # (B,T,n_emb)
        
        return attention
        
    
class Transformer_Block(nn.Module):
    '''
    One transformer block that takes in an embedding (at first block)
    or previous transformer block representation vector.
    
    Sequence length = 2*8*8 = 128 = # pixels from downsamples feature maps of MAC and PIC
    
    To-DO: Positional embedding and Class embedding
     
        Args:
        n_emb = integer, size of embedding vectors - the blocks model dimension D = number of channel in feature map
        n_head = integer, number of attention blocks
        fexpansion = integer, factor size increase of MLP hidden layer forward expansion
        emb_pdrop = float, dropout prob at embedding input
        attn_pdrop = float, dropout prob of attention output
        mlp_pdrop = float, dropout prob of mlp output
    '''
    def __init__(self,n_emb, n_head, fexpansion, emb_pdrop, attn_pdrop, mlp_pdrop):
       
        super(Transformer_Block, self).__init__()
        # Make sure each head has same size input
        assert n_emb % n_head == 0
        
        self.n_emb = n_emb
        self.n_head = n_head
        self.fexpansion = fexpansion
        self.attn_pdrop = attn_pdrop
        
        # layer_norm of input before MHA
        self.x_ln = nn.LayerNorm(self.n_emb)
        # LN of attention after MHA and before MLP
        self.mlp_ln = nn.LayerNorm(n_emb)

        # Not needed, already in MultiheadAttention
        # self.W0 = nn.Linear(n_emb, n_emb) # Project concat of all Attention head to same input token shape Fout.shape = Fin.shape
        # self.attn_drop = nn.Dropout(attn_pdrop)

        # Applies dropout to attention before W0 projection AND AFTER, both using attn_pdrop!
        self.MHA = MHA(n_emb, n_head, attn_pdrop, attn_pdrop) #dim of input: Batch x seq x feature
        #self.MHA_dropout = nn.Dropout(self.attn_pdrop) # Already applied dropout in MHA
        
        
        self.MLP = nn.Sequential(
            nn.Linear(n_emb, fexpansion*n_emb),
            nn.ReLU(), # Should be inplace=True?????
            nn.Linear(fexpansion*n_emb,n_emb),
            nn.Dropout(mlp_pdrop)
        )

        
    def forward(self, x):
        '''
        x = torch.tensor as Batch x Seq x emb_dim(channel)
        '''
        x2 = self.x_ln(x)
        attention = self.MHA(x2) # attention droup out already applied after MHA. Ignore attention weights output
        x3 = attention + x
        x4 = self.mlp_ln(x3)
        
        x4 = self.MLP(x4) # (1,128,C) #print(f'First Transformer Block out with out shape: {out.shape}')
        
        return x3 + x4
        
class Transformer(nn.Module):
    '''
    The transformer between intermediate CNN layers to obtain attention of feature maps.
    
    To-DO: Positional embedding and Class embedding
     
        Args:
        n_layers = integer, number of transformer blocks/layers
        n_emb = integer, size of embedding vectors - the blocks model dimension D.
        n_head = integer, number of attention blocks
        fexpansion = integer, factor size increase of MLP hidden layer forward expansion
        emb_pdrop = float, dropout prob at embedding input
        attn_pdrop = float, dropout prob of attention output
        mlp_pdrop = float, dropout prob of mlp output
        
    '''
    def __init__(self, n_layers,n_seq, n_emb, n_head, fexpansion, emb_pdrop, attn_pdrop, mlp_pdrop):
        super(Transformer, self).__init__()
        
        self.n_layers = n_layers
        self.n_emb = n_emb
        
        assert n_emb % n_head == 0
        
        # Positional embedding as parameters which will be learnt
        self.positional_emb = nn.Parameter(torch.zeros(1,n_seq,n_emb))
        
        # embedding dropout applies only at first transformer block
        self.emb_drop = nn.Dropout(emb_pdrop)
        
        # List of Transformer Blocks
        blocks = [Transformer_Block(n_emb, n_head, fexpansion, emb_pdrop, attn_pdrop, mlp_pdrop) for layer in range(n_layers)]
        
        # Unpack Transformer Blocks as sequential
        self.blocks = nn.Sequential(*blocks)
        
        # Last layernorm of output of transformer
        self.decoder_ln = nn.LayerNorm(self.n_emb)

        self.apply(self._init_weights)
        print('Linear and LayerNorm weights inits')    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def forward(self, MAC,MIC):
        '''
        MAC = torch.tensor, feature map of shape Batch x channel x 8 x 8
        MIC = torch.tensor, feature map of shape Batch x channel x 8 x 8
        '''        
                
        # Shape of input feature maps
        BS, C, H, W = MAC.shape # Batch x channel x height x width
         
        # Reshaping feature maps to B x T x -1 x h x w. 
        # T is token dimension, 1 per activation/feature map
        # Channel is -1 because it increases after every conv layer.
        MAC = MAC.view(BS,1,C,H,W)
        MIC = MIC.view(BS,1,C,H,W)
        
        # Creating input sequence by concatenating the two feature maps along token dimension
        SEQ = torch.cat([MAC,MIC],1) # B x T x -1 x H X W .
        # Reordering dimension so that channel is last
        SEQ = SEQ.permute(0,1,3,4,2) # B x T x H x W x C
        # Flattening along T,H,W to get required input sequence dimension of 1 x B x (T*H*W) x C
        
        # Then add positional embedding with BROADCASTING of positional_emb along batch dimensionsss
        SEQ = torch.flatten(SEQ,start_dim=1,end_dim=3) + self.positional_emb
        
        # Dropout nput sequences
        SEQ = self.emb_drop(SEQ)
        # Transformer blocks
        out = self.blocks(SEQ) # Batch x Seq=2*8*8 x emb_dim 
        out = self.decoder_ln(out) # output as batch x seq x channel

        out = out.view(BS, 2, H, W, C) # Regroup by token
        out = out.permute(0,1,4,2,3).contiguous() # swap dimensions back original: to BS x Token x Channel x H x W

        MAC_out = out[:, :1, :, :, :].contiguous().view(BS, C, H, W) # Gather the first half of tokens for MAC and reshape back as input MAC feature map
        MIC_out = out[:, 1:, :, :, :].contiguous().view(BS, C, H, W) # Gather latter half of tokens for MIC and reshape as input MIC feature map
        
        return MAC_out, MIC_out

################## End Transfuser Building Block ####################################

class TransfuserVIT(nn.Module):
    '''
    The transfuser model with pretrained VIT block 
    bridging between the two CNN layers.
    Args:
    - n_classes: (required) number of classes. Default None
    - n_layers: (requried) list of four ints, number of transformer encoder layers. Default [8, 8, 8, 8]
    - n_head: (requried) list of four ints, number of head per transfomer. Default [4, 4, 4, 4]
    - fexpansion: int, mulitplier for transformer block feed forward block after MHA. Default 4
    - emb_pdrop: float, prob of dropping embedding into each transformer encoder layer. a.k.a residual drop. Default 0.1
    - attn_pdrop: float, prob of dropping attention weights. Default 0.1
    - mlp_pdrop: float, prob of drop out of output of mlp block in Transformer Block. Default 0.1
    - classifier_pdrop: float, prob of mlp classifier on the flattened feature vector of CNN/Transformer backbone. Default 0.1
    - pretrained: bool, flag to use pretrained weights. Default True
    - cnns: list, of string to choose backbone CNN models. Default [resnet34. resnet34]
    - ratio: float, image width to hieght ratio. either 1.0 for square or 4/3 for rectangle. Default 1.0
    - fusion: string, how to fuse the final representation/feature vector from both CNN branch before classification. Default 'sum'
        'sum' performs element summation
        'cat' performs concetentation.
    - conv1s: bool, (epxerimental), to use 1 x 1 conv to increase/decrease feature map channels for transformer. Default False
    '''
    def __init__(self,n_classes=None, vit='vit_tiny_patch16_224', n_layers=12, pretrained=True, cnns=['resnet34','resnet34'], ratio=1.0, fusion='sum', classifier_pdrop=0.1):
        super(TransfuserVIT,self).__init__()
        
        assert n_classes is not None, 'You must input n_classes!'
        assert fusion=='sum' or fusion=='cat', 'Fusion mode invalid or missing. It must be either sum or cat.'
        
        self.n_layers = n_layers
        self.n_classes = n_classes
        # self.device = device
        self.fusion = fusion
        self.cnns = cnns

        # # Down samples feature maps as transformer input to reduce computation
        self.avgps = [8,8,8,8] # average pool sizes

        if ratio>=(4/3):
            self.ratio = 4/3
            self.Hpool = 9
            self.Wpool = 12

        if ratio==1.0: # DEFAULT
            self.ratio = ratio
            self.Hpool = 8
            self.Wpool = 8

        self.avgpool = nn.AdaptiveAvgPool2d((self.Hpool,self.Wpool))

        # Transformer embdedding dims
        self.n_embs = [64*4,128*4,256*4,512*4] if cnns[0] in ['resnet50'] else [64,128,256,512]



        self.MAC_encoder = MAC_CNN(mac_model=cnns[0], pretrained=pretrained)
        self.MIC_encoder = MIC_CNN(mic_model=cnns[1], pretrained=pretrained)

        self.ofm = 1 if cnns[0] in ['resnet18', 'resnet34'] else 4 # out feature multiplier.
        
        self.transformer1 = VIT(vit, n_layers=12, in_chans=64, out_features=64, pretrained=True, emb_pdrop=0.1)
        
        self.transformer2 = VIT(vit, n_layers=12, in_chans=128, out_features=128, pretrained=True, emb_pdrop=0.1)
        
        self.transformer3 = VIT(vit, n_layers=12, in_chans=256, out_features=256, pretrained=True, emb_pdrop=0.1)
        
        self.transformer4 = VIT(vit, n_layers=12, in_chans=512, out_features=512, pretrained=True, emb_pdrop=0.1)

        self.classifier_in_features = 512 if cnns[0] in ['resnet34','resnet18'] else 512*4 # resnet50 and densenet121 has 2048dim

        if self.fusion=='cat':
            self.classifier_in_features *= 2

        self.classifier_pdrop = classifier_pdrop
        self.classifier = nn.Sequential(
                            nn.Dropout(p=self.classifier_pdrop),
                            nn.Linear(self.classifier_in_features, self.n_classes),
                            )
                           
        print(f'Transfuser with {cnns} backbone, T_VIT{self.n_layers} made with {self.n_embs} embedding dim for transformers. Fusion via: {self.fusion} ')
        print(f'AvgPool ratio is {self.ratio:.2f} with height {self.Hpool} width {self.Wpool} total {self.Hpool*self.Wpool} embedding tokens for transformers.')

    def forward(self,MAC,MIC):
        BS, C, H, W = MAC.shape
        
        x_mac = self.MAC_encoder.resnet.conv1(MAC)
        x_mac = self.MAC_encoder.resnet.bn1(x_mac)
        x_mac = self.MAC_encoder.resnet.relu(x_mac)
        x_mac = self.MAC_encoder.resnet.maxpool(x_mac)
        
        x_mic = self.MIC_encoder.resnet.conv1(MIC)
        x_mic = self.MIC_encoder.resnet.bn1(x_mic)
        x_mic = self.MIC_encoder.resnet.relu(x_mic)
        x_mic = self.MIC_encoder.resnet.maxpool(x_mic)
        
        x_mac_1 = self.MAC_encoder.resnet.layer1(x_mac)
        x_mic_1 = self.MIC_encoder.resnet.layer1(x_mic)

        # Downsamples Layer 1 feature maps to (8x8)

        x_mac_down1 = self.avgpool(x_mac_1)
        x_mic_down1 = self.avgpool(x_mic_1)
        
        x_mac_t1, x_mic_t1= self.transformer1(x_mac_down1, x_mic_down1)

        hmac,wmac = x_mac_1.shape[-2],x_mac_1.shape[-1]
        hmic,wmic = x_mic_1.shape[-2],x_mic_1.shape[-1]

        x_mac_up1 = F.interpolate(x_mac_t1, size=(hmac,wmac), mode='bilinear')
        x_mic_up1 = F.interpolate(x_mic_t1, size=(hmic,wmic), mode='bilinear')
        
        x_mac_s1 = x_mac_1 + x_mac_up1
        x_mic_s1 = x_mic_1 + x_mic_up1
        
        x_mac_2 = self.MAC_encoder.resnet.layer2(x_mac_s1)
        x_mic_2 = self.MIC_encoder.resnet.layer2(x_mic_s1)
        
        # Downsamples Layer 1 feature maps to (8x8) and conv1D to 512 channels for transformers 

        x_mac_down2 = self.avgpool(x_mac_2)
        x_mic_down2 = self.avgpool(x_mic_2)
            
        x_mac_t2, x_mic_t2= self.transformer2(x_mac_down2, x_mic_down2)

        hmac, wmac = x_mac_2.shape[-2],x_mac_2.shape[-1]
        hmic, wmic = x_mic_2.shape[-2],x_mic_2.shape[-1]
        x_mac_up2 = F.interpolate(x_mac_t2, size=(hmac,wmac), mode='bilinear')
        x_mic_up2 = F.interpolate(x_mic_t2, size=(hmic,wmic), mode='bilinear')
        
        x_mac_s2 = x_mac_2 + x_mac_up2
        x_mic_s2 = x_mic_2 + x_mic_up2
        
        x_mac_3 = self.MAC_encoder.resnet.layer3(x_mac_s2)
        x_mic_3 = self.MIC_encoder.resnet.layer3(x_mic_s2)

    
        x_mac_down3 = self.avgpool(x_mac_3)
        x_mic_down3 = self.avgpool(x_mic_3)
            
        x_mac_t3, x_mic_t3 = self.transformer3(x_mac_down3, x_mic_down3)       

        hmac, wmac = x_mac_3.shape[-2],x_mac_3.shape[-1]
        hmic, wmic = x_mic_3.shape[-2],x_mic_3.shape[-1]
        x_mac_up3 = F.interpolate(x_mac_t3, size=(hmac,wmac), mode='bilinear')
        x_mic_up3 = F.interpolate(x_mic_t3, size=(hmic,wmic), mode='bilinear')
        
        x_mac_s3 = x_mac_3 + x_mac_up3
        x_mic_s3 = x_mic_3 + x_mic_up3
        
        x_mac_4 = self.MAC_encoder.resnet.layer4(x_mac_s3)
        x_mic_4 = self.MIC_encoder.resnet.layer4(x_mic_s3)

        # Downsamples Layer 1 feature maps to (8x8) 
        x_mac_down4 = self.avgpool(x_mac_4)
        x_mic_down4 = self.avgpool(x_mic_4)
        
        x_mac_t4, x_mic_t4 = self.transformer4(x_mac_down4, x_mic_down4)

        hmac, wmac = x_mac_4.shape[-2],x_mac_4.shape[-1]
        hmic, wmic = x_mic_4.shape[-2],x_mic_4.shape[-1]
        
        x_mac_up4 = F.interpolate(x_mac_t4, size=(hmac,wmac), mode='bilinear')
        x_mic_up4 = F.interpolate(x_mic_t4, size=(hmic,wmic), mode='bilinear')
        
        #print(f'x_mac_up1.shape: {x_mac_up1.shape}')
        x_mac_s4 = x_mac_4 + x_mac_up4
        x_mic_s4 = x_mic_4 + x_mic_up4
        
        x_mac_avgpool = self.MAC_encoder.resnet.avgpool(x_mac_s4)
        x_mic_avgpool = self.MIC_encoder.resnet.avgpool(x_mic_s4)
        
        x_mac_flatten = torch.flatten(x_mac_avgpool,start_dim=1)
        x_mac_view = x_mac_flatten.view(BS,1,-1) # Vectorise all mac feature maps 
        x_mic_flatten = torch.flatten(x_mic_avgpool,start_dim=1)
        x_mic_view = x_mic_flatten.view(BS,1,-1) # Vectorise all mic feature maps
        
        if self.fusion=='sum':
            fusion_cat = torch.cat([x_mac_view, x_mic_view], dim=1)
            fusion_sum = torch.sum(fusion_cat, dim=1)
            
            return self.classifier(fusion_sum)

        if self.fusion=='cat':
            fusion_cat = torch.cat([x_mac_view, x_mic_view], dim=-1)
            fusion_cat = torch.flatten(fusion_cat, start_dim=1)
            #print(f'fusion cat shape {fusion_cat.shape}')

            return self.classifier(fusion_cat)
    

class VIT(nn.Module):
    def __init__(self, vit='vit_tiny_patch16_224', n_layers=12, in_chans=64, out_features=128, pretrained=True, emb_pdrop=0.1):
        super(VIT,self).__init__()

        self.model = self.create_vit(vit, n_layers,in_chans, out_features, pretrained)

        # Positional embedding as parameters which will be learnt
        self.positional_emb = nn.Parameter(torch.zeros(1, 128, in_chans)) # 1, token dimension (2*8*8=number of pixels of one feature maps), number of feature maps
        self.n_layers = n_layers
        # embedding dropout applies only at first transformer block
        self.emb_drop = nn.Dropout(emb_pdrop)

    def create_vit(self, name, layers, in_chans, out_features=128, pretrained=True):
        model = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)
        model.train()
        for n,m in model.blocks.named_parameters():
            block_n = int(n.split('.')[0])
            if block_n>layers:
                # print(n)
                model.blocks[block_n] = nn.Identity()
        model.patch_embed = nn.Linear(in_chans, model.head.in_features)        
        model.head = nn.Linear(model.head.in_features, out_features)

        return model

    def forward(self, MAC, MIC):
    # Shape of input feature maps
        BS, C, H, W = MAC.shape # Batch x channel x height x width
         
        # Reshaping feature maps to B x T x -1 x h x w. 
        # T is token dimension, 1 per activation/feature map
        # Channel is -1 because it increases after every conv layer.
        MAC = MAC.view(BS,1,C,H,W)
        MIC = MIC.view(BS,1,C,H,W)
        
        # Creating input sequence by concatenating the two feature maps along token dimension
        SEQ = torch.cat([MAC,MIC],1) # B x T x -1 x H X W .

        # Reordering dimension so that channel is last
        SEQ = SEQ.permute(0,1,3,4,2) # B x T x H x W x C

        # Flattening along T,H,W to get required input sequence dimension of 1 x B x (T*H*W) x C
        # Then add positional embedding with BROADCASTING of positional_emb along batch dimensionsss
        SEQ = torch.flatten(SEQ,start_dim=1,end_dim=3) + self.positional_emb
        
        # Dropout nput sequences
        SEQ = self.emb_drop(SEQ) # 3 dim: B x (2*H*W) x Channel (number of fmaps)
        #print(f'SEQ shape {SEQ.shape}')
        # Inputing concat features as batch of token sequences into ViT
        out = self.model.patch_embed(SEQ)
        #print(f'patchemb out {out.shape}')
        out = self.model.blocks(out)
        #print(f'vit blocks out {out.shape}')
        out = self.model.norm(out)
        #print(f'last norm out {out.shape}')
        out = self.model.head(out)
        #print(f'vit out {out.shape}')
        
        out = out.view(BS, 2, H, W, C) # Regroup by token
        out = out.permute(0,1,4,2,3).contiguous() # swap dimensions back original: to BS x Token x Channel x H x W

        MAC_out = out[:, :1, :, :, :].contiguous().view(BS, C, H, W) # Gather the first half of tokens for MAC and reshape back as input MAC feature map
        MIC_out = out[:, 1:, :, :, :].contiguous().view(BS, C, H, W) # Gather latter half of tokens for MIC and reshape as input MIC feature map
        
        return MAC_out, MIC_out
