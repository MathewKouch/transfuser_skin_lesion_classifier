
import os
import numpy as np
import pickle
from PIL import ImageFile,Image # reading image
ImageFile.LOAD_TRUNCATED_IMAGES = True # Avoiding truncation OSError 

import torch
import torchvision
import torchvision.transforms as transforms

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


class AllTransforms:
    """
    Applies same flip, zoom in, rotation, and normalise to both images to preserving skin lesion orientation.
    """

    def __init__(self, stats, angles=[0,-90,90,180],zoom=[0.99,0.95,0.9,0.85,0.80],
                 p=[1.0,0.5,0.5,0.5], status='train', normalisation=True):
        
        self.status = status
        self.normalisation = normalisation
        
        zoom_transform = MyZoomTransform(zoom=zoom)
        rotation_transform = MyRotationTransform(angles=angles) # rotate 
        hflip_transform = MyHFlipTransform()
        vflip_transform = MyVFlipTransform()
        
        self.random_zoom = RandomApply(zoom_transform, p=p[1])  
        self.random_rotate = RandomApply(rotation_transform, p=p[0])
        self.random_hflip = RandomApply(hflip_transform, p=p[2])  
        self.random_vflip = RandomApply(vflip_transform, p=p[3])
                                            
        self.mac_norm = transforms.Normalize(stats['mac'][0], stats['mac'][1])
        self.mic_norm = transforms.Normalize(stats['mic'][0], stats['mic'][1])

    def __call__(self, mac, mic, labels):
        
        if self.status=='train':
            mac, mic = self.random_zoom(mac, mic)            
            mac, mic = self.random_rotate(mac, mic)
            mac, mic = self.random_hflip(mac, mic)
            mac, mic = self.random_vflip(mac, mic)
        
        if self.normalisation:                                    
            mac = self.mac_norm(mac)
            mic = self.mic_norm(mic)

        return mac, mic
                                        

class URTransforms:
    """Transforms only samples from under represented classes UR."""

    def __init__(self, stats, angles=[-180,180],zoom=[0.99,0.95,0.9,0.85,0.80],
                 p=[0.5,0.5,0.5,0.5],height=256, ratio=4/3, status='train'):
        ### Output Height and Width
        self.height = height
        self.width = int(height*ratio)
        self.width = self.width if self.width%2==0 else self.width + 1
        self.ratio = ratio
        self.status = status
        
        # Over represented represented Classses
        self.OR = [
            4.0,8.0,10.0,22.0,37.0,40.0,42.0
        ] 
        # Zoom into image by cropping out and saving zoom percent of image, at desired ratio

        self.down_size = transforms.Resize((self.height, self.width)) # Resize the zoomed image to desired H and W, matching ratio
        rotation_transform = MyRotationTransform(angles=angles) # rotate 
        hflip_transform = MyHFlipTransform()
        vflip_transform = MyVFlipTransform()
        
        if self.status=='train':
            zoom_transform = MyZoomTransform(zoom=zoom, ratio=self.ratio)
            self.random_zoom = RandomApply(zoom_transform, p=p[1])  
            self.random_rotate = RandomApply(rotation_transform,p=p[0])
            self.random_hflip = RandomApply(hflip_transform, p=p[2])  
            self.random_vflip = RandomApply(vflip_transform, p=p[3])
            self.random_affine = torchvision.transforms.RandomAffine(degrees=0, translate=(0.01, 0.01))
                                                         
        else:
            zoom_transform = MyZoomTransform(zoom=[1.0], ratio=self.ratio)
            self.random_zoom = RandomApply(zoom_transform, p=1.0)  
            
        self.mac_norm = transforms.Normalize(stats['mac'][0], stats['mac'][1])
        self.mic_norm = transforms.Normalize(stats['mic'][0], stats['mic'][1])

    def __call__(self, mac, mic, labels):
        
        if self.status=='train':
            
            macs = torch.zeros((mac.shape[0], mac.shape[1], self.height, self.width))
            mics = torch.zeros((mic.shape[0], mic.shape[1], self.height, self.width))
            
            # Just downside Over Represented (OR) samples
            
            OR_IDX = [idx for idx,l in enumerate(labels) if l in self.OR ]
            UR_IDX = [idx for idx,l in enumerate(labels) if l not in self.OR ]
            
            macs[OR_IDX] = self.down_size(mac[OR_IDX])
            mics[OR_IDX] = self.down_size(mic[OR_IDX])
                
            # Augmentations to Under Represented (UR) samples    
            mac_ = mac[UR_IDX]
            mic_ = mic[UR_IDX]
            
            mac_, mic_ = self.random_zoom(mac_, mic_)
            mac_ = self.down_size(mac_)
            mic_ = self.down_size(mic_)
            mac_, mic_ = self.random_rotate(mac_, mic_)
            mac_, mic_ = self.random_hflip(mac_, mic_)
            mac_, mic_ = self.random_vflip(mac_, mic_)
                    
            # Combine UR samples with OR samples        
            macs[UR_IDX] = mac_
            mics[UR_IDX] = mic_

            # Normalise all samples, and return
            return self.mac_norm(macs), self.mic_norm(mics)
        
        else: # just downsize all images if not in training
        
            mac = self.down_size(mac)
            mic = self.down_size(mic)
        
            mac = self.mac_norm(mac)
            mic = self.mic_norm(mic)
             
            return mac, mic
    
class MyRotationTransform:
    """Rotate both images at the same angle, chosen uniformly from a list of angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, mac, mic):
        angle = random.choice(self.angles)
        
        mac = transforms.functional.rotate(mac, angle)
        mic = transforms.functional.rotate(mic, angle)
        return mac, mic
   

class MyZoomTransform:
    """Zoom into image by cropping out smaller portion (z<1), then resize back to original size. 
    """

    def __init__(self,zoom=[1.0]):
        self.zoom = zoom
        
    def __call__(self,mac,mic):
        z = random.choice(self.zoom) 
        
        Hmac = mac.shape[-2]
        Wmac = mac.shape[-1]
        
        Hmic = mic.shape[-2]
        Wmic = mic.shape[-1]
        
        mac_resize = transforms.Resize((Hmac, Wmac))
        mic_resize = transforms.Resize((Hmic, Wmic))
        
        mac = mac_resize(transforms.functional.center_crop(mac, (int(Hmac*z),int(Wmac*z))))
        mic = mic_resize(transforms.functional.center_crop(mic, (int(Hmic*z),int(Wmic*z))))
        
        return mac, mic
    
class MyHFlipTransform:
    """Horizontally flip both images."""

    def __init__(self):
        pass
        
    def __call__(self,mac,mic):
        mac = transforms.functional.hflip(mac)
        mic = transforms.functional.hflip(mic)
        return mac, mic

class MyVFlipTransform:
    """Vertically flip both images."""

    def __init__(self):
        pass
        
    def __call__(self,mac,mic):
        mac = transforms.functional.vflip(mac)
        mic = transforms.functional.vflip(mic)
        return mac, mic
class RandomApply:
    ''' Applies the input transform specified by a given prob '''
    def __init__(self, tf, p):
        self.tf = tf
        self.prob = p # prob
    def __call__(self, mac, mic):
        if random.uniform(0.0,1.0)<self.prob:
            mac, mic = self.tf(mac, mic)
        return mac, mic

class MyRotationTransform_:
    """Rotate a single image from a list of angles."""

    def __init__(self, angles):
        
        self.angles = angles

    def __call__(self, mac):
        
        angle = random.choice(self.angles)
        
        mac = transforms.functional.rotate(mac, angle)
        
        return mac
   

class MyZoomTransform_:
    """Zoom a single image by cropping out smaller portion (z<1), then resize back to original size. 
    """

    def __init__(self,zoom=[1.0]):
        
        self.zoom = zoom
        
    def __call__(self,mac):
        z = random.choice(self.zoom) 
        
        Hmac = mac.shape[-2]
        Wmac = mac.shape[-1]
                
        mac_resize = transforms.Resize((Hmac, Wmac))
        
        mac = mac_resize(transforms.functional.center_crop(mac, (int(Hmac*z),int(Wmac*z))))
        
        return mac