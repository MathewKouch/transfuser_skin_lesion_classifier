import os # directory
from PIL import ImageFile,Image # reading image
ImageFile.LOAD_TRUNCATED_IMAGES = True # Avoiding truncation OSError 
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import math
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import pandas as pd
from augmentations import AllTransforms
from my_utilities2 import get_cost_matrix

def create_data(batch_size=72, H=256, RATIO=1.0, H_mac=256, H_mic=320, DEVICE='cpu', nw=13, TEST_CODE='False'):
    
    # List of the 65 molemap classes
    CLASSES = sorted(list(pd.read_csv('dataset/train.csv').label.value_counts().keys()))
    
    MOLEMAP = 'Molemap_Images_2020-02-11_d4'

    ROOT = f'../data/{MOLEMAP}'

    # Mole Map distance matrix
    D = torch.tensor(get_cost_matrix(CLASSES)).to(DEVICE)

    RATIO = 1.0
    
    W, W_mac, W_mic = int(H*RATIO), int(H_mac*RATIO), int(H_mic*RATIO)
    
    l2_stats = {'train':{'mac':[ [0.6792, 0.5768, 0.5310], [0.1346, 0.1311, 0.1357] ], 
                         'mic':[ [0.7478, 0.6091, 0.5826], [0.0942, 0.0948, 0.0997]] },
                'val': {'mac':[ [0.6794, 0.5771, 0.5317], [0.1341, 0.1308, 0.1354] ], 
                        'mic':[ [0.7476, 0.6097, 0.5832], [0.0937, 0.0946, 0.0997] ]}, 
                'test': {'mac': [ [0.6795, 0.5773, 0.5317], [0.1342, 0.1309, 0.1356] ], 
                        'mic':[ [0.7478, 0.6093, 0.5826], [0.0944, 0.0952, 0.1001] ]}, 
                'eval':{'mac':[ [0.67945, 0.5772, 0.5317], [0.13415, 0.13085, 0.1355] ], 
                 'mic':[[0.7477, 0.6095, 0.5829], [0.09405, 0.0949, 0.0999]]}, 
               }

    t_stat, v_stat, te_stat, eval_stat = l2_stats['train'], l2_stats['val'], l2_stats['test'], l2_stats['eval']
    
    # Bi modal concurrent augmentations
    train_tf = [AllTransforms(t_stat, angles=[0, 90, 180, -90],p = [1, 0.5, 0.5, 0.5], status='train')]
    val_tf = [AllTransforms(eval_stat, status='eval')]

    # Augmentations of original image to prepare from dataloader
    tf_mac = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(H_mac),
            transforms.CenterCrop((H_mac,W_mac))
    ])

    tf_mic = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(H_mic),
            transforms.CenterCrop((H_mic,W_mic))
    ])

    tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(H),
            transforms.CenterCrop((H,W))
    ])
    
    train_ds = ClassDataset_CSV(ROOT,'dataset/train.csv', CLASSES, transforms=[tf_mac, tf_mic] )
    val_ds = ClassDataset_CSV(ROOT,'dataset/val.csv', CLASSES, transforms=[tf_mac, tf_mic] )
    test_ds = ClassDataset_CSV(ROOT,'dataset/test.csv', CLASSES, transforms=[tf_mac, tf_mic] )
    samples_ds = ClassDataset_CSV(ROOT,'dataset/samples.csv', CLASSES, transforms=[tf, tf] )

    train_dl = DataLoader(train_ds,batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)
    test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)
    
    if TEST_CODE!='False': # Testing the code, using sample of 1 of every class (65 images only)
        train_dl = DataLoader(samples_ds,batch_size=1,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)
        val_dl = DataLoader(samples_ds,batch_size=1,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)
        test_dl = DataLoader(samples_ds,batch_size=1,shuffle=False, num_workers=nw, pin_memory=True, drop_last=False)
        print('TESTING CODE!!!!!!!')
        
    print(f'Train Dataset Size: {len(train_ds)}')
    print(f'Val Dataset Size: {len(val_ds)}')
    print(f'Test Dataset Size: {len(test_ds)}')
    print(f'Samples Dataset Size: {len(samples_ds)}')

    print(f'Train DataLoader Created with batch: {batch_size} Size: {len(train_dl)}')
    print(f'Val DataLoader Created with batch: {batch_size} Size: {len(val_dl)}')
    print(f'Test DataLoader Created with batch: {batch_size} Size: {len(test_dl)}')
    
    return train_dl, val_dl, test_dl, train_tf, val_tf, CLASSES, D
    
   
    
class ClassDataset(Dataset):
    '''
        !! This class is only used to pair clinical (MAC) and dermoscopic (MIC.POL) images during init.
        !! Use ClassDataset_CSV to generate a dataset from seperate train/val/test csv files.

        Args:
        ROOT_DIR = relative directory of the MoleMap folder
        CLASSES = alphabetically sorted list of classes in MoleMap to assign label to a class via index in this list. 
        level = int of 1 or 2. It is the pairing strategy to pair clinical and dermoscopic images by their img name.
                Image name strings are split by underscore '_'.

                level 1 means pairing by the first word in the split (the human id). 
                Do not use this as it can pair different skin lesion from the same human, 
                if they have more than one of the same kind of lesion. 

                level 2 pairs skin lesion by matching the first two words (the human and skin lesion id). 
                This is the default setting for pairing clincal and dermoscopic skin lesions, 
                and creates only one pair even if there are multiple images of the same lesion.
        transfrom = torchvision transform class to apply image augmentation

        new = boolean flag to collect ALL skin lesion images in molemap and pair them up. 
            This fills up the data attribute of the class that holds the entire paired dataset.
            
            !!!!!!!!! Pairing data will take up to 40 minutes

        '''
    def __init__(self,ROOT_DIR,CLASSES, level=2,transforms=None, new=False):
        
        self.root_dir = ROOT_DIR
        self.folders = os.listdir(ROOT_DIR) # List of all 74 folders
        self.classes = CLASSES# List of all 65 classes
        self.data = [] # Holds all the MAC-MIC pair images and their class labels as list
        self.transforms = transforms # contains a list of transforms function
        self.level = level # level = 1 for just human id, 2 for human id (dot) lesion id. Use level 2!
        
        self.cl = {} # Contains class name and count by all 65 classes
        self.clf = {} # Contains folder name of all 74 folders
        self.class_num = 0
        # Extracting one pair of MAC and MIC images per unique id (person) in every CLASS
        if new != False:
            print('Creating Dataset...')
            for cidx, CLASS in enumerate(self.folders):
                class_name = '_'.join(CLASS.split('_')[:3])
                print(f'{CLASS}: {cidx+1}/{len(self.folders)}') # Status of creating dataset
                MAC_list = os.listdir(os.path.join(ROOT_DIR,CLASS,'MAC'))
                MIC_list = os.listdir(os.path.join(ROOT_DIR,CLASS,'MIC.POL'))

    #             # ALphabetically sorted images
    #             MAC_list = sorted(MAC_list, reverse=False)
    #             MIC_list = sorted(MIC_list, reverse=False)

    #             # IDs in MAC and MIC
    #             MAC_ID = sorted(list(set([x.split('.')[0] for x in MAC_list])), reverse=False)
    #             MIC_ID = sorted(list(set([x.split('.')[0] for x in MIC_list])), reverse=False)

                # IDs in MAC and MIC
                MAC_ID = ['.'.join(x.split('.')[:self.level]) for x in MAC_list]
                MIC_ID = ['.'.join(x.split('.')[:self.level]) for x in MIC_list]

                # The IDs that exist in both MAC and MIC
                print(MAC_ID)
                ID = list(set(MAC_ID).intersection(set(MIC_ID)))
                #ID = list(set(MAC_ID).intersection(set(MIC_ID)))

                # DIctionary of key-value pairs of ID:[[MAC],[MIC]] full image file names
                Image_Dict={}

                for i in ID[:]: # Grabs instance of an ID of MAC and MIC.
                    for img_name in MAC_list:
                        if i=='.'.join(img_name.split('.')[:self.level]):
                            if i not in Image_Dict:
                                Image_Dict[i] = [class_name,CLASS,i, [img_name],[]]
                    for img_name in MIC_list:
                        if i=='.'.join(img_name.split('.')[:self.level]):
                            if i in Image_Dict:
                                Image_Dict[i][4].append(img_name)
                            break


                self.data += list(Image_Dict.values())
        
    def __len__(self):
        return len(self.data)        
        
    def get_csv(self):
        ''' Use after init and generating new data. Returns the paired skin lesion images as dataframe'''
        df = pd.DataFrame(data=self.data, columns=['label', 'class_folder', 'UID', 'mac', 'mic'])
        return df

    def sort_data(self):
        '''Sort data by frequency of sample per class'''
        if len(self.data) > 0:
            for i, c in enumerate(self.data):
                if c[0] not in self.cl:
                    self.cl[c[0]] = 1
                else:
                    self.cl[c[0]] += 1
                if c[1] not in self.clf:
                    self.clf[c[1]] = 1
                else:
                    self.clf[c[1]] += 1
            cl_sorted = OrderedDict(sorted(self.cl.items(), key = lambda k:(k[1],k[0])))
            self.cl = cl_sorted
            self.class_num = len(self.cl)
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Class labels for MAC-MIC image pairs, convert to integer by finding their index in class list   
        label = torch.tensor(self.classes.index(self.data[idx][0]))
        
        # Path to the images
        MAC_img_path = os.path.join(self.root_dir,self.data[idx][1],'MAC',self.data[idx][3][0])
        MIC_img_path = os.path.join(self.root_dir,self.data[idx][1],'MIC.POL',self.data[idx][4][0])


        # REad the imagea and return as PIL
        MAC_img = Image.open(MAC_img_path)
        MIC_img = Image.open(MIC_img_path)
        
        if os.path.isfile(MAC_img_path):

            # UID
            UID = self.data[idx][2]

            # Class folder name
            class_folder = self.data[idx][1]
            # Apply transformations, if any
            if self.transforms:
                MAC_img = self.transforms[0](MAC_img)
                MIC_img = self.transforms[1](MIC_img)

            return label, class_folder, UID, MAC_img, MIC_img
        else:
            pass
        
    
class ClassDataset_CSV(Dataset):
    '''
        Use to generate Dataset from csv file of dataset.
        Returns MAC image, MIC image, Class labels, and other metadata
        args:
        fp = file path to csv
        transforms = list of transforms [mac_transforms, mic_transforms]
        '''
    def __init__(self, ROOT_DIR, fp, CLASSES, transforms=None):
        
        self.root_dir = ROOT_DIR # dir to root
        self.transforms = transforms # contains a list of transforms function
        self.mean = torch.zeros(3) # RGB mean
        self.std = torch.zeros(3) # RGB std
        self.fp = fp # dir to csv file
        self.data = pd.read_csv(self.fp)
        # Making sure file paths for images only.
        self.data = self.data[self.data.mac_path.str.endswith('.jpg') + self.data.mic_path.str.endswith('.jpg')]
        print(f'CSV loaded with {len(self.data)} samples')
        self.classes = sorted(list(self.data.label.value_counts().keys())) # extract the classes from the csv
        
        self.folder = list(self.data.class_folder.values) # list of class folder for reading image
        self.mac_path = list(self.data.mac_path.values) # list of mac img path
        self.mic_path = list(self.data.mic_path.values) # list of mic img path
        self.uid = list(self.data.UID.values) # list of UID
        self.label = list(self.data.label.values)# list of labels
        
    def mean_std_calc(self): 
        # calculates data set mean and std upon init
        pass
    
    def __len__(self):
        return len(self.label)        
        
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        label = torch.tensor(self.classes.index(self.label[idx]))
        
        # Path to the images
        MAC_img_path = os.path.join(self.root_dir, self.folder[idx],'MAC', self.mac_path[idx])
        if '.NON.' in self.mic_path[idx]:
            MIC_img_path = os.path.join(self.root_dir, self.folder[idx], 'MIC.NON', self.mic_path[idx])
        else:
            MIC_img_path = os.path.join(self.root_dir, self.folder[idx], 'MIC.POL', self.mic_path[idx])

        # Read images    
        # MAC_img = torch.tensor(Image.open(MAC_img_path), dtype=torch.float32)
        # MIC_img = torch.tensor(Image.open(MIC_img_path), dtype=torch.float32)
        # MAC_img_path = os.path.join(self.root_dir,self.data[idx][1],'MAC',self.data[idx][3])
        # if '.NON.' in self.data[idx][4]:
        #     MIC_img_path = os.path.join(self.root_dir,self.data[idx][1],'MIC.NON',self.data[idx][4])
        # else:
        #     MIC_img_path = os.path.join(self.root_dir,self.data[idx][1],'MIC.POL',self.data[idx][4])


        # REad the imagea and return as PIL
  
        
        if os.path.isfile(MAC_img_path) and os.path.isfile(MIC_img_path):
            MAC_img = Image.open(MAC_img_path)
            MIC_img = Image.open(MIC_img_path)
            # UID
            UID = self.uid[idx]
            # Class folder name
            class_folder = self.folder[idx]

            # Apply transformations, if any
            if self.transforms:
                MAC_img = self.transforms[0](MAC_img)
                MIC_img = self.transforms[1](MIC_img)

            return label, class_folder, UID, MAC_img, MIC_img
        else:
            raise Exception(f'Not a file {MAC_img_path} \n {MIC_img_path}')
            pass
        
def train_val_test_split(data, train_split=0.7, val_split=0.1, test_split=0.2):
    '''
    Returns a dataframe with 
    data is pandas dataframe containing the entire dataset
    returns deep copied data frame with train, test, val column splits
    '''
    import random

    assert train_split + val_split + test_split == 1.0, 'Train/Val/Test split need to add up to 1.0'

    data = data.copy(deep=True)
    # Check if there is a train/val/test column in the df
    if 'train' not in data.columns:
        data.train=0
        data.val=0
        data.test=0
    
    N = len(data)
    
    # Number of train, val, test samples
    N_train = int(N*train_split)
    N_val = int(N*val_split)
    N_test = N - N_train - N_val

    # Generating shuffled sampling for train/val/test split

    N_set = set(np.arange(N)) # the set of all samples indices
    samples = list(np.arange(N))
    train_samples = set(random.sample(samples, N_train)) # the training sample indices
    eval_samples = N_set - train_samples # the remaining set of samples not in train
    val_samples = set(random.sample(list(eval_samples), N_val))
    test_samples = eval_samples - val_samples

    print(N_set == set.union(train_samples, val_samples, test_samples))

    train_samples = list(train_samples)
    val_samples = list(val_samples)
    test_samples = list(test_samples)
    
    # data.iloc[list(train_samples),5] = 1
    # data.iloc[list(val_samples),6] = 1
    # data.iloc[list(test_samples),7] = 1
    
    # Adding bool to Train/Val/Test columns in dataset df
    data.train[train_samples] = 1
    data.val[val_samples] = 1
    data.test[test_samples] = 1

    return data

def xpair(df, all_df):
    '''
        Create more pairing via cross pairing of already paired clinical (MAC) and dermoscopic (MIC).
        There are more images of the same skin lesion with same UID.
        Args:
        df (dataframe) this must contain only one pair with only 1 unique UID.
        all_df (dataframe) contains all the molemap images in dataframe format.
        return:
        xpair_data (dataframe) contains original pair and xpaired data of same skin lesion (by UID).
    '''
    xpair_data = []
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        class_folder = row['class_folder']
        label = row['label']

        uid = row['UID']
        mac_path = row['mac_path']
        mic_path = row['mic_path']

        mac_uids = all_df.loc[(all_df['UID']==uid) & (all_df['class_folder']==class_folder) & (all_df['mode']=='MAC')]
        mic_uids = all_df.loc[(all_df['UID']==uid) & (all_df['class_folder']==class_folder) & (all_df['mode']=='MIC.POL')]

        mac_image_names = mac_uids['image_name'].values
        mic_image_names = mic_uids['image_name'].values
        n_macs = len(mac_image_names)
        n_mics = len(mic_image_names)
        for mac in mac_image_names:
            for mic in mic_image_names:
                xpair_data.append([label, class_folder, uid, mac, mic, n_macs, n_mics])

    xpair_data = pd.DataFrame(data=xpair_data, columns=['label', 'class_folder', 'UID', 'mac_path', 'mic_path', 'n_macs', 'n_mics'])
    xpair_data = xpair_data.sort_values(by=['label', 'class_folder', 'UID', 'n_macs', 'n_mics'])
    xpair_data.reset_index(inplace=True)
    xpair_data.drop(labels=['index'], axis=1, inplace=True)


    return xpair_data