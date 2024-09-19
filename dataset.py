import os
import random
import pandas as pd
import numpy as np
import tifffile
import albumentations as A

import torch
from torch.utils.data import Dataset



class AODdataset(Dataset):
    def __init__(self, path, train, augment):
        self.data_root = path
        self.augment = augment
        train_answer_file = os.path.join(self.data_root,'train_answer.csv')
        self.answers = pd.read_csv(train_answer_file,header=None)
        self.answers = self.answers.sample(frac=1, random_state=0).reset_index(drop=True)
        if train == 'train':
            self.answers = self.answers.iloc[:int(0.8 * len(self.answers)),:]
        elif train == 'valid':
            self.answers = self.answers.iloc[int(0.8 * len(self.answers)):,:]
        self.answers.index = range(len(self.answers))
        self.transforms = A.Compose(
        [
            A.Affine(translate_percent={'x':(-0.1,0.1),'y':0}),
            A.Rotate(),
            A.RandomScale(),
        ],
    )
        self.resize = A.Resize(128,128)

                
    def __len__(self):
        return len(self.answers)
    
    def preprocess(self,image):
        image = image[:,:,[1,3,7,11]] # Aerosol Optical Depth Retrieval for Sentinel-2 Based onConvolutional Neural Network Method [Jie Jiang, Jiaxin Liu and Donglai Jiao]
        image = (image - image.min()) / (image.max() - image.min())
        return image 
    
    def __getitem__(self, idx):
        image, aod = self.answers[0][idx], self.answers[2][idx]

        # image = os.path.join(self.data_root,'train_images','train_images',image)
        image = os.path.join(self.data_root,'train_images','train_images',image)
        image = tifffile.imread(image).astype('float32')                 
        image = self.preprocess(image)
        if self.augment:
            image = self.transforms(image=image)['image']
        
        image = self.resize(image=image)['image']
        image = torch.from_numpy(image)
        image = torch.permute(image,(-1,0,1))


        aod = aod.astype('float32')
        aod = np.array([aod])
        aod = torch.from_numpy(aod)
        
        return image,aod




