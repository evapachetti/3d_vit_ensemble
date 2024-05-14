# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:38:13 2021

@author: Germanese
"""

    

import torch
import pandas as pd
import pydicom
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import os

def crop(nparray,x0,y0,height,width):
    nparray_crop = nparray[y0-height//2:y0+height//2, x0-width//2:x0+width//2]
    return nparray_crop


class ProstateDataset(data.Dataset):
    
    
    def __init__(self, dataframe,aug_folder='Originale',size = 128):
        
        """
        csv_file (string): percorso al file csv con le annotazioni
        transform (callable, optional): trasformazione opzionale da applicare al campione
        """
        self.info = dataframe
        self.aug_folder = aug_folder
        self.dir_path = r"C:\Users\Germanese\Desktop\Eva\Lavoro\Lavoro MDPI\Dataset\ProstateX-2_128x128_CC\\"
        
    def __len__ (self):
            return len(self.info)
        
    def __getitem__(self, idx):
            if torch.is_tensor(idx): # se idx è un tensore
                idx = idx.tolist() # lo converto in una lista
            folder = self.info.iloc[idx,0]
            volume_path = self.dir_path + folder + "\\" + self.aug_folder
            slices = os.listdir(volume_path)
            z = self.info.iloc[idx,4] # coordinata z della fetta con il tumore
            if folder != 'ProstateX-0179':
                real_z = len(slices) - z + 1 # numero dell'immagine corrispondente a quella 
            else:
                real_z = z #il paziente 179 entra con la testa quindi la numerazione è corretta
            if real_z <= 9:
                slice_idx = slices.index('1-'+'0'+str(real_z)+'.png')
            else:
                slice_idx = slices.index('1-'+str(real_z)+'.png')
                
            five_slices = slices[slice_idx-2:slice_idx+3]
            
            volume = np.zeros((128,128,5,1)) #H,W,Z,C
            k = 0
            for s in five_slices:
                image_path = volume_path + "\\" + s
                v = np.array(Image.open(image_path))
                volume[:,:,k] = np.expand_dims(v,axis = 2)
                k+=1

            label = str(self.info.iloc[idx, 1])
            patient = self.info.iloc[idx,0]
            zone = self.info.iloc[idx,2]
            
            if label == 'LG':
                label = np.array(0)
            else:
                label = np.array(1)
        
                
            label = torch.from_numpy(label)
            
            return volume, label, patient, zone
        

class ToTensorDataset(torch.utils.data.Subset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        volume = self.dataset[idx][0]
        label = self.dataset[idx][1]
        patient = self.dataset[idx][2]
        zone = self.dataset[idx][3]
        
        if self.transform:
            volume = torch.from_numpy(volume).float()
            volume = volume.permute(3,0,1,2)
            #volume = self.transform(volume).float()
        
        return ((volume,label))

    def __len__(self):
        return len(self.dataset)



        