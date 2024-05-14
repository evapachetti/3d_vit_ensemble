# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:18:36 2021

@author: Germanese
"""

import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from create_volume_dataset_PX2 import ProstateDataset, ToTensorDataset
import numpy as np
from stratified_group_data_splitting import StratifiedGroupKFold
import os


logger = logging.getLogger(__name__)


def get_loader(args,cv):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    if args.dataset == "prostateX":
        
        path = r"C:\Users\Germanese\Desktop\Eva\Lavoro\Lavoro MDPI\Csv splitting\ProstateX-2\CV 3D"
        csv_file_train = os.path.join(path,"png_PROSTATEx2_T2_volume_Training_CV"+str(cv+1)+".csv")
        csv_file_val = os.path.join(path,"png_PROSTATEx2_T2_volume_Validation_CV"+str(cv+1)+".csv")
    
        
        trainset = list(ProstateDataset(csv_file_train))
        validset = list(ProstateDataset(csv_file_val))
        
        volumes_train = [i[0] for i in list(trainset)]
        mean = np.mean(volumes_train)
        std = np.std(volumes_train)
        
        #AUGMENTATION
        aug_suffix = ['Rotazione', 'Flip verticale', 'Flip orizzontale']
        trainsets_aug = []
        
        for h in aug_suffix: # Per ogni tecnica di data augmentation
            trainsets_aug.append(ProstateDataset(csv_file_train,aug_folder = h)) # dataset con solo immagini alterate
        
        #SOLO PER PROSTATEX
        hg_positions = [i for i in range(len(trainset)) if trainset[i][1].item() == 1] # trovo tutte le posizioni del training set dove label = HG
        positions = [hg_positions[i] for i in range(0,len(hg_positions),3)] #fisso le immagini che vengono aumentate (dipendono solo da come è stato suddiviso il dataset) (in tutto 8 posizioni * 3 = 24 immagini aggiunte alle 24 di partenza)
        #Le LG sono 48, le HG con augmentation sono 24+24 = 48
        
        for trainset_aug in trainsets_aug:
            trainset_aug = list(trainset_aug)
            [trainset.append(trainset_aug[position]) for position in positions] # appendo al trainset principale le immagini HG alterate scelte
                
        for i in range(len(trainset)):
            trainset[i] = list(trainset[i])
            trainset[i][0] = (trainset[i][0]-mean)
            trainset[i] = tuple(trainset[i])
        
        for i in range(len(validset)):
            validset[i] = list(validset[i])
            validset[i][0] = (validset[i][0]-mean)
            validset[i] = tuple(validset[i])

    
        trainset = ToTensorDataset(trainset, transforms.ToTensor())
        validset = ToTensorDataset(validset, transforms.ToTensor())

  
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    valid_sampler = SequentialSampler(validset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=0,
                              pin_memory=True)
    valid_loader = DataLoader(validset,
                             sampler=valid_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=True) if validset is not None else None
    

    return train_loader, valid_loader
