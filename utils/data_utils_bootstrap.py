# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 10:53:05 2021

@author: Germanese
"""


import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import numpy as np
from stratified_group_data_splitting import StratifiedGroupKFold
import pandas as pd
import os


logger = logging.getLogger(__name__)

import pandas as pd


# def bootstrapping(input_csv, seed):
#     ind_data = pd.read_csv(input_csv) # lettura file csv dei dati di test
#     index = ind_data.index
#     ind_data_patient = ind_data.Patient.unique() # creazione della lista di valori che può assumere la colonna “Lesion”, presente nel mio ind_data. Attenzione che questa lista cambia per ogni modalità di acquisizione.
#     ind_data_patient_df = pd.DataFrame(ind_data_patient, columns=['Patient']) # trasformo la lista in DataFrame
#     bootstrap_sample_size_pat = len(ind_data_patient_df) 
#     bootstrap_sample_size_les = len(ind_data)
#     bootstrap_sample = ind_data_patient_df.sample(n = bootstrap_sample_size_les, replace = True, random_state = seed, axis='index') # campiono casualmente e con ripetizione le lesioni
#     new_indices = []
#     for j in bootstrap_sample.Patient.tolist():
#         new_indices.append(index[ind_data['Patient'] == j].tolist()) # prendo le fette corrispondenti alle lesioni selezionate (anche ripetute)
#     flat_new_indices = [item for sublist in new_indices for item in sublist] # metto tutti gli indici in un unico vettore
#     boot_trainset = ind_data.iloc[flat_new_indices,:]
      
#     return boot_trainset

def bootstrapping(input_csv, seed):
    ind_data = pd.read_csv(input_csv) # lettura file csv dei dati di test
    ind_data = ind_data.reset_index()
    index = ind_data.index
    ind_data_case_df = pd.DataFrame(list(ind_data['index']), columns=['Case'])
    bootstrap_sample_size = len(ind_data_case_df)
    bootstrap_sample = ind_data_case_df.sample(n = bootstrap_sample_size, replace = True, random_state = seed, axis='index') # campiono casualmente e con ripetizione le lesioni
    new_indices = []
    for j in bootstrap_sample.Case.tolist():
        new_indices.append(index[ind_data['index'] == j].tolist())# prendo le fette corrispondenti alle lesioni selezionate (anche ripetute)
    flat_new_indices = [item for sublist in new_indices for item in sublist] # metto tutti gli indici in un unico vettore
    boot_trainset = ind_data.iloc[flat_new_indices,:].drop('index',axis=1)
      
    return boot_trainset

def get_loader(args,seed):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    if args.dataset == "prostateX":
        
        from create_volume_dataset_bootstrapping_PX2 import ProstateDataset, ToTensorDataset

        
        csv_path = r"C:\Users\Germanese\Desktop\Eva\Lavoro\Lavoro MDPI\Csv splitting\ProstateX-2\Split 3D"
        csv_file_train = os.path.join(csv_path,"png_PROSTATEx2_T2_volume_Training_final.csv")
        csv_file_val = os.path.join(csv_path,"png_PROSTATEx2_T2_volume_Validation_final.csv")
        
        val_dataframe = pd.read_csv(csv_file_val)
        validset = list(ProstateDataset(val_dataframe))
        
        boot_trainframe = bootstrapping(csv_file_train, seed)
        trainset = list(ProstateDataset(boot_trainframe))

        
        volumes_train = [i[0] for i in list(trainset)]
        mean = np.mean(volumes_train)
        std = np.std(volumes_train)
        
        #AUGMENTATION
        aug_suffix = ['Rotazione', 'Flip verticale', 'Flip orizzontale']
        trainsets_aug = []
        
        for h in aug_suffix: # Per ogni tecnica di data augmentation
            trainsets_aug.append(ProstateDataset(boot_trainframe,aug_folder = h)) # dataset con solo immagini alterate
        
        #SOLO PER PROSTATEX
        hg_positions = [i for i in range(len(trainset)) if trainset[i][1].item() == 1] # trovo tutte le posizioni del training set dove label = HG
        positions = [hg_positions[i] for i in range(0,len(hg_positions),2)] #fisso le immagini che vengono aumentate (dipendono solo da come è stato suddiviso il dataset) (in tutto 8 posizioni * 3 = 24 immagini aggiunte alle 24 di partenza)
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
            
    if args.dataset == "Careggi":
        
        from create_volume_dataset_bootstrapping_Careggi import ProstateDataset, ToTensorDataset

        n_pos_aug = 2

        csv_path = r"C:\Users\Germanese\Desktop\Eva\Lavoro\Lavoro MDPI\Csv splitting\Careggi"
        csv_file_train = os.path.join(csv_path,"DICOM_Careggi_Volume_Training.csv")
        csv_file_val = os.path.join(csv_path,"DICOM_Careggi_Volume_Validation.csv")
        
        val_dataframe = pd.read_csv(csv_file_val)
        validset = list(ProstateDataset(val_dataframe))
        
        boot_trainframe = bootstrapping(csv_file_train, seed)
        trainset = list(ProstateDataset(boot_trainframe))

        
        volumes_train = [i[0] for i in list(trainset)]
        mean = np.mean(volumes_train)
        std = np.std(volumes_train)
        
        #AUGMENTATION
        aug_suffix = ['Rotazione', 'Flip verticale', 'Flip orizzontale']
        trainsets_aug = []
        
        for h in aug_suffix: # Per ogni tecnica di data augmentation
            trainsets_aug.append(ProstateDataset(boot_trainframe,aug_folder = h)) # dataset con solo immagini alterate
        
        #SOLO PER PROSTATEX
        hg_positions = [i for i in range(len(trainset)) if trainset[i][1].item() == 1] # trovo tutte le posizioni del training set dove label = HG
        positions = [hg_positions[i] for i in range(0,len(hg_positions),n_pos_aug)] 
        
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
