# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:24:12 2022

@author: Germanese
"""

import random
import numpy as np
import torch
import ml_collections
import pandas as pd

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize(dataset, mean, std=1):
    
    for i in range(len(dataset)):
        dataset[i] = list(dataset[i])
        dataset[i][0] = (dataset[i][0]-mean)
        dataset[i] = tuple(dataset[i])
        
    return dataset

def save_best_metrics(save_path, model, specificity,sensitivity,accuracy,roc_auc,pr_auc,f2_score, true_labels, predicted_labels, class_probabilities):
            best_spec = specificity
            best_sens = sensitivity
            best_acc = accuracy
            best_auc = roc_auc
            best_aupr = pr_auc
            best_f2 = f2_score
            tl = true_labels
            pl = predicted_labels
            cp = class_probabilities
            torch.save(model.state_dict(),save_path)    
            
            return best_spec, best_sens, best_acc, best_auc,best_aupr, best_f2, tl, pl, cp
    

def testing_model(dataloader,model,device):

    predicted_labels = []
    true_labels = []
    class_probabilities = []
    features_vectors = []
    logits = []
    
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        y = y.long()
        with torch.no_grad():
            output = model(x)[0]
            logits.append(output.item())
            #features_v = np.array(model(x)[2][:,0].squeeze()) #vettore delle features
            features_v = 0 #DA RICAMBIARE
            p = torch.sigmoid(output)
            if p > 0.5:
                predicted = 1
            else:
                predicted = 0
            
            predicted_labels.append(predicted)    
            true_labels.append(y.item())
            class_probabilities.append(p)
            features_vectors.append(features_v)

    class_probabilities = [i.item() for i in class_probabilities]
    
    return true_labels, predicted_labels, class_probabilities, features_vectors

def parameters_config(conf):
    
    configurations = {}
    mlp_dim = [2048,3072]
    n_layers = [4,6,8]
    hs_nh = [(64,4), (32,8), (16,16)]
    k=1

    for dim in mlp_dim:
        for n in n_layers:
            for hs,nh in hs_nh:
                configurations['Configuration '+str(k)] = [16,dim,n,hs,nh]
                k += 1
                         

    mlp_dim = [2204]
    n_layers = [4,6]
    hs_nh = [(16,4),(8,8)]

    k = 19
    for dim in mlp_dim:
        for n in n_layers:
            for hs,nh in hs_nh:
                configurations['Configuration '+str(k)] = [8,dim,n,hs,nh]
                k += 1
                
    ps,dim,n,hs,nh = configurations['Configuration '+str(conf)][0], configurations['Configuration '+str(conf)][1], configurations['Configuration '+str(conf)][2],  configurations['Configuration '+str(conf)][3],  configurations['Configuration '+str(conf)][4]


    return ps,dim,n,hs,nh 


def get_config(ps,dim,n,hs,nh):
    """Returns the EvaViT configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (ps, ps, 5)})
    config.hidden_size = hs
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = dim
    config.transformer.num_heads = nh
    config.transformer.num_layers = n
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

def calculate_confidence_metrics(true_labels,predicted_labels, class_probabilities, metric='CRNC'):
    
    zipped_data = list(zip(true_labels,predicted_labels,class_probabilities))
    
    true_negatives = len([data for data in zipped_data if data[0]==data[1]==0])
    true_negatives_0_3 = len([data for data in zipped_data if (data[0]==data[1]==0 and data[2]<0.3)])
    false_negatives = len([data for data in zipped_data if (data[0]==1 and data[1]==0)])
    true_positives = len([data for data in zipped_data if data[0]==data[1]==1])
    true_positives_0_7 = len([data for data in zipped_data if (data[0]==data[1]==1 and data[2] >0.7)])
    false_positives = len([data for data in zipped_data if (data[0]==0 and data[1]==1)])
    

    try:
        csp = true_negatives_0_3/(true_negatives+false_positives)
    except ZeroDivisionError:
        csp = 0
    
   
    try:
        cse = true_positives_0_7/(true_positives+false_negatives)
    except ZeroDivisionError:
        cse = 0
            
        
    return csp,cse



def brier_score_one_class(y_true, y_prob, *, sample_weight=None, cl = 0):
        
    true_prob = list(zip(y_true,y_prob))
    
    y_true = np.array([l for l,p in true_prob if l == cl],int)
    y_prob = np.array([p for l,p in true_prob if l == cl])

    return np.average((y_true - y_prob) ** 2)


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