
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:43:57 2022

@author: Germanese
"""

from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import recall_score, roc_auc_score, fbeta_score, balanced_accuracy_score, average_precision_score
from models.modeling import VisionTransformer
import numpy as np
from matplotlib import pyplot as plt
import os
from tools import set_seed, normalize, parameters_config, get_config, calculate_confidence_metrics, brier_score_one_class, testing_model
from sklearn.metrics import brier_score_loss
import pandas as pd
from create_dataset import ProstateDataset, ToTensorDataset
from itertools import combinations
from tqdm import tqdm

cv = 5 #number of k-fold
set_seed()
im_size = 128

csv_path = os.path.join(os.getcwd(),"csv_files","cross_validation")
csv_file_test = os.path.join(csv_path, "test.csv")
testset = list(ProstateDataset(csv_file_test))

batch_size_test = 1
device = torch.device("cpu")

#%% TEST BASE VIT

conf = 5
ps,dim,n,hs,nh = parameters_config(conf)
config = get_config(ps,dim,n,hs,nh)

out_path = os.path.join(os.getcwd(),"output")
net_path = os.path.join(out_path,"baseline_models") # directory where trained ensemble models are stored

PATH = os.path.join(net_path,"Best_model_Conf_"+str(conf)+".bin")

res_base, median_res_base, mean_res_base, percents_2_5_base, percents_97_5_base, percents_25_base, percents_75_base = {},{},{},{},{},{},{}


for k in tqdm(range(cv)):
    
    load_path = os.path.join(net_path,"conf_"+str(conf)+"cv_"+str(cv)+".bin")
    res_base['CV '+str(k)] = {}
    
    csv_file_train = os.path.join(csv_path,"training_cv"+str(k+1)+".csv") # training k-cv csv
    csv_file_val = os.path.join(csv_path,"validation_cv"+str(k+1)+".csv") # validation k-cv csv
    
    trainset, validset = list(ProstateDataset(csv_file_train)),list(ProstateDataset(csv_file_val))
    
    # normalize data
    volumes_train = [i[0] for i in list(trainset)]
    mean = np.mean(volumes_train)
 
    testset = normalize(testset,mean)
    validset = normalize(validset,mean)
    
    # dataloader
    valset_tf = ToTensorDataset(validset, transforms.ToTensor())
    valoader = torch.utils.data.DataLoader(valset_tf, batch_size=batch_size_test, shuffle=False, num_workers=0)
    
    testset_tf = ToTensorDataset(testset, transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset_tf, batch_size=batch_size_test, shuffle=False, num_workers=0)
    
    # load model
    model = VisionTransformer(config,im_size, zero_head=False, num_classes=1,vis=True)
    model.load_state_dict(torch.load(PATH))

    model.to(device)
    model.eval()
    
    true_labels, predicted_labels, class_probabilities, features_vectors = testing_model(testloader, model,device)
    
    b_accuracy = balanced_accuracy_score(true_labels, predicted_labels) # Balanced accuracy
    specificity = recall_score(true_labels, predicted_labels,pos_label = 0) #Specificity
    sensitivity = recall_score(true_labels, predicted_labels) #Sensitivity
    roc_auc = roc_auc_score(true_labels,class_probabilities) # Getting ROC AUC
    f2_score = fbeta_score(true_labels, predicted_labels, beta = 2)
    ap_score = average_precision_score(true_labels,class_probabilities)
    
    brier_score = brier_score_loss(true_labels, class_probabilities)
    bs_0 = brier_score_one_class(true_labels, class_probabilities, cl = 0)
    bs_1 = brier_score_one_class(true_labels, class_probabilities, cl = 1)
    csp,cse = calculate_confidence_metrics(true_labels, predicted_labels, class_probabilities) 
    
    res_base['CV '+str(k)]['Specificity'] = specificity
    res_base['CV '+str(k)]['Sensitivity'] = sensitivity
    res_base['CV '+str(k)]['Balanced Accuracy'] = b_accuracy
    res_base['CV '+str(k)]['AUROC'] = roc_auc
    res_base['CV '+str(k)]['AUPRC'] = ap_score
    res_base['CV '+str(k)]['F2-score'] = f2_score
    res_base['CV '+str(k)]['Brier score'] = brier_score
    res_base['CV '+str(k)]['BS_0'] = bs_0
    res_base['CV '+str(k)]['BS_1'] = bs_1    
    res_base['CV '+str(k)]['CSP'] = csp    
    res_base['CV '+str(k)]['CSE'] = cse
    
    
for s in range(cv):
    for k in res_base['CV '+str(s)].keys():
        median_res_base[k] = np.median([res_base['CV '+str(k)][k] for k in range(cv)])
        mean_res_base[k] = np.mean([res_base['CV '+str(k)][k] for k in range(cv)])
        percents_2_5_base[k] = np.percentile([res_base['CV '+str(k)][k] for k in range(cv)],2.5)
        percents_97_5_base[k] = np.percentile([res_base['CV '+str(k)][k] for k in range(cv)],97.5)
        percents_25_base[k] = np.percentile([res_base['CV '+str(k)][k] for k in range(cv)],25)
        percents_75_base[k] = np.percentile([res_base['CV '+str(k)][k] for k in range(cv)],75)


print ("---Base ViT results (median and 90% CI)---")
print()
for key in median_res_base.keys():
    print(key+": ", str(round(median_res_base[key],3)) + "["+ str(round(percents_2_5_base[key],3)) + "-" + str(round(percents_97_5_base[key],3)) + "]")


#%% TEST ENSEMBLE VIT

class TransformerEnsemble(nn.Module):
    def __init__(self, transformer_1, transformer_2, transformer_3,in_features=3, n_classes = 1):
        super(TransformerEnsemble, self).__init__()
        self.transformer_1 = transformer_1
        self.transformer_2 = transformer_2
        self.transformer_3 = transformer_3
       
        self.classifier = nn.Linear(in_features, n_classes)
    
    def forward(self,x1):
        out1 = self.transformer_1(x1)[0]
        out2 = self.transformer_2(x1)[0]
        out3 = self.transformer_3(x1)[0]
     
        x = torch.cat((out1,out2,out3), dim = 1)
        out = self.classifier(x)
        
        return out


configurations = list(range(1,19))
combs = list(combinations(configurations, 3))
net_path = os.path.join(os.getcwd(),"output","ensemble_models") # directory where trained ensemble models are stored
load_path = os.path.join(net_path,"Best_Ensemble.bin")

res_ens, median_res_ens, mean_res_ens, percents_2_5_ens, percents_97_5_ens, percents_25_ens, percents_75_ens = {},{},{},{},{},{},{}

for comb in tqdm(combs): 
    for k in tqdm(range(cv)):
        
        load_path = os.path.join(net_path,"conf_"+str(conf)+"cv_"+str(cv)+".bin")

        res_ens['CV '+str(k)] = {}
    
        valset = list(ProstateDataset(csv_file_val))
        testset = list(ProstateDataset(csv_file_test))

        valset = normalize(valset,mean)
        testset = normalize(testset,mean)
        
        valset_tf = ToTensorDataset(valset, transforms.ToTensor())
        valoader = torch.utils.data.DataLoader(valset_tf, batch_size=batch_size_test, shuffle=False, num_workers=0)
        
        testset_tf = ToTensorDataset(testset, transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset_tf, batch_size=batch_size_test, shuffle=False, num_workers=0)

        c_t1, c_t2, c_t3 = comb[0], comb[1], comb[2]
        
        ps1,dim1,n1,hs1,nh1 = parameters_config(c_t1)
        ps2,dim2,n2,hs2,nh2 = parameters_config(c_t2)
        ps3,dim3,n3,hs3,nh3 = parameters_config(c_t3)
            
        config_1, config_2, config_3 = get_config(ps1,dim1,n1,hs1,nh1), get_config(ps2,dim2,n2,hs2,nh2), get_config(ps3,dim3,n3,hs3,nh3)
            
        PATH_1 = os.path.join(net_path,"Conf_"+str(c_t1)+"\Best_model_Conf_"+str(c_t1)+".bin")
        PATH_2 = os.path.join(net_path,"Conf_"+str(c_t2)+"\Best_model_Conf_"+str(c_t2)+".bin")
        PATH_3 = os.path.join(net_path,"Conf_"+str(c_t3)+"\Best_model_Conf_"+str(c_t3)+".bin")
        
            
        transformer_1 = VisionTransformer(config_1, 128, zero_head=True, num_classes=1)
        transformer_1.load_state_dict(torch.load(PATH_1, map_location=device))
        
        transformer_2 = VisionTransformer(config_2, 128, zero_head=True, num_classes=1)
        transformer_2.load_state_dict(torch.load(PATH_2, map_location = device))
        
        transformer_3 = VisionTransformer(config_3, 128, zero_head=True, num_classes=1)
        transformer_3.load_state_dict(torch.load(PATH_3, map_location = device))
            
        
        ensemble = TransformerEnsemble(transformer_1, transformer_2, transformer_3)
        ensemble.load_state_dict(torch.load(load_path, map_location=device))    
        ensemble.eval()
        
        true_labels, predicted_labels, class_probabilities, features_vectors = testing_model(testloader, ensemble,device)
        
        b_accuracy = balanced_accuracy_score(true_labels, predicted_labels) # Balanced accuracy
        specificity = recall_score(true_labels, predicted_labels,pos_label = 0) #Specificity
        sensitivity = recall_score(true_labels, predicted_labels) #Sensitivity
        roc_auc = roc_auc_score(true_labels,class_probabilities) # Getting ROC AUC
        f2_score = fbeta_score(true_labels, predicted_labels, beta = 2)
        ap_score = average_precision_score(true_labels,class_probabilities)
        
        brier_score = brier_score_loss(true_labels, class_probabilities)
        bs_0 = brier_score_one_class(true_labels, class_probabilities, cl = 0)
        bs_1 = brier_score_one_class(true_labels, class_probabilities, cl = 1)
        csp,cse = calculate_confidence_metrics(true_labels, predicted_labels, class_probabilities) 
        
        res_ens['CV '+str(k)]['Specificity'] = specificity
        res_ens['CV '+str(k)]['Sensitivity'] = sensitivity
        res_ens['CV '+str(k)]['Balanced Accuracy'] = b_accuracy
        res_ens['CV '+str(k)]['AUROC'] = roc_auc
        res_ens['CV '+str(k)]['AUPRC'] = ap_score
        res_ens['CV '+str(k)]['F2-score'] = f2_score
        res_ens['CV '+str(k)]['Brier score'] = brier_score
        res_ens['CV '+str(k)]['BS_0'] = bs_0
        res_ens['CV '+str(k)]['BS_1'] = bs_1    
        res_ens['CV '+str(k)]['CSP'] = csp    
        res_ens['CV '+str(k)]['CSE'] = cse


    for s in range(cv):
        for k in res_ens['CV '+str(s)].keys():
            median_res_ens[k] = np.median([res_ens['CV '+str(k)][k] for k in range(cv)])
            mean_res_ens[k] = np.mean([res_ens['CV '+str(k)][k] for k in range(cv)])
            percents_2_5_ens[k] = np.percentile([res_ens['CV '+str(k)][k] for k in range(cv)],2.5)
            percents_97_5_ens[k] = np.percentile([res_ens['CV '+str(k)][k] for k in range(cv)],97.5)
            percents_25_ens[k] = np.percentile([res_ens['CV '+str(k)][k] for k in range(cv)],25)
            percents_75_ens[k] = np.percentile([res_ens['CV '+str(k)][k] for k in range(cv)],75)
            
    print("---Ensemble ViT results (median and 90% CI)---")
    print()

    for key in median_res_ens.keys():
        print(key+": ",str(round(median_res_ens[key],3)) + "["+ str(round(percents_2_5_ens[key],3)) + "-" + str(round(percents_97_5_ens[key],3)) + "]")
        
    
    
    
#%% STATISTICAL TESTS

from scipy import stats
import seaborn as sns
import textwrap

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)

auprcs_base = [res_base[k]['AUPRC'] for k in res_base.keys()]
auprcs_ens = [res_ens[k]['AUPRC'] for k in res_ens.keys()]

cse_base = [res_base[k]['CSE'] for k in res_base.keys()]
cse_ens = [res_ens[k]['CSE'] for k in res_ens.keys()]

bs_base = [res_base[k]['Brier score'] for k in res_base.keys()]
bs_ens = [res_ens[k]['Brier score'] for k in res_ens.keys()]


#TEST FOR NORMALITY: SHAPIRO WILK

shapiro_test_base = stats.shapiro(bs_base)
shapiro_test_ens = stats.shapiro(bs_ens)

#TEST FOR DISTRIBUTION EQUALITY

median_test = stats.median_test(bs_base,bs_ens)
mw_test = stats.mannwhitneyu(auprcs_base, auprcs_ens)

plt.figure()
sns.histplot(bs_base, kde = True, color='orange')
sns.histplot(bs_ens, kde = True, color='navy')

df = pd.DataFrame([auprcs_base,auprcs_ens,cse_base,cse_ens,bs_base,bs_ens]).transpose().rename(columns={0:"AUPRC Base",1:"AUPRC Ensemble", 2:"CSE Base", 3:"CSE Ensemble",4:"BS Base",5:"BS Ensemble"})

plt.figure()
ax= sns.boxplot(data=df)
wrap_labels(ax,5)
plt.savefig(os.path.join(out_path,"box_plot.eps",format='eps'))

#%% Wilcoxon test

wilcox_test = stats.wilcoxon(auprcs_base,auprcs_ens)
print(wilcox_test)
