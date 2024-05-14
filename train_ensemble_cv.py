# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import torch.nn as nn
import torch
from models.modeling import VisionTransformer
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score, roc_auc_score,fbeta_score, average_precision_score
import os
from itertools import combinations
from tools import set_seed, normalize, parameters_config, get_config, calculate_confidence_metrics, brier_score_one_class,save_best_metrics
from tqdm import tqdm
import xlsxwriter
from create_dataset import ProstateDataset, ToTensorDataset
import logging

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


workbook = xlsxwriter.Workbook('Ensemble.xlsx')
worksheet = workbook.add_worksheet()
bold = workbook.add_format({'bold': True})

row = 1
column = 0

worksheet.write(row,column, 'Ensemble', bold)
column = 1


metrics = ['Specificity', 'Sensitivity', 'Balanced Accuracy', 'AUROC', 'AUPRC', 'F2-score','CSP', 'CSE', 'BSNC', 'BSPC', 'BS']

for metric in metrics:
    worksheet.write(row,column, metric, bold)
    column += 1

column = 0
row +=1


train_batch_size = 4
eval_batch_size = 1


class TransformerEnsemble(nn.Module):
    def __init__(self, transformer_1, transformer_2,transformer_3, in_features=3, n_classes = 1):
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
        out = torch.sigmoid(self.classifier(x))
        
        return out

cv = 5 # default is 5-fold CV

configurations = list(range(1,19))
combs = list(combinations(configurations, 3))
combs = [tuple(combs[combs.index((5,9,11))])]
net_path = os.path.join(os.getcwd(),"output","baseline_models") # directory where trained baseline models are stored
out_path = os.path.join(os.getcwd(),"output")

criterion = torch.nn.BCELoss()

csv_path = os.path.join(os.getcwd(),"csv_files","cross_validation")


for comb in tqdm(combs): 

    logger.info("Ensemble combination #" + str(comb))

    results = {} #results of each combination in k-fold CV
    c_t1, c_t2, c_t3 = comb[0], comb[1], comb[2]
  
    worksheet.write(row,column,str(c_t1)+"+"+str(c_t2))   
    column = 1
    worksheet.write(row,column,'Validation') 
    
    ps1,dim1,n1,hs1,nh1 = parameters_config(c_t1)
    ps2,dim2,n2,hs2,nh2 = parameters_config(c_t2)
    ps3,dim3,n3,hs3,nh3 = parameters_config(c_t3)
 
    config_1, config_2, config_3 = get_config(ps1,dim1,n1,hs1,nh1), get_config(ps2,dim2,n2,hs2,nh2), get_config(ps3,dim3,n3,hs3,nh3)
  
    PATH_1 = os.path.join(net_path,"Conf_"+str(c_t1),"Best_model_Conf_"+str(c_t1)+".bin") # Best trained baseline model for each configuration
    PATH_2 = os.path.join(net_path,"Conf_"+str(c_t2),"Best_model_Conf_"+str(c_t2)+".bin")
    PATH_3 = os.path.join(net_path,"Conf_"+str(c_t3),"Best_model_Conf_"+str(c_t3)+".bin")

    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    for k in tqdm(range(cv)): # k-fold CV
    
        logger.info("Cross-validation fold #" + str(k+1))
        save_path = os.path.join(out_path,"cv_"+str(k+1)+".bin")

        set_seed()
        
        results["CV" +str(k+1)] = {} # results of configuration comb at this cv

        csv_file_train = os.path.join(csv_path,"training_cv"+str(k+1)+".csv") # training k-cv csv
        csv_file_val = os.path.join(csv_path,"validation_cv"+str(k+1)+".csv") # validation k-cv csv
        
        trainset, validset = list(ProstateDataset(csv_file_train)), list(ProstateDataset(csv_file_val))

        # data normalization 
        volumes_train = [i[0] for i in trainset]
        mean = np.mean(volumes_train)

        trainset = normalize(trainset,mean)
        validset = normalize(validset,mean)

        # conversion to tensor
        trainset, validset = ToTensorDataset(trainset, transforms.ToTensor()), ToTensorDataset(validset, transforms.ToTensor())

        # data loader
        train_sampler = RandomSampler(trainset) 
        valid_sampler = SequentialSampler(validset)
        train_loader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=train_batch_size,
                                  num_workers=0,
                                  pin_memory=True)
        valid_loader = DataLoader(validset,
                                 sampler=valid_sampler,
                                 batch_size=eval_batch_size,
                                 num_workers=0,
                                 pin_memory=True) if validset is not None else None

        # load pre-trained transformers (baseline version)
        transformer_1 = VisionTransformer(config_1, 128, zero_head=True, num_classes=1)
        transformer_1.load_state_dict(torch.load(PATH_1, map_location=device))
        
        transformer_2 = VisionTransformer(config_2, 128, zero_head=True, num_classes=1)
        transformer_2.load_state_dict(torch.load(PATH_2, map_location = device))
        
        transformer_3 = VisionTransformer(config_3, 128, zero_head=True, num_classes=1)
        transformer_3.load_state_dict(torch.load(PATH_3, map_location = device))
        
        # define ensemble model
        ensemble = TransformerEnsemble(transformer_1, transformer_2, transformer_3)
        ensemble.to(device)
        
        optimizer = optim.Adam(ensemble.parameters(), lr= 1e-4)
        num_epochs = 100
        
        dset_loaders = {'train': train_loader, 'val': valid_loader}
        classes = ('LG','HG')
        
        val_loss_array, train_loss_array, val_accuracy_array, train_accuracy_array, aucs = [], [], [], [], []
        best_spec, best_sens, best_bacc, best_auc, best_aupr, best_f2 = 0,0,0,0,0,0
        best_choice = False
        old_valid_loss = 100
    

        #---TRAINING---
        
        for epoch in range(num_epochs): 
    
            predicted_labels, true_labels, class_probabilities = [],[],[]
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    ensemble.train(True)
                else:
                    ensemble.eval()
                    
                running_loss, running_corrects = 0.0, 0.0
        
                for data in dset_loaders[phase]:
                    
                    inputs,labels = data[0], data[1]
                    inputs, labels = inputs.float().to(device), labels.float().to(device)
                    
                    optimizer.zero_grad() 
                    outputs = ensemble(inputs)

                    if len(outputs)>1:
                        outputs = torch.squeeze(outputs)
                    else: 
                        outputs = torch.squeeze(outputs).unsqueeze(0)
                                    
                    loss = criterion(outputs,labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    elif phase == 'val':
                        predicted = 1 * (outputs > 0.5)
                        predicted_labels.append(predicted)
                        true_labels.append(labels)
                        class_probabilities.extend(outputs)
        
                    running_loss += loss.item()
                        
                phase_loss = running_loss / dset_loaders[phase].__len__()
                phase_acc = running_corrects / dset_loaders[phase].__len__()
                
                if phase == 'train':
                    train_loss_array.append(phase_loss)
                    train_accuracy_array.append(phase_acc)
                    train_loss = phase_loss
                else:
                    val_loss_array.append(phase_loss)
                    val_accuracy_array.append(phase_acc)
                    valid_loss = phase_loss
                    
                    true_labels = [i.item() for i in true_labels]
                    class_probabilities = [i.item() for i in class_probabilities]
                    
                    b_accuracy = balanced_accuracy_score(true_labels, predicted_labels) # Balanced accuracy
                    specificity = recall_score(true_labels, predicted_labels,pos_label = 0) #Specificity
                    sensitivity = recall_score(true_labels, predicted_labels) #Sensitivity
                    roc_auc = roc_auc_score(true_labels,class_probabilities) # AUROC
                    pr_auc = average_precision_score(true_labels, class_probabilities) #AUPRC
                    f2_score = fbeta_score(true_labels, predicted_labels, beta = 2) #F2-score
                    
                    # Custom decision process to ensure both spec and sens > 0.5 if this happens, otherwise I look at AUROC alone
                    if specificity > 0.6 and sensitivity > 0.6:
                        if best_choice == False:
                            best_spec, best_sens, best_bacc, best_auc, best_aupr, best_f2, tl, pl, cp = save_best_metrics(save_path, ensemble, specificity, sensitivity, b_accuracy, roc_auc, pr_auc, f2_score, true_labels, predicted_labels, class_probabilities)
                            best_choice = True
                        else: 
                            if roc_auc > best_auc:
                                best_spec, best_sens, best_bacc, best_auc, best_aupr, best_f2, tl,pl, cp = save_best_metrics(save_path, ensemble, specificity, sensitivity, b_accuracy, roc_auc, pr_auc, f2_score, true_labels, predicted_labels, class_probabilities)
                    else:
                        if best_choice == False:
                            if roc_auc > best_auc:
                                best_spec, best_sens, best_bacc, best_auc,best_aupr, best_f2, tl,pl, cp = save_best_metrics(save_path, ensemble, specificity, sensitivity, b_accuracy, roc_auc, pr_auc, f2_score, true_labels, predicted_labels, class_probabilities)
                     
        # Confidence metrics
        csp,cse = calculate_confidence_metrics(tl, pl, cp) 
        bs = brier_score_loss(tl, cp)
        bsnc = brier_score_one_class(tl, cp, cl = 0)
        bspc = brier_score_one_class(tl, cp, cl = 1)
                    
        
        results["CV"+ str(k+1)]['Specificity'] = best_spec
        results["CV"+ str(k+1)]['Sensitivity'] = best_sens
        results["CV"+ str(k+1)]['Balanced Accuracy'] = best_bacc
        results["CV"+ str(k+1)]['AUROC'] = best_auc
        results["CV"+ str(k+1)]['AUPRC'] = best_aupr
        results["CV"+ str(k+1)]['F2-score'] = best_f2
        results["CV"+ str(k+1)]['CSP'] = csp
        results["CV"+ str(k+1)]['CSE'] = cse
        results["CV"+ str(k+1)]['BSNC'] = bsnc
        results["CV"+ str(k+1)]['BSPC'] = bspc
        results["CV"+ str(k+1)]['BS'] = bs


    for metric in metrics:
        mean_metric = np.mean([results["CV"+ str(k+1)][metric] for k in range(cv)])
        std_metric = np.std([results["CV"+ str(k+1)][metric] for k in range(cv)])
        worksheet.write(row,column,str(round(mean_metric,3))+" "+"("+str(round(std_metric,3))+")")  
        column +=1

    row += 2
    column = 0
    
workbook.close()

    
