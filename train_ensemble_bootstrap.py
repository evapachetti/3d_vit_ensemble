
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:45:17 2022

@author: Germanese
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
from tools import set_seed, normalize, parameters_config, get_config, calculate_confidence_metrics, brier_score_one_class, save_best_metrics, bootstrapping, testing_model
from tqdm import tqdm
import xlsxwriter
import logging
from create_dataset import ProstateDataset, ToTensorDataset
from create_dataset_bootstrapping import ProstateDataset as ProstateDatasetBoot


logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

Nrep = 3 # how many times repeat bootstrap

workbook = xlsxwriter.Workbook('Ensemble_Bootstrap.xlsx')
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
    def __init__(self, transformer_1, transformer_2, transformer_3, in_features=3, n_classes = 1):
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


configurations = list(range(1,19))
combs = list(combinations(configurations, 3))
out_path = os.path.join(os.getcwd(),"output")
net_path = os.path.join(out_path,"baseline_models") # directory where trained baseline models are stored

criterion = torch.nn.BCELoss()

csv_path = os.path.join(os.getcwd(),"csv_files","fixed_split")
csv_file_train = os.path.join(csv_path,"training.csv") # training csv
csv_file_val = os.path.join(csv_path,"validation.csv") # validation csv
csv_file_test = os.path.join(csv_path,"test.csv") # test csv

for comb in combs: #for each combination compute bootstrap

    logger.info("Ensemble combination #" + str(comb))
    
    results_val = {'Specificity':[], 'Sensitivity':[], 'Balanced Accuracy':[], 'AUROC':[], 'AUPRC':[], 'F2-score':[],'CSP':[], 'CSE':[], 'BSNC':[], 'BSPC':[], 'BS':[]} 
    results_test = {'Specificity':[], 'Sensitivity':[], 'Balanced Accuracy':[], 'AUROC':[], 'AUPRC':[], 'F2-score':[],'CSP':[], 'CSE':[], 'BSNC':[], 'BSPC':[], 'BS':[]} 
   
    c_t1, c_t2, c_t3 = comb[0], comb[1], comb[2]

    worksheet.write(row,column,str(c_t1)+"+"+str(c_t2)+"+"+str(c_t3))   
    column = 1
    worksheet.write(row,column,'Validation') 
    
    ps1,dim1,n1,hs1,nh1 = parameters_config(c_t1)
    ps2,dim2,n2,hs2,nh2 = parameters_config(c_t2)
    ps3,dim3,n3,hs3,nh3 = parameters_config(c_t3)
 
    config_1, config_2, config_3 = get_config(ps1,dim1,n1,hs1,nh1), get_config(ps2,dim2,n2,hs2,nh2), get_config(ps3,dim3,n3,hs3,nh3)

    #WD
    PATH_1 = os.path.join(net_path,"Conf_"+str(c_t1)+"\Best_model_Conf_"+str(c_t1)+".bin")
    PATH_2 = os.path.join(net_path,"Conf_"+str(c_t2)+"\Best_model_Conf_"+str(c_t2)+".bin")
    PATH_3 = os.path.join(net_path,"Conf_"+str(c_t3)+"\Best_model_Conf_"+str(c_t3)+".bin")

    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    for k in tqdm(range(Nrep)): 
        
        save_path = os.path.join(os.getcwd(),"output","Bootstrap_ensemble_models","Bootstrap_"+str(k)+".bin")

        set_seed()
        
        boot_trainframe = bootstrapping(csv_file_train, k)
        trainset = list(ProstateDatasetBoot(boot_trainframe)) #bootstrapped train
        
        validset = list(ProstateDataset(csv_file_val))
        testset = list(ProstateDataset(csv_file_test))


        # normalize data
        volumes_train = [i[0] for i in trainset]
        mean = np.mean(volumes_train)

        trainset, validset, testset = normalize(trainset,mean), normalize(validset,mean), normalize(testset,mean)
        
        # convert to tensor
        trainset, validset, testset_tf = ToTensorDataset(trainset, transforms.ToTensor()), ToTensorDataset(validset, transforms.ToTensor()), ToTensorDataset(testset, transforms.ToTensor())

        # dataloader
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
        
        testloader = torch.utils.data.DataLoader(testset_tf, batch_size=eval_batch_size, shuffle=False, num_workers=0)


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
                    
                running_loss = 0.0
                running_corrects = 0.0
        
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
                    roc_auc = roc_auc_score(true_labels,class_probabilities) # Getting ROC AUC
                    pr_auc = average_precision_score(true_labels, class_probabilities)
                    f2_score = fbeta_score(true_labels, predicted_labels, beta = 2)
                    
                    
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
                                                                                                                            
        # Confidence metrices        
        csp,cse = calculate_confidence_metrics(tl, pl, cp) 
        bs = brier_score_loss(tl, cp)
        bsnc = brier_score_one_class(tl, cp, cl = 0)
        bspc = brier_score_one_class(tl, cp, cl = 1)
                    
        
        results_val['Specificity'].append(best_spec)
        results_val['Sensitivity'].append(best_sens)
        results_val['Balanced Accuracy'].append(best_bacc)
        results_val['AUROC'].append(best_auc)
        results_val['AUPRC'].append(best_aupr)
        results_val['F2-score'].append(best_f2)
        results_val['CSP'].append(csp)
        results_val['CSE'].append(cse)
        results_val['BSNC'].append(bsnc)
        results_val['BSPC'].append(bspc)
        results_val['BS'].append(bs)
        
        ### HOLD-OUT-TEST ###
        true_labels_t, predicted_labels_t, class_probabilities_t, features_vectors = testing_model(testloader, ensemble,device)
        
        b_accuracy = balanced_accuracy_score(true_labels_t, predicted_labels_t) # Balanced accuracy
        specificity = recall_score(true_labels_t, predicted_labels_t, pos_label = 0) #Specificity
        sensitivity = recall_score(true_labels_t, predicted_labels_t) #Sensitivity
        roc_auc = roc_auc_score(true_labels_t,class_probabilities_t) # Getting ROC AUC
        pr_auc = average_precision_score(true_labels_t, class_probabilities_t)
        f2_score = fbeta_score(true_labels_t, predicted_labels_t, beta = 2)
        
        csp,cse = calculate_confidence_metrics(true_labels_t, predicted_labels_t, class_probabilities_t) 
        bs = brier_score_loss(true_labels_t, class_probabilities_t)
        bsnc = brier_score_one_class(true_labels_t, class_probabilities_t, cl = 0)
        bspc = brier_score_one_class(true_labels_t, class_probabilities_t, cl = 1)
        
        results_test['Specificity'].append(specificity)
        results_test['Sensitivity'].append(sensitivity)
        results_test['Balanced Accuracy'].append(b_accuracy)
        results_test['AUROC'].append(roc_auc)
        results_test['AUPRC'].append(pr_auc)
        results_test['F2-score'].append(f2_score)
        results_test['CSP'].append(csp)
        results_test['CSE'].append(cse)
        results_test['BSNC'].append(bsnc)
        results_test['BSPC'].append(bspc)
        results_test['BS'].append(bs)
        

        for metric in metrics:
            worksheet.write(row,column,str(round(results_val[metric][k],3)))
            column +=1
        column=1
        row+=1
        for metric in metrics:
            worksheet.write(row,column,str(round(results_test[metric][k],3)))
            column +=1
        column=1
        row+=2
       
   
    for metric in metrics:
        np.save(os.path.join(out_path,"Bootstrap_results_"+metric),results_test[metric])

    row += 2
    column = 0


workbook.close()

    
