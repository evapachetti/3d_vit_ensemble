# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import torch.nn as nn # type: ignore
import torch # type: ignore
from models.modeling import VisionTransformer, TransformerEnsemble
import numpy as np # type: ignore
import torch.optim as optim # type: ignore
import torchvision.transforms as transforms # type: ignore
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler # type: ignore
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score, roc_auc_score,fbeta_score, average_precision_score # type: ignore
import os
from itertools import combinations
from tools import set_seed, normalize, parameters_config, get_config, calculate_confidence_metrics, brier_score_one_class,save_best_metrics
from tqdm import tqdm # type: ignore
import xlsxwriter # type: ignore
from create_dataset import ProstateDataset, ToTensorDataset
import logging

# Set reproducibility seed
set_seed()

# Set up logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set constants
cv = 5 # default is 5-fold CV
train_batch_size = 4
eval_batch_size = 1
num_epochs = 100
old_valid_loss = 100
classes = ('LG','HG')
best_choice = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create Excel workbook and worksheet
workbook = xlsxwriter.Workbook('Ensemble_Bootstrap.xlsx')
worksheet = workbook.add_worksheet()
bold = workbook.add_format({'bold': True})

# Write headers
row, column = 1, 0
worksheet.write(row, column, 'Ensemble', bold)
column = 1
metrics = ['Specificity', 'Sensitivity', 'Balanced Accuracy', 'AUROC', 'AUPRC', 'F2-score', 'CSP', 'CSE', 'BSNC', 'BSPC', 'BS']

for metric in metrics:
    worksheet.write(row, column, metric, bold)
    column += 1

# Reset column and move to next row
column = 0
row += 1


# Generate combinations
configurations = range(1, 19)
combs = list(combinations(configurations, 3))

# Define paths
base_path = os.getcwd()
output_path = os.path.join(base_path, "output")
net_path = os.path.join(output_path, "baseline_models")  # directory where trained baseline models are stored

# Define file paths
csv_path = os.path.join(base_path, "csv_files", "cross_validation")

# Define criterion
criterion = nn.BCELoss()

# Trainin loop
for comb in tqdm(combs): 

    logger.info("Ensemble combination #" + str(comb))

    results = {} #results of each combination in k-fold CV
    c_t1, c_t2, c_t3 = comb
    ensemble_name = f"{c_t1}_{c_t2}_{c_t3}"
  
    worksheet.write(row,column,ensemble_name)   
    column = 1
    worksheet.write(row,column,'Validation') 
    
    # Load pre-trained transformers
    transformer_paths = [os.path.join(net_path, f"Conf_{c}.bin") for c in [c_t1, c_t2, c_t3]]
    transformers = [VisionTransformer(get_config(*parameters_config(c)), 128, zero_head=True, num_classes=1).load_state_dict(torch.load(path, map_location=device)) for path, c in zip(transformer_paths, comb)]
    ensemble = TransformerEnsemble(*transformers).to(device)
    optimizer = optim.Adam(ensemble.parameters(), lr=1e-4)
    
    for k in tqdm(range(cv)): # k-fold CV
    
        logger.info("Cross-validation fold #" + str(k+1))
        save_path = os.path.join(output_path,"cv_"+str(k+1)+".bin")
        
        results[f"CV {k+1}"] = {} # results of configuration comb at this cv

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

      
        
        dset_loaders = {'train': train_loader, 'val': valid_loader}
        
        val_loss_array, train_loss_array, val_accuracy_array, train_accuracy_array, aucs = [], [], [], [], []
        best_spec, best_sens, best_bacc, best_auc, best_aupr, best_f2 = 0,0,0,0,0,0    

        
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
                    
                    inputs,labels = data[0].float().to(device), data[1].float().to(device)                    
                    optimizer.zero_grad() 
                    outputs = ensemble(inputs)
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                                    
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
                    
        
        metrics_dict = {
            'Specificity': best_spec,
            'Sensitivity': best_sens,
            'Balanced Accuracy': best_bacc,
            'AUROC': best_auc,
            'AUPRC': best_aupr,
            'F2-score': best_f2,
            'CSP': csp,
            'CSE': cse,
            'BSNC': bsnc,
            'BSPC': bspc,
            'BS': bs}

        for metric, value in metrics_dict.items():
            results[f"CV{k+1}"][metric] = value

    for metric in metrics:
        mean_metric = np.mean([results["CV"+ str(k+1)][metric] for k in range(cv)])
        std_metric = np.std([results["CV"+ str(k+1)][metric] for k in range(cv)])
        worksheet.write(row,column,str(round(mean_metric,3))+" "+"("+str(round(std_metric,3))+")")  
        column +=1

    row += 2
    column = 0
    
workbook.close()

    
