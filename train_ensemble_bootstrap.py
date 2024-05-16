
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
from tools import set_seed, normalize, parameters_config, get_config, calculate_confidence_metrics, brier_score_one_class, save_best_metrics, bootstrapping, testing_model
from tqdm import tqdm # type: ignore
from create_dataset import ProstateDataset, ToTensorDataset

import logging
import xlsxwriter # type: ignore
import argparse

# Set reproducibility seed
set_seed()

# Set up logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


def train_ensemble_bootstrap(args):

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
    configurations = range(1, args.max_configs)
    combs = list(combinations(configurations, args.combinations))

    # Define paths
    base_path = os.path.join(args.output_path, "baseline_models")  # directory where trained baseline models are stored

    # Define file paths
    csv_file_train = os.path.join(args.csv_path, "training.csv")  # training csv
    csv_file_val = os.path.join(args.csv_path, "validation.csv")  # validation csv
    csv_file_test = os.path.join(args.csv_path, "test.csv")  # test csv

    # Define criterion
    criterion = nn.BCELoss()

    # Training loop
    for comb in combs: #for each ensemble combination compute bootstrap

        logger.info("Ensemble combination #" + str(comb))
        
        results_val = {'Specificity':[], 'Sensitivity':[], 'Balanced Accuracy':[], 'AUROC':[], 'AUPRC':[], 'F2-score':[],'CSP':[], 'CSE':[], 'BSNC':[], 'BSPC':[], 'BS':[]} 
        results_test = {'Specificity':[], 'Sensitivity':[], 'Balanced Accuracy':[], 'AUROC':[], 'AUPRC':[], 'F2-score':[],'CSP':[], 'CSE':[], 'BSNC':[], 'BSPC':[], 'BS':[]} 
    
        c_t1, c_t2, c_t3 = comb
        ensemble_name = f"{c_t1}_{c_t2}_{c_t3}"

        worksheet.write(row,column,ensemble_name)   
        column = 1
        worksheet.write(row,column,'Validation') 
        
        # Load pre-trained transformers
        transformer_paths = [os.path.join(base_path, f"Conf_{c}.bin") for c in [c_t1, c_t2, c_t3]]
        transformers = [VisionTransformer(get_config(*parameters_config(c)), 128, zero_head=True, num_classes=1).load_state_dict(torch.load(path, map_location=args.device)) for path, c in zip(transformer_paths, comb)]
        ensemble = TransformerEnsemble(*transformers).to(args.device)
        optimizer = optim.Adam(ensemble.parameters(), lr=1e-4)
        
        
        for k in tqdm(range(args.num_rep)): 
            
            save_path = os.path.join(args.output_path,"bootstrap_ensemble_models",f"ensemble_{ensemble_name}")
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path,f"bootstrap_{k}.bin")

            boot_trainframe = bootstrapping(csv_file_train, k)
            trainset = list(ProstateDataset(boot_trainframe, bootstrap=True)) #bootstrapped train
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
                                    batch_size=args.train_batch_size,
                                    num_workers=0,
                                    pin_memory=True)
            valid_loader = DataLoader(validset,
                                    sampler=valid_sampler,
                                    batch_size=args.eval_batch_size,
                                    num_workers=0,
                                    pin_memory=True) if validset is not None else None
            
            testloader = torch.utils.data.DataLoader(testset_tf, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

            dset_loaders = {'train': train_loader, 'val': valid_loader}
            
            val_loss_array, train_loss_array, val_accuracy_array, train_accuracy_array, aucs = [], [], [], [], []
            best_spec, best_sens, best_bacc, best_auc,best_aupr, best_f2, tl,pl, cp  = 0,0,0,0,0,0,0,0,0

            for epoch in range(args.num_epochs): 
                predicted_labels, true_labels, class_probabilities = [],[],[]
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        ensemble.train()
                    else:
                        ensemble.eval()
                        
                    running_loss = 0.0
                    running_corrects = 0.0
            
                    for data in dset_loaders[phase]:
                        inputs,labels = data[0].float().to(args.device), data[1].float().to(args.device)
                        optimizer.zero_grad()      
                        outputs = ensemble(inputs)
                        outputs = torch.squeeze(outputs)
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
                        
            
            metrics_list = [
                ('Specificity', best_spec),
                ('Sensitivity', best_sens),
                ('Balanced Accuracy', best_bacc),
                ('AUROC', best_auc),
                ('AUPRC', best_aupr),
                ('F2-score', best_f2),
                ('CSP', csp),
                ('CSE', cse),
                ('BSNC', bsnc),
                ('BSPC', bspc),
                ('BS', bs)
            ]

            for key, value in metrics_list:
                results_val[key].append(value)
            
            ### HOLD-OUT-TEST ###
            true_labels_t, predicted_labels_t, class_probabilities_t, features_vectors = testing_model(testloader, ensemble,args.device)
            
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
            
            metrics_list_test = [
                ('Specificity', specificity),
                ('Sensitivity', sensitivity),
                ('Balanced Accuracy', b_accuracy),
                ('AUROC', roc_auc),
                ('AUPRC', pr_auc),
                ('F2-score', f2_score),
                ('CSP', csp),
                ('CSE', cse),
                ('BSNC', bsnc),
                ('BSPC', bspc),
                ('BS', bs)
            ]

            for key, value in metrics_list_test:
                results_test[key].append(value)
            

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
        
        row += 2
        column = 0


    workbook.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", required=True, default=100,
                        help="Number of folds in cross validation.")
    parser.add_argument("--num_rep", required=True, default = 5,
                        help="Number of repetitions in bootstrap.")
    parser.add_argument("--conf", required=True, default = 5,
                        help="Configuration number of baseline model.")
    parser.add_argument("--max_configs", required=True, default = 19,
                        help="Max number of baseline configurations consider.")
    parser.add_argument("--combinations", required=True, default = 3,
                        help="How many baseline combinations in ensemble consider.")
    parser.add_argument("--image_size",  required=True, default=128,
                        help="Image size.")
    parser.add_argument("--train_batch_size",  required=True, default=4,
                        help="Batch size for validation and test loaders.")
    parser.add_argument("--eval_batch_size",  required=True, default=1,
                        help="Batch size for validation and test loaders.")
    parser.add_argument("--device",  required=True, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Device to compute operations.")
    parser.add_argument("--csv_path",  required=True, default=os.path.join(os.getcwd(), "csv_files", "cross_validation"),
                        help="Path where csv files are stored.")
    parser.add_argument("--output_path",  required=True, default=os.path.join(os.getcwd(), "output"),
                        help="Path where store results.")
    args = parser.parse_args()

    train_ensemble_bootstrap(args)


if __name__ == "__main__":
    main()

    
