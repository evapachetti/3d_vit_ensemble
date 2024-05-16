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
from create_dataset import ProstateDataset, ToTensorDataset
import logging
import argparse


# Set reproducibility seed
set_seed()

# Set up logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)


def train_ensemble(args):

    best_choice = False

    # Ensemble to re-train on the whole dataset
    comb = args.ensemble_conf_list

    # Define paths
    net_path = os.path.join(args.output_path, "baseline_models")  # directory where trained baseline models are stored

    # Define criterion
    criterion = nn.BCELoss()

    # Training loop
    results = {} #results of each combination in k-fold CV
    c_t1, c_t2, c_t3 = comb
    ensemble_name = f"{c_t1}_{c_t2}_{c_t3}"

    # Load pre-trained transformers
    transformer_paths = [os.path.join(net_path, f"Conf_{c}.bin") for c in [c_t1, c_t2, c_t3]]
    transformers = [VisionTransformer(get_config(*parameters_config(c)), 128, zero_head=True, num_classes=1).load_state_dict(torch.load(path, map_location=args.device)) for path, c in zip(transformer_paths, comb)]
    ensemble = TransformerEnsemble(*transformers).to(args.device)
    optimizer = optim.Adam(ensemble.parameters(), lr=1e-4)
    
    save_path = os.path.join(args.output_path,"best_ensemble_model",f"ensemble_{ensemble_name}")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path,f"best_ensemble_model.bin")
    
    results = {} # results of configuration comb at this cv

    csv_file_train = os.path.join(args.csv_path,f"training.csv") # training k-cv csv
    csv_file_val = os.path.join(args.csv_path,f"validation.csv") # validation k-cv csv
    
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
                            batch_size=args.train_batch_size,
                            num_workers=0,
                            pin_memory=True)
    valid_loader = DataLoader(validset,
                            sampler=valid_sampler,
                            batch_size=args.eval_batch_size,
                            num_workers=0,
                            pin_memory=True) if validset is not None else None


    
    dset_loaders = {'train': train_loader, 'val': valid_loader}
    
    val_loss_array, train_loss_array, val_accuracy_array, train_accuracy_array, aucs = [], [], [], [], []
    best_spec, best_sens, best_bacc, best_auc, best_aupr, best_f2 = 0,0,0,0,0,0    

    
    for epoch in range(args.num_epochs): 

        predicted_labels, true_labels, class_probabilities = [],[],[]
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                ensemble.train(True)
            else:
                ensemble.eval()
                
            running_loss, running_corrects = 0.0, 0.0
    
            for data in dset_loaders[phase]:
                
                inputs,labels = data[0].float().to(args.device), data[1].float().to(args.device)                    
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
        results[metric] = value

    return results

def main():
    # Define a custom argument type for a list of integers
    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", required=True, default=100,
                        help="Number of folds in cross validation.")
    parser.add_argument("--num_cv", required=True, default = 5,
                        help="Number of folds in cross validation.")
    parser.add_argument("--ensemble_conf_list", required=True, type=list_of_ints,
                        default="5,9,11", help="Best ensemble to re-train.")
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

    results = train_ensemble(args)


if __name__ == "__main__":
    main()


    
