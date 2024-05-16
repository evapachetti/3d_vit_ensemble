
# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

from __future__ import absolute_import, division, print_function
import torch # type: ignore
import torch.nn as nn # type: ignore
import torchvision.transforms as transforms # type: ignore
from sklearn.metrics import recall_score, roc_auc_score, fbeta_score, balanced_accuracy_score, average_precision_score # type: ignore
from models.modeling import VisionTransformer, TransformerEnsemble
import numpy as np # type: ignore
from matplotlib import pyplot as plt # type: ignore
import os
from tools import set_seed, normalize, parameters_config, get_config, calculate_confidence_metrics, brier_score_one_class, testing_model
from sklearn.metrics import brier_score_loss # type: ignore
import pandas as pd # type: ignore
from create_dataset import ProstateDataset, ToTensorDataset
from itertools import combinations
from tqdm import tqdm # type: ignore
import argparse


def test_baseline(args):
    
    model = VisionTransformer(get_config(*parameters_config(args.conf)),args.image_size, zero_head=False, num_classes=1, vis=True)

    # Initialize results dictionaries
    res_base = {}
   
    for k in tqdm(range(args.cv)):
        # Load model
        model_path = os.path.join(args.base_path, f"conf_{args.conf}_cv_{args.cv}.bin")
        model.load_state_dict(torch.load(model_path))
        model.to(args.device)
        model.eval()

        res_base[f'CV {k}'] = {}
        
        # Dataset loading
        csv_file_train = os.path.join(args.csv_path, f"training_cv{k+1}.csv")
        csv_file_val = os.path.join(args.csv_path, f"validation_cv{k+1}.csv")
        validset = ProstateDataset(csv_file_val)

        # Normalize data
        volumes_train = [i[0] for i in list(ProstateDataset(csv_file_train))]
        mean = np.mean(volumes_train)
        testset = normalize(testset, mean)
        validset = normalize(validset, mean)

        testset_tf = ToTensorDataset(testset, transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset_tf, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

        # Testing
        true_labels, predicted_labels, class_probabilities, _ = testing_model(testloader, model, args.device)
        b_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
        specificity = recall_score(true_labels, predicted_labels, pos_label=0)
        sensitivity = recall_score(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, class_probabilities)
        f2_score = fbeta_score(true_labels, predicted_labels, beta=2)
        ap_score = average_precision_score(true_labels, class_probabilities)
        brier_score = brier_score_loss(true_labels, class_probabilities)
        bs_0 = brier_score_one_class(true_labels, class_probabilities, cl=0)
        bs_1 = brier_score_one_class(true_labels, class_probabilities, cl=1)
        csp, cse = calculate_confidence_metrics(true_labels, predicted_labels, class_probabilities)
    
    # Save results

    res_base[f'CV {k}']['Specificity'] = specificity
    res_base[f'CV {k}']['Sensitivity'] = sensitivity
    res_base[f'CV {k}']['Balanced Accuracy'] = b_accuracy
    res_base[f'CV {k}']['AUROC'] = roc_auc
    res_base[f'CV {k}']['AUPRC'] = ap_score
    res_base[f'CV {k}']['F2-score'] = f2_score
    res_base[f'CV {k}']['Brier score'] = brier_score
    res_base[f'CV {k}']['BS_0'] = bs_0
    res_base[f'CV {k}']['BS_1'] = bs_1    
    res_base[f'CV {k}']['CSP'] = csp    
    res_base[f'CV {k}']['CSE'] = cse

    median_res_base = {}
    mean_res_base = {}
    percents_2_5_base = {}
    percents_97_5_base = {}
    percents_25_base = {}
    percents_75_base = {}

    for k in res_base['CV 0'].keys():
        values = [res_base[f'CV {s}'][k] for s in range(args.cv)]
        median_res_base[k] = np.median(values)
        mean_res_base[k] = np.mean(values)
        percents_2_5_base[k] = np.percentile(values, 2.5)
        percents_97_5_base[k] = np.percentile(values, 97.5)
        percents_25_base[k] = np.percentile(values, 25)
        percents_75_base[k] = np.percentile(values, 75)

    # Print results
    print("---Base ViT results (median and 90% CI)---\n")
    for key, value in median_res_base.items():
        ci_2_5 = percents_2_5_base[key]
        ci_97_5 = percents_97_5_base[key]
        print(f"{key}: {round(value, 3)} [{round(ci_2_5, 3)}-{round(ci_97_5, 3)}]")
    
    return res_base



def test_ensemble(args):

    assert args.max_configs <= 19 # We implemented up to 18 baseline configurations
    configurations = list(range(1,args.max_configs))
    combs = list(combinations(configurations, args.combinations))

    res_ens, median_res_ens, mean_res_ens, percents_2_5_ens, percents_97_5_ens, percents_25_ens, percents_75_ens = {},{},{},{},{},{},{}

    for comb in tqdm(combs): 
        for k in tqdm(range(args.cv)):
            
            ens_model = os.path.join(args.ens_dir,"conf_"+str(args.conf)+"cv_"+str(args.cv)+".bin")

            res_ens['CV '+str(k)] = {}
            
            csv_file_train = os.path.join(args.csv_path, f"training_cv{k+1}.csv")

            # Normalize data
            volumes_train = [i[0] for i in list(ProstateDataset(csv_file_train))]
            mean = np.mean(volumes_train)
            testset = normalize(testset, mean)
            testset_tf = ToTensorDataset(testset, transforms.ToTensor())

            testloader = torch.utils.data.DataLoader(testset_tf, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

            c_t1, c_t2, c_t3 = comb
            
            # Load pre-trained transformers and ensemble
            transformer_paths = [os.path.join(args.base_dir, f"Conf_{c}.bin") for c in [c_t1, c_t2, c_t3]]
            transformers = [VisionTransformer(get_config(*parameters_config(c)), 128, zero_head=True, num_classes=1).load_state_dict(torch.load(path, map_location=args.device)) for path, c in zip(transformer_paths, comb)]
            ensemble = TransformerEnsemble(*transformers).to(args.device)
            ensemble.load_state_dict(torch.load(ens_model, map_location=args.device))    
            ensemble.eval()
            
            true_labels, predicted_labels, class_probabilities, _ = testing_model(testloader, ensemble,args.device)
            
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
            
            res_ens[f'CV {k}']['Specificity'] = specificity
            res_ens[f'CV {k}']['Sensitivity'] = sensitivity
            res_ens[f'CV {k}']['Balanced Accuracy'] = b_accuracy
            res_ens[f'CV {k}']['AUROC'] = roc_auc
            res_ens[f'CV {k}']['AUPRC'] = ap_score
            res_ens[f'CV {k}']['F2-score'] = f2_score
            res_ens[f'CV {k}']['Brier score'] = brier_score
            res_ens[f'CV {k}']['BS_0'] = bs_0
            res_ens[f'CV {k}']['BS_1'] = bs_1    
            res_ens[f'CV {k}']['CSP'] = csp    
            res_ens[f'CV {k}']['CSE'] = cse
        

        for k in res_ens['CV 0'].keys():
            values = [res_ens[f'CV {s}'][k] for s in range(args.cv)]
            median_res_ens[k] = np.median(values)
            mean_res_ens[k] = np.mean(values)
            percents_2_5_ens[k] = np.percentile(values, 2.5)
            percents_97_5_ens[k] = np.percentile(values, 97.5)
            percents_25_ens[k] = np.percentile(values, 25)
            percents_75_ens[k] = np.percentile(values, 75)

        # Print results
        print("---Ensemble ViT results (median and 90% CI)---\n")
        for key, value in median_res_ens.items():
            ci_2_5 = percents_2_5_ens[key]
            ci_97_5 = percents_97_5_ens[key]
            print(f"{key}: {round(value, 3)} [{round(ci_2_5, 3)}-{round(ci_97_5, 3)}]")
        
        return res_ens
            
    

def compute_statistics(res_base, res_ens, args):

    from scipy import stats # type: ignore
    import seaborn as sns # type: ignore
    import textwrap

    import textwrap

    results = {}

    def wrap_labels(ax, width, break_long_words=False):
        labels = [textwrap.fill(label.get_text(), width=width, break_long_words=break_long_words) for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=0)

    auprcs_base = [res_base[k]['AUPRC'] for k in res_base]
    auprcs_ens = [res_ens[k]['AUPRC'] for k in res_ens]

    cse_base = [res_base[k]['CSE'] for k in res_base]
    cse_ens = [res_ens[k]['CSE'] for k in res_ens]

    bs_base = [res_base[k]['Brier score'] for k in res_base]
    bs_ens = [res_ens[k]['Brier score'] for k in res_ens]


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
    plt.savefig(os.path.join(args.output_path,"box_plot.eps",format='eps'))

    results['shapiro base'] = shapiro_test_base
    results['shapiro ens'] = shapiro_test_ens
    results['median'] = median_test
    results['mann_whitney'] = mw_test

    return results



def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--cv", required=True, default = 5,
                        help="Number of folds in cross validation.")
    parser.add_argument("--conf", required=True, default = 5,
                        help="Configuration number of baseline model.")
    parser.add_argument("--max_configs", required=True, default = 19,
                        help="Max number of baseline configurations consider.")
    parser.add_argument("--combinations", required=True, default = 3,
                        help="How many baseline combinations in ensemble consider.")
    parser.add_argument("--image_size",  required=True, default=128,
                        help="Image size.")
    parser.add_argument("--test_batch_size",  required=True, default=1,
                        help="Batch size for validation and test loaders.")
    parser.add_argument("--device",  required=True, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Device to compute operations.")
    parser.add_argument("--csv_path",  required=True, default=os.path.join(os.getcwd(), "csv_files", "cross_validation"),
                        help="Path where csv files are stored.")
    parser.add_argument("--output_path",  required=True, default=os.path.join(os.getcwd(), "output"),
                        help="Path where store results.")
    parser.add_argument("--base_path",  required=True, default=os.path.join(os.getcwd(), "output", "baseline_models"),
                        help="Path where baseline parameters are stored.")
    parser.add_argument("--ens_path",  required=True, default=os.path.join(os.getcwd(), "output", "ensemble_models"),
                        help="Path where ensemble parameters are stored.")
    parser.add_argument("--baseline",  action='store_true',
                        help="Whether compute test on baseline model.")
    parser.add_argument("--ensemble",  action='store_true',
                        help="Whether compute test on ensemble model.")
    args = parser.parse_args()

    if args.baseline:   
        res_base = test_baseline(args)
    
    if args.ensemble:
        res_ens = test_ensemble(args)

    if args.baseline and args.ensemble:
        stat_results = compute_statistics(res_base, res_ens, args)

    
    
if __name__ == "__main__":
    main()



