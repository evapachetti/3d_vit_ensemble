# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti (based on https://github.com/jeonsworld/ViT-pytorch)
"""


# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import numpy as np # type: ignore
import torch # type: ignore
from tqdm import tqdm # type: ignore
from models.modeling import VisionTransformer
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_bootstrap import get_loader
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score,fbeta_score, average_precision_score, brier_score_loss # type: ignore
from sklearn.utils import class_weight # type: ignore
import random
from tools import set_seed, parameters_config, get_config, calculate_confidence_metrics, brier_score_one_class

logging.basicConfig(level=logging.INFO)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
        
def setup(args):
    # Prepare model
    assert args.config < 19
    assert args.config > 1 # Only configs between 1 and 18 are defined
    model = VisionTransformer(get_config(*parameters_config(args.config)), args.img_size, zero_head=True, num_classes=args.num_classes)
    model.to(args.device)
    num_params = count_parameters(model)
    print(num_params)
    return args, model



def valid(args, model, test_loader, global_step):
    # Validation!
    
    predicted_labels, true_labels, class_probabilities, features_vectors = [], [], [], []

    eval_losses = AverageMeter()

    model.eval()
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        y = y.float()
        with torch.no_grad():
            output = model(x)[0] #logits
            features_v = model(x)[2][:,0].squeeze() #vettore delle features
            
            p = torch.sigmoid(output) #probability
            predicted = 1 * (p > 0.5)

            true_labels.append(y.cpu().item())
            predicted_labels.append(predicted.cpu().item())    
            class_probabilities.append(p)
            features_vectors.append(features_v.cpu().numpy())
            
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    
   
    class_probabilities = [i.item() for i in class_probabilities]

    b_accuracy = balanced_accuracy_score(true_labels, predicted_labels) # Balanced accuracy
    specificity = recall_score(true_labels, predicted_labels,pos_label = 0) #Specificity
    sensitivity = recall_score(true_labels, predicted_labels) #Sensitivity
    roc_auc = roc_auc_score(true_labels,class_probabilities) # Getting ROC AUC
    f2_score = fbeta_score(true_labels, predicted_labels, beta = 2)
    ap_score = average_precision_score(true_labels,class_probabilities)

    return specificity, sensitivity, b_accuracy, roc_auc, f2_score, ap_score,true_labels, predicted_labels, class_probabilities


def train(args, model, boot):
    """ Train the model """

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

  
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
   
    
    global_step = 0
    model.zero_grad()
    best_spec, best_sens, best_acc, best_auc,best_f2,best_ap = 0,0,0,0,0,0
    best_choice = False
    
    def save_model(args, model, boot):
        model_to_save = model.module if hasattr(model, 'module') else model
        save_conf_dir = os.path.join(args.output_dir, "boot_baseline_models", f"conf{args.config}")
        os.makedirs(save_conf_dir, exist_ok=True)
        model_checkpoint = os.path.join(save_conf_dir, f"boot{boot+1}.bin")
        torch.save(model_to_save.state_dict(), model_checkpoint)
        logging.info("Saved model checkpoint to [DIR: %s]", save_conf_dir)
    
    def save_best_metrics(args,model,specificity,sensitivity,b_accuracy,roc_auc,f2_score,ap_score,true_labels, predicted_labels, class_probabilities):
                best_spec = specificity
                best_sens = sensitivity
                best_acc = b_accuracy
                best_auc = roc_auc
                best_f2 = f2_score
                best_ap = ap_score
                tl = true_labels
                pl = predicted_labels
                cp = class_probabilities

                save_model(args, model, boot)
        
                return best_spec, best_sens, best_acc, best_auc, best_f2, best_ap,tl,pl,cp
                    
    logging.info(f"***** Running Bootstrapping {boot+1} *****")


    while True: 
        
        # Prepare dataset
        train_loader, test_loader = get_loader(args, boot)

        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            y = y.float()
            
            weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y.cpu().numpy()), y=y.cpu().numpy())
            weights = torch.tensor(weights[1] if len(weights) > 1 else weights[0])

            loss = model(x, y, weights)
           
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
           
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                    
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
             
                if global_step % args.eval_every == 0:
                    specificity, sensitivity, b_accuracy, roc_auc, f2_score, ap_score, true_labels, predicted_labels, class_probabilities= valid(args, model, test_loader, global_step)
                    
                    logging.info(f"AUROC: {roc_auc}")

                    # Custom decision process to ensure both spec and sens > 0.6; otherwise, look at AUROC alone
                    if specificity > 0.6 and sensitivity > 0.6:
                        if not best_choice or roc_auc > best_auc:
                            best_spec, best_sens, best_acc, best_auc, best_f2, best_ap, tl, pl, cp = save_best_metrics(
                                args, model, specificity, sensitivity, b_accuracy, roc_auc, f2_score, ap_score, true_labels, predicted_labels, class_probabilities
                            )
                            best_choice = True
                    else:
                        if not best_choice and roc_auc > best_auc:
                            best_spec, best_sens, best_acc, best_auc, best_f2, best_ap, tl, pl, cp = save_best_metrics(
                                args, model, specificity, sensitivity, b_accuracy, roc_auc, f2_score, ap_score, true_labels, predicted_labels, class_probabilities)
                        model.train()                    
                
                if global_step % t_total == 0:
                    break
              
        losses.reset()
    
        if global_step % t_total == 0:
            break
  
    return best_spec, best_sens, best_acc, best_auc, best_ap, best_f2, tl,pl,cp
        


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name",default="prostateX",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["prostateX"], default="prostateX",
                        help="Which downstream task.")
    parser.add_argument("--config", type=int,
                        default=5, help="Which configuration to use.")
    parser.add_argument("--num_classes", type=int,
                        default=1, help="Number of output classes.")
    parser.add_argument("--num_rep", type=int,
                        default=100, help="Number of bootstrapping repetitions.")
    parser.add_argument("--output_dir", type=str, default = os.path.join(os.getcwd(),"output"),
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--csv_path",  default=os.path.join(os.getcwd(), "csv_files", "fixed_split"),
                        help="Path where csv files are stored.")
    parser.add_argument("--img_size", default=128, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=24, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Device to compute operations.")
    args = parser.parse_args()

    # Define results dictionary
    results = {}

    for boot in range(args.num_rep):

        # Set seed
        set_seed(args)

        # Model & Tokenizer Setup
        args, model = setup(args)

        best_spec, best_sens, best_acc, best_auc, best_aupr, best_f2, tl, pl,cp = train(args, model, boot)

        # Confidence metrics
        csp,cse = calculate_confidence_metrics(tl, pl, cp) 
        bs = brier_score_loss(tl, cp)
        bsnc = brier_score_one_class(tl, cp, cl = 0)
        bspc = brier_score_one_class(tl, cp, cl = 1)
    
        metrics_dict = {
            'Specificity': best_spec,
            'Sensitivity': best_sens,
            'Accuracy': best_acc,
            'AUROC': best_auc,
            'AUPRC': best_aupr,
            'F2-score': best_f2,
            'CSP': csp,
            'CSE': cse,
            'BSNC': bsnc,
            'BSPC': bspc,
            'BS': bs}
        
        for metric, value in metrics_dict.items():
            results[f"BOOT {boot+1}"] = {}
            results[f"BOOT {boot+1}"][metric] = value

    for boot in range(args.num_rep):
        logging.info(f"--- BOOTSTRAPPING # {boot+1} ---")        
        for metric, value in results[f"BOOT {boot+1}"].items():
            logging.info(f"{metric}: {value}")        

if __name__ == "__main__": main()        


