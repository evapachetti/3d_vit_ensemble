# coding=utf-8

"""
@author: Eva Pachetti (based on https://github.com/jeonsworld/ViT-pytorch)
"""

from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np # type: ignore
import torch # type: ignore
from tqdm import tqdm # type: ignore
from models.modeling import VisionTransformer
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score,fbeta_score # type: ignore
from tools import set_seed, parameters_config, get_config



logger = logging.getLogger(__name__)


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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_Best_model_Conf_5.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def save_best_metrics(args,model,specificity,sensitivity,accuracy,roc_auc,f2_score):
            best_spec = specificity
            best_sens = sensitivity
            best_acc = accuracy
            best_auc = roc_auc
            best_f2 = f2_score

            save_model(args, model)
            
            return best_spec, best_sens, best_acc, best_auc, best_f2
    


def setup(args):
    # Prepare model
    assert args.config < 19
    assert args.config > 1 # Only configs between 1 and 18 are defined
    model = VisionTransformer(get_config(*parameters_config(args.config)), args.img_size, zero_head=True, num_classes=args.num_classes)
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model



def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        


def valid(args, model, test_loader):
    
    predicted_labels, true_labels, class_probabilities, features_vectors = [], [], [], []
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        y = y.float()
        with torch.no_grad():
            output = model(x)[0] 
            features_v = model(x)[2][:,0].squeeze()
            p = torch.sigmoid(output) 
            predicted = 1 * (p > 0.5)

            true_labels.append(y.item())
            predicted_labels.append(predicted)    
            class_probabilities.append(p)
            features_vectors.append(np.array(features_v))

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    
   
    class_probabilities = [i.item() for i in class_probabilities]

    accuracy = accuracy_score(true_labels, predicted_labels) # Balanced accuracy
    specificity = recall_score(true_labels, predicted_labels,pos_label = 0) #Specificity
    sensitivity = recall_score(true_labels, predicted_labels) #Sensitivity
    roc_auc = roc_auc_score(true_labels,class_probabilities) # Getting ROC AUC
    f2_score = fbeta_score(true_labels, predicted_labels, beta = 2)


    return specificity, sensitivity, accuracy, roc_auc, f2_score, true_labels, class_probabilities, features_vectors


def train(args, model):
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

  
    # Train!
    logger.info("***** Running training *****")
   
    set_seed(args) 
    losses = AverageMeter()
    best_choice = False
    global_step = 0
    model.zero_grad()
    best_spec, best_sens, best_acc, best_auc, best_f2 = 0,0,0,0,0

    while True: 
    
        train_loader, test_loader = get_loader(args)

        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            y = y.float()
            
            weights = None # weight for al the training set

            loss = model(x, y,weights)
         
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
              
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    specificity, sensitivity, accuracy, roc_auc, f2_score, true_labels, class_probabilities, features_vectors = valid(args, model, test_loader)
                    logger.info("AUC: \t%f" % roc_auc)
                                        
                        # Custom decision process to ensure both spec and sens > 0.6; otherwise, look at AUROC alone
                    if specificity > 0.6 and sensitivity > 0.6:
                        if not best_choice or roc_auc > best_auc:
                            best_spec, best_sens, best_acc, best_auc, best_f2 = save_best_metrics(
                                args,model,specificity,sensitivity,accuracy,roc_auc,f2_score
                            )
                            best_choice = True
                    else:
                        if not best_choice and roc_auc > best_auc:
                            best_spec, best_sens, best_acc, best_auc, best_f2 = save_best_metrics(
                                args,model,specificity,sensitivity,accuracy,roc_auc,f2_score)
                        model.train()                    
                   
                    model.train()                    
                
                if global_step % t_total == 0:
                    break
             
                  
        losses.reset()
    
        if global_step % t_total == 0:
            break

    
    return best_spec, best_sens, best_acc, best_auc, best_f2



def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["prostateX","Careggi"], default="prostateX",
                        help="Which downstream task.")
    parser.add_argument("--config", type=int,
                        default=5, help="Which configuration to use.")
    parser.add_argument("--num_classes", type=int,
                        default=1, help="Number of output classes.")
    parser.add_argument("--output_dir", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--csv_path",  required=True, default=os.path.join(os.getcwd(), "csv_files", "fixed_split"),
                        help="Path where csv files are stored.")
    parser.add_argument("--img_size", default=128, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=31, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
   
    args = parser.parse_args()

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    best_spec, best_sens, best_acc, best_auc, best_f2 = train(args, model)
    results = {
                'Specificity': best_spec,
                'Sensitivity': best_sens,
                'Accuracy': best_acc,
                'AUROC': best_auc,
                'F2-score': best_f2}
    
    return results

if __name__ == "__main__":
    main()
