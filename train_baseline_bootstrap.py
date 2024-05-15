# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti (based on https://github.com/jeonsworld/ViT-pytorch)
"""


# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import numpy as np
from datetime import timedelta
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_bootstrap import get_loader
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score,fbeta_score, average_precision_score
from sklearn.utils import class_weight
import random

Nrep = 100 # number of bootstrapping repetitions

logger = logging.getLogger(__name__)

results_test = {}
metrics = ['Specificity', 'Sensitivity', 'Balanced Accuracy', 'AUROC', 'AUPRC', 'F2-score','CSP', 'CSE', 'BSNC', 'BSPC', 'BS']

for metric in metrics:
    results_test[metric]=[]

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
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 1

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    
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
            output = model(x)[0] #logits
            features_v = model(x)[2][:,0].squeeze() #vettore delle features
            
            p = torch.sigmoid(output) #probability
            predicted = 1 * (p > 0.5)

            true_labels.append(y.item())
            predicted_labels.append(predicted)    
            class_probabilities.append(p)
            features_vectors.append(np.array(features_v))
            
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    
   
    class_probabilities = [i.item() for i in class_probabilities]

    b_accuracy = balanced_accuracy_score(true_labels, predicted_labels) # Balanced accuracy
    specificity = recall_score(true_labels, predicted_labels,pos_label = 0) #Specificity
    sensitivity = recall_score(true_labels, predicted_labels) #Sensitivity
    roc_auc = roc_auc_score(true_labels,class_probabilities) # Getting ROC AUC
    f2_score = fbeta_score(true_labels, predicted_labels, beta = 2)
    ap_score = average_precision_score(true_labels,class_probabilities)

    writer.add_scalar("test/accuracy", scalar_value=b_accuracy, global_step=global_step)
    return specificity, sensitivity, b_accuracy, roc_auc, f2_score, ap_score,true_labels, predicted_labels, class_probabilities


def train(args, model, boot):
    """ Train the model """
  
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

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
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
   
    
    global_step = 0
    model.zero_grad()
    best_spec, best_sens, best_acc, best_auc,best_f2,best_ap = 0,0,0,0,0,0
    best_choice = False
    
    def save_model(args, model):
        model_to_save = model.module if hasattr(model, 'module') else model
        model_checkpoint = os.path.join(args.output_dir, "Bootstrap_"+str(boot)+".bin")
        torch.save(model_to_save.state_dict(), model_checkpoint)
        logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    
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

                save_model(args, model)
        
                return best_spec, best_sens, best_acc, best_auc, best_f2, best_ap,tl,pl,cp
                    
    logger.info("***** Running Bootstrapping "+str(boot)+" *****")


    while True: 
        
        # Prepare dataset
        train_loader, test_loader = get_loader(args, boot)

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
            
            weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.numpy())
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
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    specificity, sensitivity, b_accuracy, roc_auc, f2_score, ap_score, true_labels, predicted_labels, class_probabilities= valid(args, model, writer, test_loader, global_step)
                    
                    logger.info("ROC AUC: \t%f" % roc_auc)

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
    
    if args.local_rank in [-1, 0]:
        writer.close()
    
    return best_spec, best_sens, best_acc, best_auc, best_ap, best_f2, tl,pl,cp
        


def main(boot):
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["prostateX"], default="prostateX",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "EvaViT"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--output_dir", type=str,
                        help="The output directory where checkpoints will be written.")
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
    parser.add_argument("--num_steps", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()
    
        
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training 100 bootstrapping
    best_spec, best_sens, best_acc, best_auc, best_ap, best_f2, tl, pl,cp = train(args, model, boot)
        
    return  best_spec, best_sens, best_acc, best_auc, best_ap, best_f2, tl, pl,cp

if __name__ == "__main__":
    
    for boot in range(Nrep):
        best_spec, best_sens, best_acc, best_auc, best_ap, best_f2, tl, pl,cp = main(boot)
        


