# (ref) https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/train.py
import argparse 
import logging 
import os
import os.path as osp 
import random

import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 

import utils
import dataloader.SIGNSloader as SIGNSloader
import models.net as net 
from train import train_epoch
from val import val_epoch 


# ============ #
# Reproducible #
# ============ # 
""" 
    (ref) https://hoya012.github.io/blog/reproducible_pytorch/
    (ref) https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
"""
seed = 42

os.environ["PYTHONHASHSEED"] = str(seed)    # set PYTHONHASHSEED env var at fixed value
                                            # (ref) https://dacon.io/codeshare/2363
                                            # (ref) https://www.programmersought.com/article/16615747131/
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if use multi-GPU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# =============== # 
# Argument by CLI # 
# =============== # 
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")  # to read 
parser.add_argument('--hyperparam_dir', default='experiments/base_model', help="Directory containing params.json")  # hyperparameters
parser.add_argument('--restore_file', default=None, help="Optional, \
                                                    name of the file in --hyperparam_dir containing weights to reload before training")




#%% 
if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from CLI 
    
    
    # ===== Load the parameters from json file
    jsonPath = os.path.join(args.hyperparam_dir, 'params.json') 
    assert osp.isfile(jsonPath), f"No json configuration file found @ {jsonPath}"

    params = utils.Params(jsonPath)
#    print(params.__dict__)
#    print(params.dict['learning_rate'])


    # ===== Set device 
    gpu_no = params.GPU_no
    device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    params.device = device  # update attribute by @property 
    params.cuda = torch.cuda.is_available() 
    print(f"device: {device}")    
    

    # ===== Set the logger 
    utils.set_logger(osp.join(args.hyperparam_dir, 'train.log'))


    # ===== Create the input data pipeline 
    logging.info("Loading the datasets...")

    
    # ===== fetch dataloaders
    dataloaders = SIGNSloader.fetch_dataloader(  ['train', 'val'], 
                                                args.data_dir,
                                                params,
                                            )
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    
    logging.info("- done.")



    # ====== Log for train & val 
    train_logger = utils.Logger(  osp.join(args.hyperparam_dir, "train_epoch.log"),
                                    ['epoch', 'loss', 'acc', 'lr'],
                                )

    train_batch_logger = utils.Logger(  osp.join(args.hyperparam_dir, "train_batch.log"),
                                        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'],
                                    )                                

    val_logger = utils.Logger(  osp.join(args.hyperparam_dir, "val_epoch.log"),
                                ['epoch', 'loss', 'acc'],
                            )



    # ===== Define the model & loss, and optimizer & scheduler 
    model = net.Net(params).to(device)
    model = torch.nn.DataParallel(model, device_ids=None) 

    criterion = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=5, verbose=True) # (ref) https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau


    # ===== Run Training & Validation 
    logging.info(f"Starting training for {params.num_epochs} epoch(s) with batch_size={params.batch_size}")


    if args.restore_file is not None:
        restore_path = osp.join(args.hyperparam_dir, f"{args.restore_file}.pth.tar")
        logging.info(f"Restoring parameters from {restore_path}")
        utils.load_checkpoint(restore_path, model, optimizer)


    best_val_acc = 0.0
    metrics = dict()


    for epoch in range(params.num_epochs):

        train_epoch(params, epoch, train_dl, model, criterion, optimizer, train_logger, train_batch_logger)
        
        
        # ===== Evaluate for one epoch on validation set
        val_loss, val_acc = val_epoch(params, epoch, val_dl, model, criterion, val_logger)
        is_best = val_acc >= best_val_acc


        # ===== Scheduler Update
        #       After each epoch, do 'scheduler.step'. Note in this scheduler we need to send in loss for that epoch!    
        #       (ref) https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau

        scheduler.step(val_loss) # Reducing lr based on validation loss ; (ref) https://stackoverflow.com/questions/57375240/can-reducelronplateau-scheduler-in-pytorch-use-test-set-metric-for-decreasing-le
                                        # (ref) https://discuss.pytorch.org/t/what-does-scheduler-step-do/47764/2      
                                        # (ref) https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/


        # ===== Save weights        
        logging.info("- Save weights")

        state_dict = {  'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                    }

        utils.save_checkpoint(state_dict, is_best, args.hyperparam_dir)


        # ===== If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the directory
            best_json_path = osp.join(args.hyperparam_dir, "metrics_val_best_weights.json")

            metrics["accuracy"] = best_val_acc
            metrics["loss"] = val_loss

            utils.save_dict2json(metrics, best_json_path)


        # ===== Save latest val metrics in a json file in the model directory
        last_json_path = osp.join(args.hyperparam_dir, "metrics_val_last_weights.json")

        metrics["accuracy"] = best_val_acc
        metrics["loss"] = val_loss        
        utils.save_dict2json(metrics, last_json_path)
                                                