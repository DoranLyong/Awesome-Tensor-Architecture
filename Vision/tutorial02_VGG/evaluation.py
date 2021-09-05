# (ref) https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/evaluate.py

import argparse 
import logging 
import random 
import os
import os.path as osp 


import numpy as np
import torch
import torch.nn as nn

import utils
import dataloader.SIGNSloader as SIGNSloader
import models.net as net 

from test import eval

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
parser.add_argument('--restore_file', default='best', help="Optional, \
                                                    name of the file in --hyperparam_dir containing weights to reload before training")






#%% 
if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from CLI 

    # ===== Load the parameters from json file
    jsonPath = os.path.join(args.hyperparam_dir, 'params.json') 
    assert osp.isfile(jsonPath), f"No json configuration file found @ {jsonPath}"

    params = utils.Params(jsonPath)


    # ===== Set device 
    gpu_no = params.GPU_no
    device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    params.device = device  # update attribute by @property 
    params.cuda = torch.cuda.is_available() 
    print(f"device: {device}")        




    # ===== Set the logger 
    utils.set_logger(osp.join(args.hyperparam_dir, 'evaluate.log'))


    # ====== Create the input data pipeline
    logging.info("Loading the test dataset...")

    dataloaders = SIGNSloader.fetch_dataloader(  ['test'], 
                                                args.data_dir,
                                                params,
                                            )

    test_dl = dataloaders['test']           
    logging.info("- done.")                   


    # ===== Define the model     
    model = net.Net(params).to(device)         
    model = torch.nn.DataParallel(model, device_ids=None) 

    criterion = nn.CrossEntropyLoss()   

    logging.info("Starting evaluation")


    # Reload weights from the saved file
    utils.load_checkpoint(  checkpoint_path = osp.join(args.hyperparam_dir, f"{args.restore_file}.pth.tar"), 
                            model = model,
                        )

    # ===== Evaluate         
    test_loss, test_acc = eval(params, test_dl, model, criterion)              


    # ===== Save the evaluation
    save_path = osp.join(args.hyperparam_dir, f"metrics_test_{args.restore_file}.json")

    metrics = dict() 
    metrics["accuracy"] = test_acc 
    metrics["loss"] = test_loss

    utils.save_dict2json(metrics, save_path)

