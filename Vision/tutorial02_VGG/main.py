# (ref) https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/train.py
import argparse 
import logging 
import os
import os.path as osp 
import random

import numpy as np
from torch.utils.data import dataloader 
from tqdm import tqdm 
import torch 
import torch.optim as optim 

import utils
import dataloader.SIGNSloader as SIGNSloader



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
    gpu_no = 0
    device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
    params.device = device  # update attribute by @property 
    params.cuda = torch.cuda.is_available() 
    print(f"device: {device}")    
    

    # ===== Set the logger 
    utils.set_logger(osp.join(args.hyperparam_dir, 'train.log'))


    # ===== Crate the input data pipeline 
    logging.info("Loading the datasets...")

    
    # ===== fetch dataloaders
    dataloaders = SIGNSloader.fetch_dataloader(  ['train', 'val'], 
                                                args.data_dir,
                                                params,
                                            )
    train_dl = dataloaders['train']
#    val_dl = dataloaders['val']
    

    logging.info("- done.")


    
