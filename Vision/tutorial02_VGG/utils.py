import os
import os.path as osp 
from pathlib import Path 
import logging 
import json 
import csv
import shutil 
from typing import Optional

import torch 



#%% 
class Params(object):
    """ Class that loads hyperparameters from a json file.

        ===== usage =====
        params = Params(json_path)
        print(params.learning_rate)
        params.learning_rate = 0.5   # change the value of learning_rate in params
    """

    def __init__(self, json_path:str):

        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)    # dict update 
                                            # (ref) https://stackoverflow.com/questions/57082774/what-does-self-dict-updatefields-do?noredirect=1&lq=1


    def save(self, json_path:str):

        with open(json_path, 'w') as f: 
            json.dump(self.__dict__, f , indent=4) # indent; (ref) https://psychoria.tistory.com/703

    def update(self, json_path:str):
        # ===== Load parameters from json file 
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)


    @property 
    def dict(self):
        # @property decorator; (ref) https://www.programiz.com/python-programming/property
        #                      (ref) https://www.programiz.com/python-programming/methods/built-in/property
        """ Gives dict-like access to Params instance 
            
            ===== usage =====
            params.dict['learning_rate']
        """
        return self.__dict__

#%% 
class RunningAverage(object):
    """ A simple class that maintains the running average of a quantity

        ==== usage ====
        loss_avg = RunningAverage()
        loss_avg.update(2)
        loss_avg.update(4)
        loss_avg() -> 3
    """
    def __init__(self):
        self.steps = 0
        self.total = 0 

    def update(self, val):
        self.total += val 
        self.steps += 1 
    
    def __call__(self) -> float:
        # get the average 
        return self.total/float(self.steps)


#%% 
class Logger(object):
    def __init__(self, path:str, header:list):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del__(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()





#%% 
def calc_accuracy(model_scores:torch.Tensor, targets:torch.Tensor) -> float :
    batch_size = targets.shape[0]

    top_val, top_index = model_scores.topk(k=1, dim=1, largest=True) # return top-k ; (ref) https://pytorch.org/docs/stable/generated/torch.topk.html
    
    pred = top_index.t()    # transpose ; (ref) https://pytorch.org/docs/stable/generated/torch.t.html
                            # [batch_size, 1] -> [1, batch_size]

    check_correct = pred.eq(targets.view(1, -1))    # view() vs. reshape() ; (ref) https://sdr1982.tistory.com/317
                                                    # torch.eq()  ; (ref) https://pytorch.org/docs/stable/generated/torch.eq.html


    n_correct_elems = check_correct.float().sum().item()# convert from bool to float 
                                                        # sum of them 
                                                        # get the value 

    return n_correct_elems / batch_size                                                          



#%% 
def set_logger(log_path:str):
    """ Set the logger to log info in terminal and file `log_path`.
        In general, it is useful to have a logger so that every output to the terminal is saved in a permanent file. 
        Here we save it to `model_dir/train.log`.

        ===== usage ===== 
        logging.info("Starting training...")
    """
    # (ref) https://greeksharifa.github.io/%ED%8C%8C%EC%9D%B4%EC%8D%AC/2019/12/13/logging/
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Check handler exists  
    # (ref) https://5kyc1ad.tistory.com/269
    if not logger.handlers:
        # ===== Logging to a file 
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # ===== Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


#%% 
def save_dict2json(d:dict, json_path:str):
    """ to save dict of floats in json file 

        d: dict of float-castable values (np.float, int, float, etc.)
        json_path : path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)



#%% 
def save_checkpoint(state:dict, is_best:bool, checkpoint_dir:str):
    """ Saves model and training parameters at checkpoint_path + 'last.pth.tar'. 
        If is_best==True, also saves checkpoint_path + 'best.pth.tar' 

        Args: 
            - state: contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
            - is_best: True if it is the best model seen till now
            - checkpoint_dir: folder where parameters are to be saved
    """
    filePath = osp.join(checkpoint_dir, 'last.pth.tar')

    # ===== Make a directory to save 
    checkpoint_saveDIR = Path(checkpoint_dir)
    checkpoint_saveDIR.mkdir(parents=True, exist_ok=True) 


    # ===== Save 
    torch.save(state, filePath)

    if is_best:
        shutil.copyfile(filePath, osp.join(checkpoint_dir, 'best.pth.tar'))



#%% 
def load_checkpoint(checkpoint_path:str, model:torch.nn.Module, optimizer:Optional[torch.optim.Optimizer] = None):
    """ Loads model parameters (state_dict) from file_path. 
        If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.

        Args: 
            - checkpoint_path: file_path which needs to be loaded
            - model: model for which the parameters are loaded
            - optimizer: resume optimizer from checkpoint
    """
    assert osp.exists(checkpoint_path), f"Checkpoint file doesn't exist @ {checkpoint_path}" 

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict']) # (ref) https://pytorch.org/docs/stable/optim.html

    return checkpoint