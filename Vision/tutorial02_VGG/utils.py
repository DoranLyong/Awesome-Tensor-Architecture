import os
import os.path as osp 
import logging 
import json 
import shutil 

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
            json.dump(self.__dict, f , indent=4) # indent; (ref) https://psychoria.tistory.com/703

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

