# (ref) https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/search_hyperparams.py

import argparse 
import sys 
import os 
import os.path as osp 
from pathlib import Path 
from subprocess import check_call  # (ref) https://newsight.tistory.com/354

import utils 



PYTHON = sys.executable # get python interpreter path 


# =============== # 
# Argument by CLI # 
# =============== # 
parser = argparse.ArgumentParser()
parser.add_argument('--hyperparam_dir', default='experiments/dropout_rate', help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")




#%%
def launch_training_job(hyperparam_dir:str, data_dir:str, job_name:str, params:utils.Params):
    """ Launch training of the model with a set of hyperparameters in hyperparam_dir/job_name

        Args:
            - model_dir: directory containing config, weights and log
            - data_dir: directory containing the dataset
            - params: (dict) containing hyperparameters
    """
    # ==== Create a new folder in hyperparam_dir with unique_name "job_name"
    model_dir = osp.join(hyperparam_dir, job_name)

    saveDIR = Path(model_dir)
    saveDIR.mkdir(parents=True, exist_ok=True) 


    # ===== Write parameters in json file
    json_path = osp.join(model_dir, 'params.json')
    params.save(json_path)


    # ===== Launch training with this config
    cmd = f"{PYTHON} main.py --hyperparam_dir={model_dir} --data_dir {data_dir}"

    print(cmd)
    check_call(cmd, shell=True)








#%%
if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from CLI 

    # ===== Load the parameters from json file
    jsonPath = os.path.join(args.hyperparam_dir, 'params.json') 
    assert osp.isfile(jsonPath), f"No json configuration file found @ {jsonPath}"    

    params = utils.Params(jsonPath)



    # ===== Perform hypersearch over one parameter
    dropout_rates = [0.5, 0.65, 0.8]

    for p_dropout in dropout_rates:

        # Modify the relevant parameter in params
        params.dropout_rate = p_dropout


        # Launch job (name has to be unique)
        job_name = f"dropout_rate_{p_dropout}"
        launch_training_job(args.hyperparam_dir, args.data_dir, job_name, params)
