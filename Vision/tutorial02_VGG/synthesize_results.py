# (ref) https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/synthesize_results.py
""" Aggregates results from the 'metrics_eval_best_weights.json' in a --parent_dir
"""
import argparse
import json
import os
import os.path as osp 

from tabulate import tabulate  # (ref) https://www.delftstack.com/ko/howto/python/data-in-table-format-python/




# =============== # 
# Argument by CLI # 
# =============== # 
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/dropout_rate', help='Directory containing results of experiments')






#%% 
def aggregate_metrics(parent_dir:str, metrics:dict):
    """ Aggregate the metrics of all experiments in folder `parent_dir`.
        Assumes that `parent_dir` contains multiple experiments, with their results stored in
        `parent_dir/subdir/metrics_dev.json`

        Args: 
            - parent_dir: path to directory containing experiments results
            - metrics: subDir -> {'accuracy': ..., ...}

    """

    # ===== Get the metrics for the folder if it has results from an experiment
    metrics_file = osp.join(parent_dir, 'metrics_val_best_weights.json')

    if osp.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)


    # ===== Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not osp.isdir(osp.join(parent_dir, subdir)):
            continue

        else: 
            aggregate_metrics(osp.join(parent_dir, subdir), metrics)  # recurrent loop



#%%
def metrics_to_table(metrics:dict):
    """ Get the headers from the first subdir. 
        Assumes everything has the same metrics
    """

    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res





#%%
if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from CLI 


    # ===== Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(args.parent_dir, metrics)
    table = metrics_to_table(metrics)


    # ===== Display the table to terminal
    print(table)


    # ===== Save results in parent_dir/results.md
    save_file = osp.join(args.parent_dir, "results.md")
    with open(save_file, 'w') as f:
        f.write(table)