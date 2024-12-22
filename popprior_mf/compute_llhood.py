#!/usr/bin/env python
"""
    Compute the heldout llhood for a given model.

    @Author: Sohrab Salehi sohrab.salehi@columbia.edu
"""

import argparse
import os
import sys
from sklearn.utils import issparse
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from model_factory import ModelHandler
import umap as umap
from utils import ExperimentHandler, str2bool, sparse_tensor_from_sparse_matrix


### ----------------------------
### Compute MAE
### ----------------------------


def _compute_mae_sparse(model, dataset, n_samples):
    """
    NB: Only compute MAE_out 
    TODO: Compute MAE_in - maybe generate the sample data one chunk at a time? Given a chunk size
    Computes MAE out when the data is sparse. 
    Avoids sampling the full data matrix (which would be memory intensive).
    - Instead, only sample heldout 
    """
    MA_in = -1 # Do not compute it for now
    MAE_out = 0
    for _ in tqdm(range(n_samples)):
        with torch.no_grad():
            X_gen = model.generate_heldout_data(dataset.holdout_mask)
            delta = X_gen.coalesce().values() - torch.from_numpy(dataset.vad.data)
            MAE_out += torch.mean(torch.abs(delta))
    
    MAE_out /= n_samples
    return MA_in, MAE_out

# TODO: Compute MAE for the holdout rows only...
    


def _compute_mae(model, dataset, n_samples):
    """Computes a monte carlo estiamte of the MAE for the holdout elements"""
    assert "Amortized" not in model.__class__.__name__, "Not implemented for amortized models - need to add OBS"
    if issparse(dataset.holdout_mask):
        return _compute_mae_sparse(model, dataset, n_samples)
    
    MAE_in, MAE_out = 0, 0

    for _ in tqdm(range(n_samples)):
        with torch.no_grad():
            #X_gen = model.generate_data(1, None, None)[0].detach().numpy()
            X_gen = model.generate_data(1, None, None)[0].cpu().numpy()
            heldout_indx = np.array(dataset.holdout_mask, dtype=bool)
            MAE_in += np.mean(np.abs(X_gen[~heldout_indx] - dataset.counts[~heldout_indx]))
            MAE_out += np.mean(np.abs(X_gen[heldout_indx] - dataset.counts[heldout_indx]))
    
    MAE_in /= n_samples
    MAE_out /= n_samples

    return MAE_in, MAE_out


def compute_MAE_test(expPath, n_llhood_samples=100, seed=1234):
    """Computes Mean Absolute Error (MAE) for the heldout elements"""
    exp_handler = ExperimentHandler(expPath)
    dataset, labels = exp_handler.load_data()
    outDataset = dataset.heldout_data
    model = ModelHandler.load_model(expPath, retrain=True)
    torch.manual_seed(seed)
    heldout_mae_in, heldout_mae_out = _compute_mae(model, outDataset, n_llhood_samples)
    print(f'MAE_in Test: for heldout rows:  {heldout_mae_in}')
    print(f'MAE_OUT Test: for heldout rows: {heldout_mae_out}')
    return heldout_mae_in, heldout_mae_out

    
def compute_mae_validation(expPath, n_llhood_samples=100, seed=1234):
    """Computes MAE for the validation set."""
    exp_handler = ExperimentHandler(expPath)
    dataset, labels = exp_handler.load_data()
    model = ModelHandler.load_model(expPath)
    torch.manual_seed(seed)
    heldout_mae_in, heldout_mae_out = _compute_mae(model, dataset, n_llhood_samples)
    print(f'MAE_in validation:  {heldout_mae_in}')
    print(f'MAE_OUT validation: {heldout_mae_out}')
    return heldout_mae_in, heldout_mae_out
    

def compute_mae(expPath, heldout_row_only=True, n_llhood_samples=100, outDir=None, seed=1234):
    """
    Computes MAE for validation and test sets.

    Saves the results as a csv file.
    """
    # Setup output
    outDir = outDir if outDir is not None else expPath
    outPath = os.path.join(outDir, 'mae.csv')

    mae_test_in, mae_test_out = compute_MAE_test(expPath=expPath, n_llhood_samples=n_llhood_samples, seed=seed)
    if heldout_row_only:
        # save the validation llhood as well as n_llhood_samples
        df = pd.DataFrame({'mae_test_in': [mae_test_in], 'mae_test_out': [mae_test_out], 'n_llhood_samples': [n_llhood_samples]})
        df.to_csv(outPath, index=False)
        return mae_test_in, mae_test_out
    
    mae_val_in, mae_val_out = compute_mae_validation(expPath=expPath, n_llhood_samples=n_llhood_samples, seed=seed)
    # save the validation llhood as well as n_llhood_samples
    df = pd.DataFrame({'mae_test_in': [mae_test_in], 'mae_test_out': [mae_test_out], 'mae_val_in': [mae_val_in], 'mae_val_out': [mae_val_out], 'n_llhood_samples': [n_llhood_samples]})
    df.to_csv(outPath, index=False)
    return mae_test_in, mae_test_out, mae_val_in, mae_val_out


### ----------------------------
### Compute llhood
### ----------------------------

def compute_llhood_test(expPath, n_llhood_samples=100, seed=1234):
    """Return llhood for the heldout __rows__"""
    exp_handler = ExperimentHandler(expPath)
    dataset, labels = exp_handler.load_data()
    outDataset = dataset.heldout_data
    #expPathRows = os.path.join(expPath, 'heldout_llhood')
    model = ModelHandler.load_model(expPath, retrain=True)
    torch.manual_seed(seed)
    heldout_llhood_rows = model.compute_heldout_loglikelihood(torch.from_numpy(outDataset.vad), torch.from_numpy(outDataset.holdout_mask), outDataset.holdout_subjects, n_monte_carlo=n_llhood_samples)
    print(f'LLhood for heldout rows: {heldout_llhood_rows}')
    return heldout_llhood_rows


def compute_llhood_validation(expPath, n_llhood_samples=100, seed=1234):
    """Computes llhood for the validation set."""
    exp_handler = ExperimentHandler(expPath)
    dataset, labels = exp_handler.load_data()
    model = ModelHandler.load_model(expPath)
    heldout_llhood = model.compute_heldout_loglikelihood(torch.from_numpy(dataset.vad), torch.from_numpy(dataset.holdout_mask), dataset.holdout_subjects, n_monte_carlo=n_llhood_samples)
    print(f'LLhood for heldout: {heldout_llhood}')
    return heldout_llhood


def compute_llhood(expPath, heldout_row_only=True, n_llhood_samples=100, outDir=None, seed=1234):
    # Setup output
    outDir = outDir if outDir is not None else expPath
    outPath = os.path.join(outDir, 'llhood.csv')

    llhood_test = compute_llhood_test(expPath=expPath, n_llhood_samples=n_llhood_samples, seed=seed)
    if heldout_row_only:
        # save the validation llhood as well as n_llhood_samples
        df = pd.DataFrame({'llhood_test': [llhood_test], 'n_llhood_samples': [n_llhood_samples]})
        df.to_csv(outPath, index=False)
        return llhood_test
    
    llhood_validation = compute_llhood_validation(expPath=expPath, n_llhood_samples=n_llhood_samples, seed=seed)
    # Save the resutls as a csv file
    df = pd.DataFrame({'llhood_test': [llhood_test], 'llhood_validation': [llhood_validation], 'n_llhood_samples': [n_llhood_samples]})
    df.to_csv(outPath, index=False)
    return llhood_test, llhood_validation

# Create an arg parser for this
if __name__ == '__main__':
    """
    Utilty to compute llhood for a given trained model. 

    Handles two tasks:
        - compute_llhood: Computes llhood for a given model.
        - compute_mae: Computes MAE for a given model.

    Args:
        See help

    Examples:
        python compute_llhood.py compute_llhood -e /path/to/experiment -n 1000 -o /path/to/output
        python compute_llhood.py compute_mae -e /path/to/experiment -n 1000 -o /path/to/output
    """

    task_name = sys.argv[1]
    
    if task_name == 'compute_llhood':
        parser = argparse.ArgumentParser(description='Compute llhood for a given model.')
        parser.add_argument('--exp', '-e', type=str, help='The path to the experiment directory.')
        parser.add_argument('--out_path', '-o', type=str, help='The path to output directory.', default=None, required=False)
        parser.add_argument('--heldout_row_only', '-r', type=str2bool, nargs='?', const=True, default=True,  help="Compute llhood for the heldout rows only (only test).")
        parser.add_argument('--n_llhood_samples', '-n', type=int, help='Number of llhood samples to use.', default=1000)
        parser.add_argument('--seed', '-s', type=int, help='Random seed.', default=1234)

        args = parser.parse_args(sys.argv[2:])
        compute_llhood(expPath=args.exp, heldout_row_only=args.heldout_row_only, n_llhood_samples=args.n_llhood_samples, outDir=args.out_path, seed=args.seed)
    elif task_name == 'compute_mae':
        parser = argparse.ArgumentParser(description='Compute MAE for a given model.')
        parser.add_argument('--exp', '-e', type=str, help='The path to the experiment directory.')
        parser.add_argument('--out_path', '-o', type=str, help='The path to output directory.', default=None, required=False)
        parser.add_argument('--heldout_row_only', '-r', type=str2bool, nargs='?', const=True, default=True,  help="Compute llhood for the heldout rows only (only test).")
        parser.add_argument('--n_llhood_samples', '-n', type=int, help='Number of llhood samples to use.', default=1000)
        parser.add_argument('--seed', '-s', type=int, help='Random seed.', default=1234)

        args = parser.parse_args(sys.argv[2:])
        compute_mae(expPath=args.exp, heldout_row_only=args.heldout_row_only, n_llhood_samples=args.n_llhood_samples, outDir=args.out_path, seed=args.seed)
    else:
        raise ValueError(f'Unknown task: {task_name}')

  





