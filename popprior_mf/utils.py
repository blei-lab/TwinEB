"""
    Utility classes and functions
   
    Author: Sohrab Salehi sohrab.salehi@columbia.edu
"""

import numpy as np
from scipy import sparse
import scipy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.mixture import GaussianMixture
import torch
import glob
import pickle
import os
import argparse
import yaml
import subprocess
import string
from time import *
import random
import datetime
from scipy.spatial import distance_matrix

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


########################################################################################################################
# Utilty functions
########################################################################################################################


def estimate_array_size(sparse_matrix, tag=""):
    # Assuming sparse_matrix is a sparse SciPy array
    # check if sparse matrix is a csr scipy matrix
    if sparse.isspmatrix_csr(sparse_matrix):
        array_size_bytes = (
            sparse_matrix.data.nbytes
            + sparse_matrix.indptr.nbytes
            + sparse_matrix.indices.nbytes
        )
        array_size_megabytes = array_size_bytes / (1024**2)

    # else check if it is sparse tensor
    elif isinstance(sparse_matrix, torch.sparse.FloatTensor):
        array_size_bytes = (
            sparse_matrix._values().nbytes() + sparse_matrix._indices().nbytes()
        )
        array_size_megabytes = array_size_bytes / (1024**2)

    print(f"Size of {tag} sparse array: {array_size_megabytes:.2f} MB")
    return array_size_megabytes


def get_missing_index_scipy(X, holdout_mask, prime_number):
    """ """
    indx_obs = np.array(X.nonzero(), dtype=np.int64)
    indx_mask = np.array(holdout_mask.nonzero(), dtype=np.int64)
    if indx_mask.shape[1] == indx_obs.shape[1]:
        # Nothing to compute, just return the empty tensor
        return np.asarray([])
    elif indx_mask.shape[1] > indx_obs.shape[1]:
        # Find the missing zero indices
        return find_diff(indx1=indx_mask, indx2=indx_obs, prime_number=prime_number)
    else:
        raise ValueError(
            "indx_mask.shape[1] < indx_obs.shape[1]. X has more observed entries than holdout_mask."
        )


def get_missing_index(X, holdout_mask, prime_number=1000003):
    """
    Finds the indices of true zeros in the input data.
    i.e., (i,j) s.t. X[i,j] = 0 and holdout_mask[i,j] = 1
    By definition, these will drop out of X
    """
    # If X is a scipy matrix, use the scipy method
    if isinstance(X, scipy.sparse.csr_matrix):
        return get_missing_index_scipy(X, holdout_mask, prime_number)

    indx_obs = X.coalesce().indices()
    indx_mask = holdout_mask.coalesce().indices()
    if indx_mask.shape[1] == indx_obs.shape[1]:
        # Nothing to compute, just return the empty tensor
        return torch.LongTensor([])
    elif indx_mask.shape[1] > indx_obs.shape[1]:
        # Find the missing zero indices
        m_indx = find_diff(indx_mask, indx_obs, prime_number)
        # check_identical_diff(indx_mask, indx_obs)
        return m_indx
    else:
        raise ValueError(
            "indx_mask.shape[1] < indx_obs.shape[1]. X has more observed entries than holdout_mask."
        )


### Sanity checks for indexing
# Assert: all indices in self.vad are also in self.holdout_mask
def sanity_check_(vad, holdout_mask):
    """Check that all elements of vad are in holdout_mask"""
    print("Sanity check... Are vad in holdout_mask?")
    vad_indx = np.array(vad.nonzero())
    holdout_indx = np.array(holdout_mask.nonzero())
    # Convert vad_indx to str
    vad_str = convert_2d_to_str(vad_indx)
    holdout_str = convert_2d_to_str(holdout_indx)
    # Check if all elements in vad_str are in holdout_str
    diff = set(vad_str) - set(holdout_str)
    assert len(diff) == 0, "Some elements in vad are not in holdout_mask"
    print("Sanity check passed! All elements in vad are in holdout_mask...")


def sanity_check_2(vad, holdout_mask):
    vad_indx = np.array(vad.nonzero())
    holdout_indx = np.array(holdout_mask.nonzero())
    diff = find_diff(vad_indx, holdout_indx)
    i = 0 if holdout_indx.shape[0] == 2 else 1
    assert diff.shape[i] == 0, "Some elements in vad are not in holdout_mask"
    print("Sanity check passed! All elements in vad are in holdout_mask...")


### Sparse matrix indexing utils
def check_identical_diff(inx1, indx2):
    """Checks that the diff computed from the two methods is an identical"""
    diff1 = find_diff(inx1, indx2)
    diff2 = find_diff_str(inx1, indx2)
    diff1_str = np.sort(convert_2d_to_str(diff1))
    diff2_str = np.sort(convert_2d_to_str(diff2))
    assert (
        np.sum(diff1_str != diff2_str) == 0
    ), "The two methods of computing diff are not identical"
    print("The two methods of computing diff are identical")


def convert_2d_to_str(indx):
    """Converts a 2d array of integers to an array of strings (row, col)"""
    assert indx.shape[0] == 2
    v1 = np.char.add(np.array(indx[0, :]).astype(str), [","])
    v2 = np.char.add(v1, np.array(indx[1, :]).astype(str))
    return v2


def convert_str_to_2d(indx_str):
    """Converts an array of strings (row, col) to a 2d array of integers"""
    # split the string by comma (,), then convert to int
    vv = np.array([x.split(",") for x in indx_str.ravel()])
    return vv.T


def find_diff_str(indx1, indx2):
    """
    Given two 2d arrays of integers, find the elements in indx1 that are not in indx2
    """
    assert indx1.shape[1] >= indx2.shape[1], "indx1 should be longer than indx2"
    indx1_str = convert_2d_to_str(indx1)
    indx2_str = convert_2d_to_str(indx2)
    diff = np.setdiff1d(indx1_str, indx2_str)
    # Convert back to 2d array
    return convert_str_to_2d(diff)


def create_hash(indx, prime_number=1000003):
    """
    Given a prime number, maps the elements of the 2d indx to a unique integer.

    Args:
        indx: 2d array of integers [2, n]

    NB: Could be improved by using a better hash function
    TODO: see here: https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
          speficially: Szudzik's function

    """
    assert indx.shape[0] == 2
    return indx[0, :] * prime_number + indx[1, :]


def unhash(indx, prime_number=1000003):
    """Inverse of create_hash"""
    # Check if indx is a tensor
    if isinstance(indx, torch.Tensor):
        return torch.stack(
            (torch.div(indx, prime_number, rounding_mode="trunc"), indx % prime_number),
            dim=1,
        )
    else:
        # (indx // prime_number, indx % prime_number)
        return np.stack((indx // prime_number, indx % prime_number), axis=0)


def find_diff(indx1, indx2, prime_number=1000003, verbose=False):
    """
    Find the elements in indx1 that are not in indx2.

    Uses a simple bijective map, (i, j) -> i * prime_number + j
    to quickly find the set difference between indx1 and indx2.

    Args:
        indx1: 2d array of integers [2, n]
        indx2: 2d array of integers [2, m]
        prime_number: a prime number larger than the maximum value in indx1

    Returns:
        diff_2d: 2d array of integers [2, n-m]

    NB: Assumes indx2 is a subset of indx1.
    """
    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    printv(f"Prime number: {prime_number}")
    # ensure indx1 is longer than inx2
    # assert indx1.shape[1] >= indx2.shape[1], "indx1 should be longer than indx2."
    # assert prime_number > indx1.max(), "prime_number should be larger than the maximum value in indx1"
    # Encode the indices (i, j) -> i * prime_number + j
    printv("Creating hash for indx1")
    indx1_hash = create_hash(indx1, prime_number)
    print("Creating hash for indx2")
    indx2_hash = create_hash(indx2, prime_number)
    # Find the set difference (np.setdiff1d(indx1_hash, indx2_hash))
    # Check if indx1_hash is a tensor
    # printv("Finding the set difference")
    if isinstance(indx1_hash, torch.Tensor):
        diff = indx1_hash[~torch.isin(indx1_hash, indx2_hash)]
    else:
        # diff = indx1_hash[~np.isin(indx1_hash, indx2_hash)]
        diff = np.setdiff1d(indx1_hash, indx2_hash)
    # Decode the mapping (diff // prime_number, diff % prime_number)
    printv("Deconding the hash...")
    diff_2d = unhash(diff, prime_number)
    return diff_2d.T


def convert_sparse_tensor_to_sparse_matrix(sparse_tensor):
    """Convert a torch-sparse to a scipy-sparse csr"""
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    shape = sparse_tensor.shape
    sparse_matrix = sparse.csr_matrix(
        (values.cpu().numpy(), indices.cpu().numpy()), shape=shape
    )
    return sparse_matrix


def sparse_tensor_from_sparse_matrix(sparse_matrix):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    assert sparse.issparse(sparse_matrix), "Input must be a scipy sparse COO matrix."
    coo = sparse_matrix.tocoo()
    indices = np.mat([coo.row, coo.col])
    return torch.sparse_coo_tensor(indices=indices, values=coo.data, size=coo.shape)


### Math Utils


def streaming_logsumexp(x, r, alpha):
    """
    Compute a running logsumexp, vectorized to support multiple datapoints.

    Args:
        x [num_datapoints]: A vector of new datapoints for computing logsumexp.
        r [num_datapoints]: A running vector of logsumexp values.
        alpha [num_datapoints]: The maximum value of inputs so far.

    Returns:
        Updated r and alpha

    Usecase:
        When the logsumexp is computed across rows of a matrix, but the columns are arriving sequentially (rows: observations, columns: different datasets)
        Computing loglikelihood of heldout data.

    Math:
        r_i = \sum_{j=1}^{i} exp(x_j - \alpha_i) = exp(-\alpha_i) + \sum_{j=1}^{i} exp(x_j)
        if x_{i+1} <= \alpha_i then just accumulate: r_{i+1} = r_i + exp(x_{i+1}-\alpha_i)
        else: cancel the old maximum and use the new one (i.e., \alpha_{i+1} = x_{i+1})
        r'_{i} = exp(\alpha_i - x_{i+1})*r_i
        r_{i+1} = r'_{i} + exp(x_{i+1} - \alpha_{i+1}) = r'_{i} + 1
        After all columns are in, then return log(r_{i+1}) + \alpha
    Adopted from: http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
    """
    r0 = r + torch.exp(x - alpha)
    r1 = r * torch.exp(alpha - x) + 1.0
    # Set inf to zero
    r0 = torch.where(torch.isinf(r0), torch.zeros_like(r0), r0)
    r1 = torch.where(torch.isinf(r1), torch.zeros_like(r1), r1)
    cond = (x <= alpha).float()
    r = r0 * cond + r1 * (1.0 - cond)
    alpha = alpha * cond + x * (1.0 - cond)
    return r, alpha


#### Command line arguments


def main_data_setup_args():
    """
    Configure argparser for setting up data to run create_heldout_data()
    """
    arg_parser = argparse.ArgumentParser(description="Setup the data")
    arg_parser.add_argument(
        "--filePath", "-i", type=str, help="Path to the anndata object"
    )
    arg_parser.add_argument(
        "--saveDir", "-o", type=str, help="Path to the output directory"
    )
    arg_parser.add_argument(
        "--holdoutPortion", "-p", type=float, help="Portion of the data to holdout"
    )
    arg_parser.add_argument(
        "--force", "-f", type=str2bool, help="Ignore the existing data", default=True
    )
    arg_parser.add_argument("--cacheDir", "-c", type=str, help="Data cache directory")
    arg_parser.add_argument(
        "--correlationCutOff",
        "-l",
        type=float,
        help="The correlation cutoff at which to drop genes.",
        default=0.9,
    )
    arg_parser.add_argument(
        "--holdoutRows", "-r", type=str2bool, help="Holdout rows", default=True
    )
    # add ignore_pca boolean
    arg_parser.add_argument(
        "--ignore_pca",
        "-g",
        type=str2bool,
        help="Does not generate standardized data (used for ppca).",
        default=False,
    )
    # get seed
    arg_parser.add_argument(
        "--seed",
        "-z",
        type=int,
        help="The seed for torch manual random number generator",
        default=0,
        required=False,
    )
    return arg_parser


def main_plot_setup_args():
    """
    Configure argparser for plotting the results.
    """
    arg_parser = argparse.ArgumentParser(description="Plot results")
    arg_parser.add_argument(
        "--merge_table_path", "-i", type=str, help="Path to the merge table"
    )
    arg_parser.add_argument(
        "--save_dir", "-o", type=str, help="Path to the output directory"
    )
    return arg_parser


def none_or_float(value):
    """Helper function for parsing floats"""
    if value == "None":
        return None
    else:
        return float(value)


def main_run_model_setup_args():
    """
    Configure argparser for running the model
    """
    arg_parser = argparse.ArgumentParser(description="Run model inference")
    arg_parser.add_argument(
        "--picklePath", "-i", nargs="+", type=str, help="Path to the pickle file"
    )
    # add a param for device
    arg_parser.add_argument(
        "--device", "-d", type=str, help="Device to use, e.g., cpu, cuda:0"
    )
    arg_parser.add_argument(
        "--save_dir", "-o", type=str, help="Path to the output directory"
    )
    arg_parser.add_argument(
        "--latent_dim", "-l", type=int, help="Latent dimension", default=3
    )
    arg_parser.add_argument(
        "--num_samples",
        "-s",
        type=int,
        help="Number of samples to estimate elbo",
        default=5,
    )
    arg_parser.add_argument(  # ast.literal_eval
        "--test", "-t", type=str2bool, help="Test mode", default=False
    )
    arg_parser.add_argument(
        "--max_steps", "-m", type=int, help="Maximum number of steps", default=1000
    )
    arg_parser.add_argument(
        "--factor_model",
        "-f",
        type=str,
        help="Factor model (e.g., PPCA, PMF, PCAEB, AmortizedPPCAEB)",
        default="AmortizedPPCAEB",
    )
    arg_parser.add_argument(
        "--batch_size", "-b", type=int, help="Batch size", default=200
    )
    arg_parser.add_argument(
        "--tolerance", "-tol", type=float, help="Tolerance for early stopping", default=10
    )
    arg_parser.add_argument(
        "--num_pseudo_obs",
        "-k",
        type=int,
        help="Number of pseudo observations for local variables in EB models",
        default=0,
        nargs="?",
    )
    arg_parser.add_argument(
        "--num_pseudo_obs_global",
        "-psi",
        type=int,
        help="Number of pseudo observations for global variables in EB models",
        default=0,
        nargs="?",
    )
    arg_parser.add_argument(
        "--annealing_factor",
        "-a",
        type=float,
        help="Annealing coefficient for the KL divergence of prior and VI posterior",
        default=1.0,
    )
    arg_parser.add_argument(
        "--n_llhood_samples",
        "-n",
        type=int,
        help="Number of samples to estimate heldout likelihood",
        default=100,
    )
    arg_parser.add_argument(
        "--stddv_datapoints",
        "-svd",
        type=float,
        help="The standard deviation for the data likelihood",
        default=1.0,
    )
    arg_parser.add_argument(
        "--log_cadence", "-c", type=float, help="How often to write the log", default=10
    )
    arg_parser.add_argument(
        "--lib_size",
        "-lib",
        type=float,
        help="The target library size to normalize all cells to in the PMF encoder family",
        default=1000,
    )
    arg_parser.add_argument(
        "--seed",
        "-z",
        type=int,
        help="The seed for torch manual random number generator",
        default=0,
    )
    arg_parser.add_argument(
        "--row_learning_rate",
        "-rlr",
        type=float,
        help="Learning rate for row variables",
        default=0.001,
    )
    arg_parser.add_argument(
        "--column_learning_rate",
        "-clr",
        type=float,
        help="Learning rate for the column variables",
        default=0.001,
    )
    arg_parser.add_argument(
        "--mixture_learning_rate",
        "-mlr",
        type=float,
        help="Learning rate for the mixture variables",
        default=0.1,
    )
    # local prior parameters
    arg_parser.add_argument(
        "--row_prior_scale",
        "-rps",
        type=float,
        help="Scale for the prior on the local variables",
        default=1.0,
    )
    arg_parser.add_argument(
        "--row_prior_concentration",
        "-rpc",
        type=float,
        help="Concentration (alpha) for the prior on the local variables in count space",
        default=0.1,
    )
    arg_parser.add_argument(
        "--row_prior_rate",
        "-rpr",
        type=float,
        help="Rate (beta) for the prior on the local variables in count space",
        default=0.3,
    )
    # global prior paramters
    arg_parser.add_argument(
        "--column_prior_scale",
        "-cps",
        type=float,
        help="Scale for the prior on the global variables",
        default=1.0,
    )
    arg_parser.add_argument(
        "--column_prior_concentration",
        "-cpc",
        type=float,
        help="Concentration (alpha) for the prior on the global variables in count space",
        default=0.1,
    )
    arg_parser.add_argument(
        "--column_prior_rate",
        "-cpr",
        type=float,
        help="Rate (beta) for the prior on the global variables in count space",
        default=0.3,
    )
    arg_parser.add_argument(
        "--optimizer",
        "-optim",
        type=str,
        help="Optimizer to use (e.g., Adam, SGD)",
        default="Adam",
    )
    arg_parser.add_argument(
        "--scheduler",
        "-sch",
        type=str,
        help="Scheduler to use (e.g., ReduceLROnPlateau, StepLR)",
        default="ReduceLROnPlateau",
    )
    arg_parser.add_argument(
        "--use_warmup",
        "-w",
        type=str2bool,
        help="Whether to use warm up for learning rate.",
        default=False,
    )
    arg_parser.add_argument(
        "--subsample_zeros",
        "-sbz",
        type=str2bool,
        help="Whether to approximate llhood by subsampling zeros.",
        default=False,
    )
    # add an str argument for prior_family, with diffult None
    arg_parser.add_argument(
        "--prior_family",
        "-pf",
        type=str,
        help="Prior family (e.g., Gaussian, Gamma)",
        default=None,
    )
    # add an argument for initialization of the row and column parameters (a string, with default xavier)
    arg_parser.add_argument(
        "--init",
        "-init",
        type=str,
        help="Initialization method (e.g., xavier, nmf)",
        default="xavier",
    )
    arg_parser.add_argument(
        "--var_fam_scale",
        "-vfs",
        type=none_or_float,
        help="The scale for the variational family",
        default=None,
    )
    arg_parser.add_argument(
        "--var_fam_init_scale",
        "-is",
        type=float,
        help="The initial scale for the variational family - still will be learned. -1 for xavier_uniform, or a positive number to set all to this value.",
        default=-1.0,
    )
    arg_parser.add_argument(
        "--sparse",
        "-sp",
        type=str2bool,
        help="Whether to inject point masses new zero in the population prior mixture.",
        default=False,
    )
    # add a hacky argument to specify how much more masking should happen
    arg_parser.add_argument(
        "--masking_factor",
        "-mf",
        type=float,
        help="How much more masking should happen on the test set.",
        default=0.0,
    )
    arg_parser.add_argument(
        "--var_family",
        "-vfd",
        type=str,
        help="Variational family distribution (e.g., lognormal, normal)",
        default="lognormal",
    )
    # add a pseudo_var_family with default lognormal
    arg_parser.add_argument(
        "--pseudo_var_family",
        "-pvfd",
        type=str,
        help="Pseudo variational family distribution (e.g., lognormal, normal)",
        default="lognormal",
    )
    arg_parser.add_argument(
        "--run_hold_out_rows",
        "-rhor",
        type=str2bool,
        help="Whether to compute test-holdout loglikelihood",
        default=True,
    )
    arg_parser.add_argument(
        "--save_checkpoint",
        "-scp",
        type=str2bool,
        help="Whether to save the best model (by ELBO) as a checkpoint",
        default=True,
    )
    arg_parser.add_argument(
        "--train_mode_switch",
        "-tms",
        type=str2bool,
        help="Whether to switch traning from both to column and rows.",
        default=False,
    )
    arg_parser.add_argument(
        "--use_custom_scheduler",
        "-ucs",
        type=str2bool,
        help="Whether to use a custom scheduler.",
        default=False,
    )
    arg_parser.add_argument(
        "--use_batch_sampler",
        "-ubs",
        type=str2bool,
        help="Whether to use the default batch sampler. Strong recommand setting True.",
        default=True,
    )
    arg_parser.add_argument(
        "--track_grad_var",
        "-tgv",
        type=str2bool,
        help="Whether to track variance of the gradients. This is very computationally expensive.",
        default=False,
    )
    arg_parser.add_argument(
        "--restore_best_model",
        "-rbm",
        type=str2bool,
        help="Whether to restore best model (checkpoint) based on ELBO at the end of training.",
        default=True,
    )
    arg_parser.add_argument(
        "--clipGradients",
        "-cg",
        type=str2bool,
        help="Whether to clip the gradients. Helps convergence when the variance variational parameters is also learnt.",
        default=True,
    )
    arg_parser.add_argument(
        "--elbo_mode",
        "-em",
        type=str,
        help="How to use particles to train the ELBO (either parallel (default) or sequential). Sequential should be used for gradient tracing.",
        default="parallel",
    )
    arg_parser.add_argument(
        "--schedule_free_epochs",
        "-sfe",
        type=int,
        help="Number of epochs before starting to use the scheduler.",
        default=0,
    )    
    arg_parser.add_argument(
        "--n_elbo_particles",
        "-nep",
        type=int,
        help="Number of samples (particles) to use when evaluating the ELBO sequentially.",
        default=3,
    )    
    arg_parser.add_argument(
        "--max_gradient_norm",
        "-mgn",
        type=float,
        help="Maximum gradient to allow in gradient clipping.",
        default=2.0,
    )
    arg_parser.add_argument(
        "--stopping_loss_threshold",
        "-slt",
        type=float,
        help="Threshold for early stopping. If cur_loss - prev_loss < threshold, then stops",
        default=2.0,
    )
    arg_parser.add_argument(
        '--init_exp_dir', 
        '-ied',
        type=str, 
        help='The path to the experiment directory from which to initialize the model params.',
        default=None,
    )
    arg_parser.add_argument(
        "--scheduler_patience",
        "-sps",
        type=int,
        help="Patience parameter for the scheduller.",
        default=3,
    )       
    arg_parser.add_argument(
        "--use_mixture_weights",
        "-umw",
        type=str2bool,
        help="Use mixture weights in the population prior",
        default=True,
    )
    arg_parser.add_argument(
        "--regularize_prior",
        "-rp",
        type=str2bool,
        help="Whether to regularize the learned prior parameters.",
        default=False,
    )
    arg_parser.add_argument(
        "--binarize_data",
        "-bd",
        type=str2bool,
        help="Whether to binarize data (sets non-zero values to 1).",
        default=True,
    )
    # add a binary for self.use_normalizing_flow_prior
    arg_parser.add_argument(
        "--use_normalizing_flow_prior",
        "-unfp",
        type=str2bool,
        help="Whether to use a normalizing flow prior. Default is False.",
        default=False,
    )
    return arg_parser


def parse_config(arg_parser):
    """Parses arguments from a config file"""

    args = []
    for i in range(len(arg_parser._action_groups)):
        for j in range(len(arg_parser._action_groups[i]._group_actions)):
            opt = arg_parser._action_groups[i]._group_actions[j].option_strings
            print(arg_parser._action_groups[i]._group_actions[j].help)
            print(opt)
            if opt[1] == "--help":
                continue
            args.append(opt[0].replace("--", ""))

    print(args)


### General utils


def runCMD(cmdStr):
    result = subprocess.run(
        cmdStr.split(" "),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        print(result.returncode, result.stdout, result.stderr)
    return result


def check_columns(df, columns):
    """
    Check if the dataframe has the given columns.
    """
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Dataframe does not have column: {column}")


def str2bool(v):
    """
    Add support for boolean argument for argparse.

    Args:
        v: the output of argparse.

    Returns:
        True or False.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def print_model_diagnostics(model):
    """
    Given a torch.nn model, print out the model paramters
    """
    print("Model's state_dict:")
    args = {}
    for key, value in model.state_dict().items():
        args[key] = list(value.size())
    indentation = len(max(args, key=len))
    for key, value in args.items():
        print(f"{key:{indentation}}", "\t", f"{value}")
    print(model)
    # for p,v in model.qz_distribution.prior.encoder.named_parameters():
    #     print(p, v.size())


def gmm_initialization(data, K, max_iter=100, n_init=1, **kwargs):
    """Initialize the pseudo observations using the GMM initialization.
    Args:
        data: A tensor of shape [N x D]

    Returns:
        A tensor of shape [K x D]
    """
    # print('Fixing random state to zero for GMM')
    # gm = GaussianMixture(n_components=K, random_state=0, max_iter=max_iter, n_init=n_init, **kwargs).fit(data)
    gm = GaussianMixture(
        n_components=K, max_iter=max_iter, n_init=n_init, **kwargs
    ).fit(data)
    return gm.means_


def transform_pca(obs, n_components=None):
    """
    Computes a PCA decomposition of the obs.

    Args:
        obs [num_data, data_dim]: The data to be decomposed.
        n_components [int]: The number of components to keep.

    Returns:
        A tuple: the loading matrix and PCs , i.e. w and z
                    w: [data_dim, n_components]
                    z: [n_components, num_data]

    NB: obs is assumed to be centered.
    """
    assert n_components is not None, "n_components must be specified."
    from sklearn.decomposition import PCA

    # print('Fixing random state to 0 for PCA!')
    # pca = PCA(n_components=n_components, random_state=0)
    pca = PCA(n_components=n_components)
    pca.fit(obs)

    z_ = pca.transform(obs).T
    w_ = pca.components_.T
    return (w_, z_)


def transform_nmf(obs, n_components=None, **kwargs):
    """
    Computes a Non-negative matrix factorization of the obs, obs = h @ w.

    Args:
        obs [num_data, data_dim]: The data to be decomposed.
        n_components [int]: The number of components to keep.

    Returns:
        A tuple: the loading matrix and PCs , i.e. w and z
                    h: [data_dim, n_components]
                    w: [n_components, num_data]

    NB: obs is assumed to be centered.
    """
    assert n_components is not None, "n_components must be specified."
    nmf = NMF(n_components=n_components, init="random", random_state=0, **kwargs).fit(
        obs
    )
    w_ = nmf.transform(obs).T
    h_ = nmf.components_.T
    return (h_, w_)


def smooth_centroids(obs, _lambda, centroids, labels, n_components, n_neighbors):
    # randomly sample a n_neighbors from each cluster
    neighbours = []
    for i in range(n_components):
        # find the indices of the points in the cluster
        cluster_indx = np.where(labels == i)[0]
        # randomly sample n_neighbors from the cluster, or all of the members of the cluster
        if len(cluster_indx) < n_neighbors:
            cluster_neighbours = cluster_indx
        else:
            cluster_neighbours = np.random.choice(
                cluster_indx, size=n_neighbors, replace=False
            )
        # append the neighbours
        neighbours.append(cluster_neighbours)
    # compute the weights
    weights = []
    for i in range(n_components):
        # find the distance of each neighbour to the centroid
        dist = np.linalg.norm(obs[neighbours[i], :] - centroids[i, :], axis=1)
        # normalize the distance
        sum_dist = np.sum(dist)
        if sum_dist == 0:
            dist = np.ones_like(dist)
        else:
            dist = dist / sum_dist
        weights.append(dist)
    # compute the weighted average (in each cluster, the )
    smoothed_centroids = []
    for i in range(n_components):
        sc = np.average(obs[neighbours[i], :], axis=0, weights=weights[i])
        sc = _lambda * centroids[i, :] + (1 - _lambda) * sc
        smoothed_centroids.append(sc)
    # convert to numpy array
    smoothed_centroids = np.array(smoothed_centroids)
    return smoothed_centroids


def from_kmeans(obs, n_components, n_init, _lambda, n_neighbors):
    kmeans = KMeans(n_clusters=n_components, random_state=0, n_init=n_init).fit(obs)
    # TODO: find the centroids, since the data is standardized, find the centroids in the original space
    centroids = kmeans.cluster_centers_
    # smooth the centroids by its neighbours
    if n_neighbors > 0:
        centroids = smooth_centroids(
            obs, _lambda, centroids, kmeans.labels_, n_components, n_neighbors
        )
    return centroids


def vanilla_dist(obs, n_components, smoothed_centroids):
    """
    Compute the row_vars
    a. compute the distance between each row and each smoothed centroid, and use that as the weight
    b. variance normalize this
    """
    row_vars = np.zeros((obs.shape[0], n_components))
    for i in range(n_components):
        # compute the distance between each row and the smoothed centroid
        dist = np.linalg.norm(obs - smoothed_centroids[i, :], axis=1)
        # normalize the distance
        dist = dist / np.sum(dist)
        # compute the weighted average
        row_vars[:, i] = dist
    # TODO: variance normalize this?
    return row_vars


def _fuzzy_clustering_test(obs, smoothed_centroids, n_components):
    """Sanity check: compare vanilla and vectorized versions of fuzzy clustering"""
    dmat = distance_matrix(
        obs,
        smoothed_centroids,
    )
    # 1. compute the slow version
    def method_1(dmat, alpha):
        hh = np.zeros((obs.shape[0], n_components))
        for i in range(obs.shape[0]):
            for k in range(n_components):
                hh[i, k] = np.sum((dmat[i, k] / dmat[i, :]) ** alpha)
                hh[i, k] = 1 / hh[i, k]
        # replace nans with zero
        hh = np.nan_to_num(hh)
        return hh

    def method_2(dmat, alpha):
        # now an efficient version of this
        # 1. compute nn \in R^{n_obs by n_components} as d ** alpha
        # 2. compute ll \in R^n_obs rowSums(1/nn)
        # 3. compute hh = [diag(ll) * (nn)]^-1
        nn = dmat**alpha
        ll = np.sum(1 / nn, axis=1)
        hh = 1 / (np.diag(ll) @ nn)
        hh = np.nan_to_num(hh)
        return hh

    f = 2
    alpha = 2 / (f - 1)
    # check if method_1 and method_2 produce very close results
    x1 = method_1(dmat, alpha=alpha)
    x2 = method_2(dmat, alpha=alpha)
    assert np.allclose(
        x1, x2, equal_nan=True
    ), "The two methods should produce the same results."


def rowvars_from_fuzzy_kmeans(obs, smoothed_centroids, f=2):
    """
    Compute a k_means fuzzy clustering of the data
    h_ik = [sum_{l=1}^K ((D(x_i, c_k)/D(x_i, c_l))^(2/(f-1)) )]^-1

    Ding, Chris, Xiaofeng He, and Horst D. Simon. "On the equivalence of nonnegative matrix factorization and spectral clustering." Proceedings of the 2005 SIAM international conference on data mining. Society for Industrial and Applied Mathematics, 2005.
    Equation (23)
    """
    # Compute the distance between each row and smoothed centroid [n_obs by n_components]
    alpha = 2 / (f - 1)
    dmat = distance_matrix(obs, smoothed_centroids)
    nn = dmat**alpha
    ll = np.sum(1 / nn, axis=1)
    hh = 1 / (np.diag(ll) @ nn)
    hh = np.nan_to_num(hh)
    return hh


def kmeans_init(obs, n_components=None, n_neighbors=20, n_init=10, _lambda=0.5):
    """
    1. for the column vars, just average each cluster and concatenate them
    2. for the row vars, use the distance function for each centroid

    # TODO: implement more flavours:
    a. use the centroids, then subsample m from the cluster, and average them
    b. randomly subsample n_component groups, and average them to use as the column

    n_neighbors = 0, does not smooth the centroids
    """
    # 1. standardize the data, using the usual sklearn method
    # 2. fit kmeans and find the centroids
    # 3. take nn samples from each cluster and average them, weighted by their distance to the centroid
    scaler = StandardScaler()
    obs_scaled = scaler.fit_transform(obs)
    smoothed_centroids = from_kmeans(
        obs_scaled,
        n_components=n_components,
        n_init=n_init,
        _lambda=_lambda,
        n_neighbors=n_neighbors,
    )
    # smoothed_centroids is (n_components by n_features)
    # stack together to construct the col_vars (n_features by n_components)
    col_vars = smoothed_centroids.T

    # comptue the row_vars (soft clustering) [n_obs by n_components]
    row_vars = rowvars_from_fuzzy_kmeans(obs_scaled, smoothed_centroids, n_components)
    # ensure that both col_vars and row_vars are non-negative

    # Since the data is standardized, enforce positivity and scale
    the_scale = np.sqrt(np.mean(obs) / n_components)
    col_vars = np.abs(col_vars) * the_scale
    row_vars = np.abs(row_vars) * the_scale

    assert np.all(col_vars >= 0), "col_vars should be non-negative"
    assert np.all(row_vars >= 0), "row_vars should be non-negative"

    return col_vars, row_vars.T


########################################################################################################################
# Utilty Classes
########################################################################################################################


class ConfigHandler:
    """Set of utilities to create and parse config files for running the models."""

    def __init__(self, configPath=None, arg_parser=None, expPath=None):
        """
        Args:
            confiPath (string): .yaml file with the configuration parameters.
        """
        if (configPath is None) == (expPath is None):
            raise ValueError("Exactly one of configPath or expPath must be specified.")

        self.configPath = (
            configPath if (expPath is None) else os.path.join(expPath, "config.yaml")
        )
        self.arg_parser = (
            arg_parser if arg_parser is not None else main_run_model_setup_args()
        )

    def parse_config(self, force=False):
        """
        Ensures that the config file matches input arguments specified in the driver.py file.

        Args:
            if force (bool): If True, will only keep legal arguments.

        Returns:
            Returns a namespace object with the parsed (legal) arguments. Use vars() to retrieve the dictionary
        """
        with open(self.configPath, "r") as file:
            config = yaml.load(file, yaml.Loader)
        args = []
        for key, value in config.items():
            args.append(f"--{key}")
            args.append(f"{value}")

        if force:
            # Just keep the legal arguments
            new_args = []
            legal_args = self._get_legal_args()
            for i in range(len(args)):
                if "--" in args[i]:
                    if args[i] in legal_args:
                        new_args.append(args[i])
                        new_args.args.append(args[i + 1])
            args = new_args

        return self.arg_parser.parse_args(args)

    def generate_empty_config(self, write_to_file=True):
        lines = []
        for i in range(len(self.arg_parser._action_groups)):
            for j in range(len(self.arg_parser._action_groups[i]._group_actions)):
                opt = self.arg_parser._action_groups[i]._group_actions[j].option_strings
                if opt[1] == "--help":
                    continue
                help = self.arg_parser._action_groups[i]._group_actions[j].help
                default = self.arg_parser._action_groups[i]._group_actions[j].default
                opt = opt[0].replace("--", "")
                lines.append(f"# {help}")
                lines.append(f"{opt}: {default}")

        if write_to_file:
            # Write lines to file one item per line
            with open(self.configPath, "w") as file:
                for line in lines:
                    file.write(line + "\n")

        return lines

    def _get_legal_args(self):
        config = {}
        for i in range(len(self.arg_parser._action_groups)):
            for j in range(len(self.arg_parser._action_groups[i]._group_actions)):
                opt = self.arg_parser._action_groups[i]._group_actions[j].option_strings
                if opt[1] == "--help":
                    continue
                opt = opt[0].replace("--", "")
                config[opt] = ""
        return config

    def write_config(self, **kwargs):
        with open(self.configPath, "w") as file:
            yaml.dump(kwargs, file)

    def read_config(self):
        with open(self.configPath, "r") as file:
            config = yaml.load(file, yaml.Loader)
        return config

    def write_updated_config(self, **kwargs):
        config = self.read_config()
        for key, value in kwargs.items():
            config[key] = value
        self.write_config(**config)


class CustomDatasetDense(torch.utils.data.Dataset):
    """A customized torch dataset."""

    def __init__(
        self,
        data,
        holdout_portion,
        obs_names=None,
        feature_names=None,
        holdout_mask=None,
        modules=None,
        heldout_data=None,
        labels=None,
        original_data_path=None,
    ):
        """
        Args:
            data (np.array): A numpy array of shape (num_datapoints, data_dim)
            data_dir (string): Directory with all the data.
            modules: A data.frame in tidy format with source,target,module columns that shows the highly correlated genes
            heldout_data: A h5ad object with the heldout rows.
            original_data_path (string): Path to the original data file (e.g. .h5ad) used to generate the data.
        """
        # TODO: use torch sparse for this...
        # See here: https://pytorch.org/docs/stable/sparse.html
        self.counts = np.array(data, dtype=np.double)
        # self.counts = torch.tensor(data, dtype=torch.double).to_sparse
        num_datapoints, data_dim = self.counts.shape
        self.num_datapoints = num_datapoints
        self.data_dim = data_dim
        self.obs_names = obs_names
        self.feature_names = feature_names
        self.modules = modules
        self.heldout_data = heldout_data
        self.labels = labels
        self.original_data_path = original_data_path
        self.__train_valid_split__(holdout_mask, holdout_portion)

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        return idx, self.train[idx], self.holdout_mask[idx]

    def __generate_holdout_mask(self, holdout_portion):
        """
        Generates a holdout mask for the data. The holdout mask is a binary matrix of shape (num_datapoints, data_dim)
        where each element is 1 with probability holdout_portion and 0 otherwise.
        NB: Will holdout zero elements as well.
        Args:
            holdout_portion (float): The portion of the data to holdout.
        Returns:
            A binary matrix of shape (num_datapoints, data_dim) where each element is 1 with probability holdout_portion and 0 otherwise.

        """
        n_holdout = int(holdout_portion * self.num_datapoints * self.data_dim)
        # Sample the row and columns to holdout
        holdout_row = np.random.randint(
            self.num_datapoints, size=n_holdout
        )  # n_holdout >> num_datapoints
        holdout_col = np.random.randint(self.data_dim, size=n_holdout)
        holdout_mask_initial = (
            sparse.coo_matrix(
                (np.ones(n_holdout), (holdout_row, holdout_col)),
                shape=self.counts.shape,
            )
        ).toarray()
        holdout_mask = np.minimum(1, holdout_mask_initial)
        return holdout_mask, holdout_row

    def __generate_holdout_mask_obs(self, holdout_portion):
        """
        Only holdout elements that are none zero.
        """
        # get the index of all non-zero elements in the matrix
        non_zero_idx = np.nonzero(self.counts)
        # pick holdout_portion of the non-zero elements
        n_holdout = int(holdout_portion * len(non_zero_idx[0]))
        holdout_idx = np.random.choice(len(non_zero_idx[0]), n_holdout, replace=False)
        holdout_row = non_zero_idx[0][holdout_idx]
        holdout_col = non_zero_idx[1][holdout_idx]
        holdout_mask_initial = (
            sparse.coo_matrix(
                (np.ones(n_holdout), (holdout_row, holdout_col)),
                shape=self.counts.shape,
            )
        ).toarray()
        holdout_mask = np.minimum(1, holdout_mask_initial)
        return holdout_mask, holdout_row

    def __train_valid_split__(self, holdout_mask, holdout_portion):
        """
        Randomly holdouts n_holdout of the matrix.
        Splits the matrix into train and validation.
        Reuses the holdout mask if one is provided.
        Sets:
            self.holdout_mask (np.array): A boolean array of shape (num_datapoints, data_dim) - True if the datapoint is held out.
            self.train (np.array): A numpy array of shape (num_datapoints, data_dim)
        """
        # TODO: Consider holding out entire rows...
        if holdout_mask is None:
            holdout_mask, holdout_row = self.__generate_holdout_mask(holdout_portion)
            # holdout_mask, holdout_row = self.__generate_holdout_mask_obs(holdout_portion)

            self.holdout_subjects = np.unique(holdout_row)
            # x1 = np.sort(self.holdout_subjects)
            # x2 = np.arange(self.num_datapoints)
            # assert (x1 == x2).all()

        # TODO: Holdout subject is vanishingly unlikely to be all the rows
        self.holdout_subjects = np.arange(self.num_datapoints)
        self.holdout_mask = holdout_mask

        self.train = np.multiply(1.0 - self.holdout_mask, self.counts)
        self.vad = np.multiply(self.holdout_mask, self.counts)


# Ideas: https://discuss.pytorch.org/t/sparse-dataset-and-dataloader/55466
# See also here: https://discuss.pytorch.org/t/dataloader-loads-data-very-slow-on-sparse-tensor/117391/6
class CustomDataset(torch.utils.data.Dataset):
    """A customized torch dataset that handles sparse data."""

    @staticmethod
    def create_holdout_mask(holdout_portion, x):
        num_datapoints, data_dim = x.shape
        n_holdout = int(holdout_portion * num_datapoints * data_dim)
        print("n_holdout: ", n_holdout)
        # Sample the row and columns to holdout
        holdout_row = np.random.randint(
            num_datapoints, size=n_holdout
        )  # n_holdout >> num_datapoints
        holdout_col = np.random.randint(data_dim, size=n_holdout)
        # drop duplicated (holdout_row, holdout_col) pairs using zip and set
        holdout_row, holdout_col = zip(*set(zip(holdout_row, holdout_col)))
        n_holdout = len(holdout_row)
        holdout_mask = sparse.csr_matrix(
            (np.ones(n_holdout), (holdout_row, holdout_col)),
            shape=(num_datapoints, data_dim),
        )
        print("Done")
        return holdout_mask, holdout_row

    @staticmethod
    def sparse_scipy_to_torch(sparse_matrix):
        print(sparse_matrix)
        # Convert a SciPy sparse matrix to a PyTorch sparse tensor
        sparse_matrix = sparse_matrix.tocoo()
        values = sparse_matrix.data
        indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
        indices = torch.tensor(indices, dtype=torch.long)
        shape = torch.Size(sparse_matrix.shape)

        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def __init__(
        self,
        data,
        holdout_portion,
        obs_names=None,
        feature_names=None,
        holdout_mask=None,
        modules=None,
        heldout_data=None,
        labels=None,
        original_data_path=None,
        x_mean=None,
        x_std=None,
    ):
        """
        Args:
            data (np.array): A numpy array of shape (num_datapoints, data_dim)
            data_dir (string): Directory with all the data.
            modules: A data.frame in tidy format with source,target,module columns that shows the highly correlated genes
            heldout_data: A h5ad object with the heldout rows.
            original_data_path (string): Path to the original data file (e.g. .h5ad) used to generate the data.
        """
        # TODO: use torch sparse for this...
        # See here: https://pytorch.org/docs/stable/sparse.html
        # self.counts = np.array(data, dtype=np.double)
        assert isinstance(data, scipy.sparse.csr.csr_matrix)
        self.counts = data
        # self.counts = torch.tensor(data, dtype=torch.double).to_sparse
        num_datapoints, data_dim = self.counts.shape
        self.num_datapoints = num_datapoints
        self.data_dim = data_dim
        self.obs_names = obs_names
        self.feature_names = feature_names
        self.modules = modules
        self.heldout_data = heldout_data
        self.labels = labels
        self.original_data_path = original_data_path
        self.__train_valid_split__(holdout_mask, holdout_portion)
        self.x_mean = x_mean
        self.x_std = x_std

    def __len__(self):
        # return len(self.counts)
        return self.counts.shape[0]

    def __getitem__(self, idx):
        # return idx, self.train[idx], self.holdout_mask[idx]
        # return idx, self.train.getrow(idx).A.squeeze(), self.holdout_mask.getrow(idx).A.squeeze()
        # return idx, self.train.getrow(idx), self.holdout_mask.getrow(idx)
        # return idx, self.train[idx], self.holdout_mask[idx]
        # return idx, self.train[idx, :], self.holdout_mask[idx, :]
        return idx, self.train[idx], self.holdout_mask[idx]

    # def __generate_holdout_mask(self, holdout_portion, only_nonzero=True):
    def __generate_holdout_mask(self, holdout_portion, only_nonzero=False):
        """
        Generates a holdout mask for the data. The holdout mask is a binary matrix of shape (num_datapoints, data_dim)
        where each element is 1 with probability holdout_portion and 0 otherwise.
        NB: Will holdout zero elements as well.

        Args:
            holdout_portion (float): The portion of the data to holdout.
            only_nonzero (bool): If True (default), will only holdout non-zero elements.

        Returns:
            A binary matrix of shape (num_datapoints, data_dim) where each element is 1 with probability holdout_portion and 0 otherwise.

        """
        print("Generating holdout mask...")
        if only_nonzero:
            # extract non-zero indeces from self.counts
            nonzero_row, nonzero_col = self.counts.nonzero()
            n_holdout = int(holdout_portion * len(nonzero_row))
            print("n_holdout: ", n_holdout)
            # Sample n_holdout from the lenght of nonzero_row
            rnd_indx = np.random.choice(len(nonzero_row), n_holdout, replace=False)
            holdout_row = nonzero_row[rnd_indx]
            holdout_col = nonzero_col[rnd_indx]
            holdout_mask = sparse.csr_matrix(
                (np.ones(n_holdout), (holdout_row, holdout_col)),
                shape=self.counts.shape,
            )

        else:
            print("Keeping zeros...")
            n_holdout = int(holdout_portion * self.num_datapoints * self.data_dim)
            print("n_holdout: ", n_holdout)
            # Sample the row and columns to holdout
            holdout_row = np.random.randint(
                self.num_datapoints, size=n_holdout
            )  # n_holdout >> num_datapoints
            holdout_col = np.random.randint(self.data_dim, size=n_holdout)
            # drop duplicated (holdout_row, holdout_col) pairs using zip and set
            holdout_row, holdout_col = zip(*set(zip(holdout_row, holdout_col)))
            n_holdout = len(holdout_row)
            holdout_mask = sparse.csr_matrix(
                (np.ones(n_holdout), (holdout_row, holdout_col)),
                shape=self.counts.shape,
            )

        # drop duplicate rows and columns, otherwise COO will sum them up
        # holdout_mask_initial = (
        #     sparse.coo_matrix(
        #         (np.ones(n_holdout), (holdout_row, holdout_col)),
        #         shape=self.counts.shape,
        #         dtype=np.int_
        #     )
        # #).toarray()
        # )
        # holdout_mask = np.minimum(1, holdout_mask_initial)
        return holdout_mask, holdout_row

    def __train_valid_split__(self, holdout_mask, holdout_portion):
        """
        Randomly holdouts n_holdout of the matrix.
        Splits the matrix into train and validation.
        Reuses the holdout mask if one is provided.

        Sets:
            self.holdout_mask (np.array): A boolean array of shape (num_datapoints, data_dim) - True if the datapoint is held out.
            self.train (np.array): A numpy array of shape (num_datapoints, data_dim)
        """
        # TODO: Consider holding out entire rows...
        if holdout_mask is None:
            print("######## Generating holdout mask... ########")
            holdout_mask, holdout_row = self.__generate_holdout_mask(holdout_portion)
            # holdout_mask, holdout_row = self.__generate_holdout_mask_obs(holdout_portion)

            self.holdout_subjects = np.unique(holdout_row)
            # x1 = np.sort(self.holdout_subjects)
            # x2 = np.arange(self.num_datapoints)
            # assert (x1 == x2).all()

        # TODO: Holdout subject is vanishingly unlikely to be all the rows
        self.holdout_subjects = np.arange(self.num_datapoints)
        self.holdout_mask = holdout_mask

        # in self.counts, set non-zero indices of self.holdout_mask to zero
        # self.train = np.multiply(1.0 - self.holdout_mask.A, self.counts)
        self.train = self.counts - self.counts.multiply(self.holdout_mask)
        # self.train = self.counts.multiply(1.0 - self.holdout_mask.A).tocsr()
        print("Not eliminating zeros...")
        # self.train.eliminate_zeros()
        # self.vad = np.multiply(self.holdout_mask, self.counts)
        # self.vad = self.counts.multiply(self.holdout_mask.A).tocsr()
        self.vad = self.counts.multiply(self.holdout_mask).tocsr()

        # sanity_check_(self.vad, self.holdout_mask)
        sanity_check_2(self.vad, self.holdout_mask)
        # self.vad.eliminate_zeros()
        # assert self.vad.nnz == self.holdout_mask.nnz
        # Don't assert the number of non zero elements in self.vad is equal to the number of non zero elements in self.holdout_mask
        # There are original zeros in the self.count
        # XXX

        print("Done...")


class CustomDatasetG(torch.utils.data.Dataset):
    """A customized torch dataset that handles sparse data. Returns the data as index, value pairs."""

    @staticmethod
    def sparse_scipy_to_torch(sparse_matrix):
        print(sparse_matrix)
        # Convert a SciPy sparse matrix to a PyTorch sparse tensor
        sparse_matrix = sparse_matrix.tocoo()
        values = sparse_matrix.data
        indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
        indices = torch.tensor(indices, dtype=torch.long)
        shape = torch.Size(sparse_matrix.shape)

        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def __init__(
        self,
        data,
        holdout_portion,
        obs_names=None,
        feature_names=None,
        holdout_mask=None,
        modules=None,
        heldout_data=None,
        labels=None,
        original_data_path=None,
        x_mean=None,
        x_std=None,
    ):
        """
        Args:
            data (np.array): A numpy array of shape (num_datapoints, data_dim)
            data_dir (string): Directory with all the data.
            modules: A data.frame in tidy format with source,target,module columns that shows the highly correlated genes
            heldout_data: A h5ad object with the heldout rows.
            original_data_path (string): Path to the original data file (e.g. .h5ad) used to generate the data.
        """
        # TODO: use torch sparse for this...
        # See here: https://pytorch.org/docs/stable/sparse.html
        # self.counts = np.array(data, dtype=np.double)
        print("In custom dataset")
        assert isinstance(data, scipy.sparse.csr.csr_matrix)
        self.counts = data
        # self.counts = torch.tensor(data, dtype=torch.double).to_sparse
        num_datapoints, data_dim = self.counts.shape
        self.num_datapoints = num_datapoints
        self.data_dim = data_dim
        self.obs_names = obs_names
        self.feature_names = feature_names
        self.modules = modules
        self.heldout_data = heldout_data
        self.labels = labels
        self.original_data_path = original_data_path
        self.__train_valid_split__(holdout_mask, holdout_portion)
        self.x_mean = x_mean
        self.x_std = x_std

        # eliminate zeros
        self.train.eliminate_zeros()
        self.train_values = self.train.data
        self.train_indices = torch.tensor(self.train.nonzero())
        self.holdout_mask_values = self.holdout_mask.data
        self.holdout_mask_indices = torch.tensor(self.holdout_mask.nonzero())

    def __len__(self):
        # return len(self.counts)
        return self.train.shape[0]

    def __getitem__(self, idx):
        # return idx, self.train[idx], self.holdout_mask[idx]
        # return idx, self.train.getrow(idx).A.squeeze(), self.holdout_mask.getrow(idx).A.squeeze()
        # return idx, self.train.getrow(idx), self.holdout_mask.getrow(idx)
        # return idx, self.train[idx], self.holdout_mask[idx]
        # return idx, self.train[idx, :], self.holdout_mask[idx, :]
        # return idx, self.train[idx], self.holdout_mask[idx]

        # train_idx_mask = self.train_indices[0] == idx
        # holdout_mask_idx_mask = self.holdout_mask_indices[0] == idx
        # train_indices = self.train_indices[:, train_idx_mask]
        # train_values = self.train_values[train_idx_mask]
        # holdout_mask_indices = self.holdout_mask_indices[:, holdout_mask_idx_mask]
        # holdout_mask_values = self.holdout_mask_values[holdout_mask_idx_mask]
        data_row = self.train.getrow(idx)
        holdout_mask_row = self.holdout_mask.getrow(idx)

        data_indices = torch.tensor(
            np.vstack((np.full_like(data_row.indices, idx), data_row.indices)),
            dtype=torch.long,
        )
        data_values = torch.tensor(data_row.data, dtype=torch.float32)
        holdout_mask_indices = torch.tensor(
            np.vstack(
                (np.full_like(holdout_mask_row.indices, idx), holdout_mask_row.indices)
            ),
            dtype=torch.long,
        )
        holdout_mask_values = torch.tensor(holdout_mask_row.data, dtype=torch.float32)

        return (
            idx,
            (data_indices, data_values),
            (holdout_mask_indices, holdout_mask_values),
        )

        return (
            idx,
            (train_indices, train_values),
            (holdout_mask_indices, holdout_mask_values),
        )

    # def __generate_holdout_mask(self, holdout_portion, only_nonzero=True):
    def __generate_holdout_mask(self, holdout_portion, only_nonzero=False):
        """
        Generates a holdout mask for the data. The holdout mask is a binary matrix of shape (num_datapoints, data_dim)
        where each element is 1 with probability holdout_portion and 0 otherwise.
        NB: Will holdout zero elements as well.

        Args:
            holdout_portion (float): The portion of the data to holdout.
            only_nonzero (bool): If True (default), will only holdout non-zero elements.

        Returns:
            A binary matrix of shape (num_datapoints, data_dim) where each element is 1 with probability holdout_portion and 0 otherwise.

        """
        print("Generating holdout mask...")
        if only_nonzero:
            # extract non-zero indeces from self.counts
            nonzero_row, nonzero_col = self.counts.nonzero()
            n_holdout = int(holdout_portion * len(nonzero_row))
            print("n_holdout: ", n_holdout)
            # Sample n_holdout from the lenght of nonzero_row
            rnd_indx = np.random.choice(len(nonzero_row), n_holdout, replace=False)
            holdout_row = nonzero_row[rnd_indx]
            holdout_col = nonzero_col[rnd_indx]
            holdout_mask = sparse.csr_matrix(
                (np.ones(n_holdout), (holdout_row, holdout_col)),
                shape=self.counts.shape,
            )

        else:
            print("Keeping zeros...")
            n_holdout = int(holdout_portion * self.num_datapoints * self.data_dim)
            print("n_holdout: ", n_holdout)
            # Sample the row and columns to holdout
            holdout_row = np.random.randint(
                self.num_datapoints, size=n_holdout
            )  # n_holdout >> num_datapoints
            holdout_col = np.random.randint(self.data_dim, size=n_holdout)
            # drop duplicated (holdout_row, holdout_col) pairs using zip and set
            holdout_row, holdout_col = zip(*set(zip(holdout_row, holdout_col)))
            n_holdout = len(holdout_row)
            holdout_mask = sparse.csr_matrix(
                (np.ones(n_holdout), (holdout_row, holdout_col)),
                shape=self.counts.shape,
            )

        # drop duplicate rows and columns, otherwise COO will sum them up
        # holdout_mask_initial = (
        #     sparse.coo_matrix(
        #         (np.ones(n_holdout), (holdout_row, holdout_col)),
        #         shape=self.counts.shape,
        #         dtype=np.int_
        #     )
        # #).toarray()
        # )
        # holdout_mask = np.minimum(1, holdout_mask_initial)
        return holdout_mask, holdout_row

    def __generate_holdout_mask_obs(self, holdout_portion):
        """
        Only holdout elements that are none zero.
        """
        # get the index of all non-zero elements in the matrix
        non_zero_idx = np.nonzero(self.counts)
        # pick holdout_portion of the non-zero elements
        n_holdout = int(holdout_portion * len(non_zero_idx[0]))
        holdout_idx = np.random.choice(len(non_zero_idx[0]), n_holdout, replace=False)
        holdout_row = non_zero_idx[0][holdout_idx]
        holdout_col = non_zero_idx[1][holdout_idx]
        holdout_mask_initial = (
            sparse.coo_matrix(
                (np.ones(n_holdout), (holdout_row, holdout_col)),
                shape=self.counts.shape,
            )
        ).toarray()
        holdout_mask = np.minimum(1, holdout_mask_initial)
        return holdout_mask, holdout_row

    def __train_valid_split__(self, holdout_mask, holdout_portion):
        """
        Randomly holdouts n_holdout of the matrix.
        Splits the matrix into train and validation.
        Reuses the holdout mask if one is provided.

        Sets:
            self.holdout_mask (np.array): A boolean array of shape (num_datapoints, data_dim) - True if the datapoint is held out.
            self.train (np.array): A numpy array of shape (num_datapoints, data_dim)
        """
        # TODO: Consider holding out entire rows...
        if holdout_mask is None:
            print("######## Generating holdout mask... ########")
            holdout_mask, holdout_row = self.__generate_holdout_mask(holdout_portion)
            # holdout_mask, holdout_row = self.__generate_holdout_mask_obs(holdout_portion)

            self.holdout_subjects = np.unique(holdout_row)
            # x1 = np.sort(self.holdout_subjects)
            # x2 = np.arange(self.num_datapoints)
            # assert (x1 == x2).all()

        # TODO: Holdout subject is vanishingly unlikely to be all the rows
        self.holdout_subjects = np.arange(self.num_datapoints)
        self.holdout_mask = holdout_mask

        # in self.counts, set non-zero indices of self.holdout_mask to zero
        # self.train = np.multiply(1.0 - self.holdout_mask.A, self.counts)
        self.train = self.counts.multiply(1.0 - self.holdout_mask.A).tocsr()
        print("Not eliminating zeros...")
        # self.train.eliminate_zeros()
        # self.vad = np.multiply(self.holdout_mask, self.counts)
        self.vad = self.counts.multiply(self.holdout_mask.A).tocsr()
        # self.vad.eliminate_zeros()
        # assert self.vad.nnz == self.holdout_mask.nnz
        # Don't assert the number of non zero elements in self.vad is equal to the number of non zero elements in self.holdout_mask
        # There are original zeros in the self.count
        print("Done...")


class BatchHandler:
    """
    Common utilities for a Batch experiment
    """

    def __init__(self, batchPath):
        """
        Args:
            batchPath (string): Directory with all the data.
        """
        self.batchPath = batchPath

    def load_data(self, standardized=True):
        """
        Load the data shared by all batches.

        Args:
            standardized (bool): If True, loads the standardized data. Otherwise, loads the raw count data.

        Returns:
            A tuple (dataset, labels)
            dataset: A numpy array of shape (num_datapoints, data_dim)
            labels: A numpy array of shape (num_datapoints,)
        """
        if standardized:
            print(glob.glob(f"{self.batchPath}/*_standard.pkl")[0])
            dataPath = glob.glob(f"{self.batchPath}/*_standard.pkl")[0]
        else:
            print(glob.glob(f"{self.batchPath}/*_counts.pkl")[0])
            dataPath = glob.glob(f"{self.batchPath}/*_counts.pkl")[0]
        dataset = pickle.load(open(dataPath, "rb"))
        # labels = [i.split("_")[1] for i in dataset.obs_names]
        labels = dataset.labels

        return dataset, labels

    @staticmethod
    def logs_to_csv(path, tag=None):
        """
        Convert SummaryWriter logs to csv for each tag scalar or the given tag

        Args:
            path: the path to the log directory (str)
            tag: the scalars tag or if None, all tags available under the scalars

        Returns:
            a pd.dataframe with columns: w_times, step, value, tag
        """
        acc = EventAccumulator(path)
        acc.Reload()
        if tag is None:
            tags = acc.Tags()["scalars"]
        else:
            assert tag in acc.Tags()["scalars"]
        res = None
        for tag in tags:
            df = pd.DataFrame(acc.Scalars(tag))
            df["tag"] = tag
            if res is None:
                res = df
            else:
                res = pd.concat([res, df])

        return res

    def summarize_batch(self, outDir=None):
        """
        Create a data.frame with the model, latent_var, logDir, and saves it under batchPath/summary.csv

        Args:
            The path to batch directory. See below for the expected structure.

        Returns:
                A tuple of data.frames, one with the scalars, one for paths.
                df1.columns: model, batch, latent, paths
                df2.columns: model, latent, w_times, step, value, tag
        Note:
            Expects the following structure:
            ├── model_name
            │  ├── latent_var_1
            │  │  └── out
            │  │      ├── config.yaml
            │  │      ├── events.out.tfevents.*
            │  │      └── model_trained.pt
            │  ├── ...
        """
        if outDir is None:
            outDir = self.batchPath
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        # glob.glob(f"{self.batchPath}/*/*/out/")

        models = os.listdir(self.batchPath)
        # Ignore . and plots
        # TODO: Add a whitelist of models to include
        models = [i for i in models if "." not in i]
        models = [i for i in models if "plots" not in i]
        print(models)
        df = pd.DataFrame()
        for model in models:
            latents = os.listdir(os.path.join(self.batchPath, model))
            # TODO: Whitelist this
            latents = [
                i for i in latents if i.replace("_", "").replace(".", "").isnumeric()
            ]
            paths = [os.path.join(self.batchPath, model, l, "out") for l in latents]
            tmp = pd.DataFrame(
                {
                    "model": model,
                    "batch": os.path.basename(self.batchPath),
                    "latent": latents,
                    "paths": paths,
                }
            )
            df = pd.concat([df, tmp])
        df.reset_index(drop=True, inplace=True)
        res = pd.DataFrame()
        for i in range(df.shape[0]):
            tmp = self.logs_to_csv(df.paths.tolist()[i])
            tmp["model"] = df.iloc[i].model
            tmp["latent"] = df.iloc[i].latent
            res = pd.concat([res, tmp])
        res.to_csv(os.path.join(self.batchPath, "summary.csv.gz"))
        return (res, df)


class ExperimentHandler:
    """
    Common utilities for a single experiment
    """

    def __init__(self, expPath):
        """
        Args:
            expPath (string): Directory with experiment results
        """
        self.expPath = expPath
        self.config = self.load_config()

    def summarize_exp(self):
        """
        Load the SummaryWriter logs and save it under expPath/summary.csv.gz

        Returns:
            A data.frame with the elbo scalars
        """
        res = BatchHandler.logs_to_csv(self.expPath)
        res.to_csv(os.path.join(self.expPath, "summary.csv.gz"))
        return res

    def load_config(self):
        # Load the config file
        with open(os.path.join(self.expPath, "config.yaml"), "r") as file:
            config = yaml.load(file, yaml.Loader)
        return config

    def load_data(self):
        # Load the data
        dataset = pickle.load(open(self.config["picklePath"], "rb"))
        # labels = [i.split("_")[1] for i in dataset.obs_names]
        labels = dataset.labels

        return dataset, labels


class FileNameUtils:
    """
    Set of functions for generating random file names
    """

    @staticmethod
    def get_time_stamp():
        """Returns a string with the current time stamp suitable for naming files"""
        return datetime.datetime.now().strftime("%Y%m-%d-%H%-M%S.%f")

    @staticmethod
    def get_random_str(N):
        """Returns a string with N alphanumeric random characters, suitable for naming files"""
        return "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(N)
        )

    @staticmethod
    def get_file_name(generic_name, suffix=""):
        """Adds a time stamp to the name"""
        return "{}_{}_{}{}".format(
            generic_name,
            FileNameUtils.get_random_str(5),
            FileNameUtils.get_time_stamp(),
            suffix,
        )


def cs(matrix):
    """
    Compute sparsity of a matrix
    """
    # if matrix is sparse, make it dense
    if isinstance(matrix, torch.sparse.FloatTensor):
        mm = matrix.to_dense()
        mm = mm.cpu().detach().numpy()
    else:
        mm = matrix.cpu().detach().numpy()
    mm = np.squeeze(mm)
    return np.sum(mm == 0) / np.prod(mm.shape)


def save_dataset(dataset, tag='', param_save_dir=None, the_vad=None):
    """
    Can load with sparse.load_npz(os.path.join(param_save_dir, "the_train_training.npz"))
    """
    assert param_save_dir is not None, "param_save_dir must be provided"
    # save each to param_save_dir using pickle as a sparse matrix
    sparse.save_npz(os.path.join(param_save_dir, f"the_train_{tag}.npz"), dataset.train)
    sparse.save_npz(os.path.join(param_save_dir, f"the_mask_{tag}.npz"), dataset.holdout_mask)
    sparse.save_npz(os.path.join(param_save_dir, f"the_counts_{tag}.npz"), dataset.counts)
    if the_vad is not None:
        sparse.save_npz(os.path.join(param_save_dir, f"the_vad_{tag}.npz"), the_vad)


class SimOutcome:
    """Holds the outcome portion of a simulation."""

    def __init__(
        self,
        orig_data_path,
        lambda_coeff,
        seed,
        obs_df,
        betas,
        simulation_args,
        misc_dict,
    ):
        self.original_data_path = orig_data_path
        self.lambda_coeff = lambda_coeff
        self.seed = seed
        self.obs_df = obs_df
        self.simulation_args = simulation_args
        self.betas = betas
        self.misc_dict = misc_dict

    def save(self, path):
        """Save the object as a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)



