"""
    Script to setup the data for experiments with CDM.
    Author: De-identified Author
"""

import re
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse, csr_matrix, csc_matrix
import anndata
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.discriminant_analysis import StandardScaler
from utils import CustomDataset


def split_path_strings(picklePath, tag):
    """
    Split the path string into the stem and the tag
    """
    if tag == "":
        return picklePath
    parDir, fileName = os.path.split(picklePath)
    fileName, ext = os.path.splitext(fileName)
    picklePath = os.path.join(parDir, f"{fileName}_{tag}{ext}")
    return picklePath


def pickPicklePath(picklePath, factor_model):
    """
    Expects a list of files with the following format:
    ['[../data/lx33_uu_500_0.2.pkl,', '../data/lx33_uu_500_0.2_standard.pkl,', '../data/lx33_uu_500_0.2_counts.pkl]']
    Strips the brackets and returns the path to the pickle file
    """
    print(picklePath)
    # Clean up the file_names: drop comma, and [, and ]
    for i in range(len(picklePath)):
        picklePath[i] = re.sub("[,\[\]]", "", picklePath[i])

    # OLD: # Load the standardized data if it is a pca
    # tag = "standard" if ("PCA" in factor_model) else "counts"
    # Load the log_transformed data if it is a pca, and normalize later during training using the x_mean and x_std
    tag = "" if ("PCA" in factor_model) else "counts"
    if len(picklePath) == 1:
        picklePath = split_path_strings(picklePath[0], tag)
        return picklePath
    else:
        indx = [i for i in range(len(picklePath)) if tag in picklePath[i]][0]
        return picklePath[indx]


def handle_correlated_features(adata, corlim=0.8):
    """
    Removes features with correlation above the threshold to respect overlap. See Wang et al. 2019.

    Args:
        adata (anndata.AnnData): The anndata object to remove features from.
        corlim (float): The correlation threshold, between 0 and 1.

    Returns:
        adata (anndata.AnnData): The anndata object with the features removed, and updated X and layers["counts"]
        dat: A data.frame in tidy format for the highly correlated features as modules.
    """
    #X = adata.layers["counts"].toarray() if issparse(adata.layers["counts"]) else np.array(adata.layers["counts"])
    if corlim == 1:
        return adata, None
    X = __get_array(adata.layers["counts"])
    gene_names = adata.var_names.tolist()

    X, gene_names, adata = filter_zero_genes(adata, X, gene_names)

    upper = np.triu(np.corrcoef(X, rowvar=False), k=1)
    # dat = get_connected_modules(corlim, gene_names, upper)

    print(
        f"Maximum correlation coefficient: {upper[np.triu_indices(upper.shape[0], k=1)].max():.2f}"
    )

    # Drop highly correlated genes
    to_drop = np.abs(upper).max(axis=0) > corlim
    dat = None
    if to_drop.sum() > 0:
        print(f"Dropping highly correlated genes: {sum(to_drop)}/{X.shape[1]}")

        # Keep track of which genes where removed
        drop_indx = np.where(to_drop)[0]
        max_indx = np.argmax(np.abs(upper), axis=0)[to_drop]
        dat = pd.DataFrame(
            {
                "drop_genes": np.array(gene_names)[drop_indx],
                "kept_pair": np.array(gene_names)[max_indx],
                "cor": upper.max(axis=0)[drop_indx],
            }
        )

        # Drop the highly correlated genes
        X = X[:, np.where(~to_drop)[0]]
        adata = adata[:, np.where(~to_drop)[0]]
        gene_names = [gene_names[i] for i in np.where(~to_drop)[0]]

        # TODO: this is time consuming Sanity check: Are there any two genes with more than corlim correlation?
        # upper = np.triu(np.corrcoef(X, rowvar=False), k=1)
        # vv = upper[np.triu_indices(upper.shape[0], k=1)]
        # assert vv.max() <= corlim and vv.min() >= -corlim

        # Sparsity of the matrix
        sparsity = np.count_nonzero(X) / (X.shape[0] * X.shape[1]) * 100
        print("Sparsity of the matrix {}".format(100 - sparsity))

    adata.layers["counts"] = X
    assert np.sum(adata.var_names == gene_names) == len(gene_names)
    assert adata.X.shape[1] == adata.layers["counts"].shape[1]

    return adata, dat


def filter_zero_genes(adata, X, gene_names):
    """Drop all zero genes"""
    to_keep = X.sum(axis=0) > 0
    if (~to_keep).sum() > 0:
        print(f"Dropping all zero genes: {sum(~to_keep)}/{X.shape[1]}")
        gene_names = [gene_names[i] for i in np.where(to_keep)[0]]
        X = X[:, np.where(to_keep)[0]]
        adata = adata[:, np.where(to_keep)[0]]
        assert np.sum(adata.var_names == gene_names) == len(gene_names)
        assert X.shape[1] == len(gene_names)

    return X, gene_names, adata


def __compute_mean_std_sparse(mat):
    """
    Ignores zeros, and then computes the mean and std of non zero elements.
    Sets nan to 0 for mean and to 1 for std.
    Warning: it is converting the csr matrix into a csc matrix which could be slwo.
    """
    mat_csc = csc_matrix(mat)
    # Calculate indices where the matrix is non-zero
    non_zero_data = np.split(mat_csc.data, mat_csc.indptr[1:-1])
    # Calculate mean and standard deviation for each column
    means = [np.mean(data) if data.size > 0 else 0 for data in non_zero_data]
    stds = [np.std(data) if data.size > 0 else 1. for data in non_zero_data]
    means = np.array(means)
    stds = np.array(stds)

    # Set nan or inf to zero for mean and to 1 for std
    means[np.isnan(means) | np.isinf(means)] = 0
    stds[np.isnan(stds) | np.isinf(stds)] = 1

    # Convert zero stds to 1
    stds[stds == 0] = 1
    return means, stds

def __compute_mean_std(mat):
    """
    A basic, non-memory efficient implementation of computing the mean and std of non zero elements.
    """
    # Compute the mean and std of non zero elements
    # 1. Set all zero elements to nan
    X_nan = mat.copy() if not issparse(mat) else mat.A.copy()
    # if X_nan is sparse, make it dense
    X_nan[X_nan == 0] = np.nan
    # 2. Compute the mean and std of each column, igonring nans
    X_mean = np.nanmean(X_nan, axis=0)
    # Set nan's to zero
    X_mean[np.isnan(X_mean)] = 0
    X_std = np.nanstd(X_nan, axis=0)
    # Set nan's to one
    X_std[np.isnan(X_std)] = 1
    return X_mean, X_std


def __handle_sparse_normalization(X, M, S):
    """Standardize only the non zero elements of each column of X"""
    if isinstance(X, csr_matrix):
        coo = X.tocoo()
        indices = np.array([coo.row, coo.col])
        values = (coo.data - M[indices[1]]) / S[indices[1]]
        # Set nan's to zero
        values[np.isnan(values)] = 0
        X = csr_matrix((values.squeeze(), indices), shape=X.shape)
    else:
        raise ValueError("X must be a csr_matrix")
    return X


def __handle_normalization(X, M, S):
    """
    Given a matrix X, mean M and std S, normalize X by (X - M) / S, but only consider the non zero elements of X.
    """
    # Set zero elements of X to nan, then subtract the mean and divide by std
    X_nan = X.copy() if not issparse(X) else X.A.copy()
    X_nan[X_nan == 0] = np.nan
    X_norm = (X_nan - M) / S
    # Set nan's to zero
    X_norm[np.isnan(X_norm)] = 0
    return X_norm




def get_connected_modules(corlim, gene_names, upper, plot=False):
    """Create the connected modules based on correlation"""
    rows, cols = np.where(upper > corlim)
    rows0, cols0 = np.where(upper < -corlim)
    rows = np.concatenate((rows, rows0))
    cols = np.concatenate((cols, cols0))
    row_names = [gene_names[i] for i in rows]
    col_names = [gene_names[i] for i in cols]
    edges = zip(row_names, col_names)
    gr = nx.Graph()
    gr.add_edges_from(edges)
    if plot:
        nx.draw(gr, node_size=1)
        plt.show()
    # Return genes that are correlated, split into modules
    S = [gr.subgraph(c).copy() for c in nx.connected_components(gr)]
    dat = None
    for i in range(len(S)):
        tmp = pd.DataFrame(S[i].edges, columns=["source", "target"])
        tmp["module"] = i
        if dat is None:
            dat = tmp
        else:
            dat = pd.concat([dat, tmp])
    return dat


def __save_file(
    X,
    holdout_portion,
    outPath,
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
    dataset = CustomDataset(
        X,
        holdout_portion=holdout_portion,
        obs_names=obs_names,
        feature_names=feature_names,
        holdout_mask=holdout_mask,
        modules=modules,
        heldout_data=heldout_data,
        labels=labels,
        original_data_path=original_data_path,
        x_mean=x_mean,
        x_std=x_std,
    )
    print('Saving dataset to: ', outPath)
    with open(outPath, "wb") as f:
        pickle.dump(dataset, f)
    return dataset.holdout_mask


def __safe_std(X, ignore_nan=False, axis=0):
    """
    Compute the standard deviation of the data, ignoring NaNs if ignore_nan is True, will set std to 1 if std is 0.
    """
    # check if X is a sparse matrix
    if issparse(X):
        sscalter = StandardScaler(with_mean=False, with_std=True).fit(X)
        std = sscalter.scale_
    else:  
        if ignore_nan:
            std = np.nanstd(X, axis=axis)
        else:
            std = X.std(axis=axis)

    std[std == 0] = 1
    return std

def __get_array(X):
    """Ensure that X is a numpy array (and not a sparse matrix))"""
    #return np.array(X.A if issparse(X) else X, dtype=np.double)
    assert isinstance(X, scipy.sparse.csr.csr_matrix)
    return X


def __create_holdout_rows(holdout_portion_used, dat):
    """Holdout rows (obs) randomly."""
    heldout_rows = np.random.choice(
        dat.shape[0], int(dat.shape[0] * holdout_portion_used), replace=False
    )
    heldin_rows = np.array(list(set(range(dat.shape[0])) - set(heldout_rows)))
    dat_out = dat[heldout_rows, :].copy()
    dat_in = dat[heldin_rows, :].copy()
    return dat_in, dat_out

def __create_holdout_rows_cols(holdout_portion_used, dat):
    """Holdout rows (obs) and cols (features) randomly."""
    heldout_rows = np.random.choice(
        dat.shape[0], int(dat.shape[0] * holdout_portion_used), replace=False
    )
    heldout_cols = np.random.choice(
        dat.shape[1], int(dat.shape[1] * holdout_portion_used), replace=False
    )
    heldin_rows = np.array(list(set(range(dat.shape[0])) - set(heldout_rows)))
    heldin_cols = np.array(list(set(range(dat.shape[1])) - set(heldout_cols)))
    # Create the 4 parts of the matrix
    dat_in = dat[heldin_rows, :][:, heldin_cols].copy()
    dat_rowout = dat[heldout_rows, :][:, heldin_cols].copy()
    dat_colout = dat[heldin_rows, :][:, heldout_cols].copy()
    dat_out = dat[heldout_rows, :][:, heldout_cols].copy()
    return dat_in, dat_rowout, dat_colout, dat_out


# Create holdout data
def create_heldout_data(
    filePath,
    save_dir,
    holdout_portion=0.2,
    force=False,
    data_cache_path=None,
    correlation_limit=0.9,
    holdout_rows=True,
    seed=0, 
    ignore_pca=False,
):
    """
    Creates a heldout data set from the original data set.
    Expects the log transformed data in X and the counts data in .layers['counts']

    Parameters
    ----------
    filePath : str
        Path to the h5ad file where the data is stored in slot X.
    save_dir : str
        Path to the directory where the heldout data set will be saved.
    holdout_portion : float
        Proportion of the data set that will be heldout.
    correlation_limn : float
        The correlation threshold, between 0 and 1, at which to drop highly correlated features.
    holdout_rows: bool
        If True, holdout rows as well as single elements of the matrix.
    seed: int
        Random seed for reproducibility.
    ignore_pca: bool
        If True, does not generate the non-count data.

    Returns
    -------
    outPath : [str, str, str]
        Path to the heldout data set in pickle format (for log, standardized_log, counts)

    """
    print(f'Setting seed to {seed}')
    np.random.seed(seed)
    stem = Path(filePath).stem
    # TODO: For backwards compatability...
    #nameStr = f"{stem}_{holdout_portion}_{correlation_limit}_{seed}"
    nameStr = f"{stem}_{holdout_portion}_{correlation_limit}"
    outPath = os.path.join(save_dir, f"{nameStr}.pkl")
    outPathCounts = os.path.join(save_dir, f"{nameStr}_counts.pkl")

    if os.path.exists(outPath) and not force:
        print("Holdout data already exists. Skip.")
        return outPath

    # Also skip if the data exists in the data_cache_path directory. Create a symlynk
    print(f"data_cache_path: {data_cache_path}")
    if not force and data_cache_path is not None:
        tmpPath = Path(os.path.join(data_cache_path, os.path.basename(outPath)))
        tmpPathCounts = Path(os.path.join(data_cache_path, os.path.basename(outPathCounts)))
        # print(f"tmpPath: {tmpPath}")
        # print(f"tmpPathCounts: {tmpPathCounts}")
        if tmpPath.exists() and tmpPathCounts.exists():
            print("Holdout data already exists. Creating symlinks.")
            os.symlink(tmpPath, outPath)
            os.symlink(tmpPathCounts, outPathCounts)
            return [outPath, outPathCounts]
        else:
            # raise ValueError(f'The data_cache_path directory does not contain the data. Expected {tmpPath} and {tmpPathCounts}')
            print("Holdout data does not exist. Creating.")
            print(f"tmpPath: {tmpPath}")
            print(f"tmpPathCounts: {tmpPathCounts}")
            
    dat = anndata.read_h5ad(filePath)

    # Check that dat.layers['counts'] exists and is sparse, if not sparse convert to csr matrix
    if 'counts' not in dat.layers.keys():
        raise ValueError('Counts layer not found in data.')
    if not issparse(dat.layers['counts']):
        dat.layers['counts'] = csr_matrix(dat.layers['counts'])

    # Drop highly correlated features
    dat, modules = handle_correlated_features(adata=dat, corlim=correlation_limit)

    holdout_portion_used = holdout_portion if holdout_rows else holdout_portion / 2

    if holdout_rows:
        dat_in, dat_out = __create_holdout_rows(dat=dat, holdout_portion_used=holdout_portion_used)
    else:
        dat_out = None
        dat_in = dat.copy()

    ##---------- Save an standardized version of the data
    # TODO: Computing X mean for the holdout mask separately to avoid data leakage - ignoring now since we test on the heldout rows
    X = __get_array(dat_in.X)
    # Find only the mean and variance of NON-zero elements

    # 1. Create a holdout mask
    # 2. Apply the holdout mask to the training data
    # 3. Compute the mean and std of the non zero elements

    holdout_mask, holdout_row = CustomDataset.create_holdout_mask(holdout_portion=holdout_portion_used, x=X)
    # Set values of X to zero where holdout_mask is True
    print('Computing X_h')
    X_h = (X - X.multiply(holdout_mask))
    # compute sparsity of X
    def cs(X):
        print(1 - X.nnz / (X.shape[0] * X.shape[1]))

    # Now compute the more efficient way
    print('Computing X_mean and X_std')
    X_mean, X_std = __compute_mean_std_sparse(X_h)
    print('Done')


    feature_names = dat_in.var_names.tolist()
    obs_names = dat_in.obs_names.tolist()
    labels = dat_in.obs["labels"].tolist()
    labels_out = None if dat_out is None else dat_out.obs["labels"].tolist()
    print(f'ignore_pca: {ignore_pca}')
    if not ignore_pca:
        # Centering is constly on sparse matrices!
        # If dat_out is not None, create and save a hodout mask for it too and save it as a CustomDataset
        if dat_out is not None:
            X_out = __get_array(dat_out.X)
            X_out = __handle_sparse_normalization(X_out, X_mean, X_std)
            out_dataset = CustomDataset(X_out, holdout_portion=holdout_portion, obs_names=dat_out.obs_names, feature_names=dat_out.var_names, labels=labels_out, holdout_mask=None, x_mean=X_mean, x_std=X_std)
        else:
            out_dataset = None

        ## Standardize using x_mean and x_std
        X = __handle_sparse_normalization(X, X_mean, X_std)

        ##---------- Save the log transformed data
        print('Saving file...')
        #holdout_mask = __save_file(
        holdout_mask0 = __save_file(
            X=X,
            holdout_portion=holdout_portion,
            outPath=outPath,
            holdout_mask=holdout_mask,
            obs_names=obs_names,
            feature_names=feature_names,
            modules=modules,
            heldout_data=out_dataset,
            labels=labels,
            original_data_path=filePath,
            x_mean=X_mean,
            x_std=X_std
        )
    else:
        print('Not saving standardized data... Will create a dummpy file. ')
        out_dataset = None
        #holdout_mask = None
        # create an empty file to indicate that the data was not saved - a hacky way to not have to change the code downstream
        with open(outPath, 'wb') as f:
            pickle.dump([], f)
        
    ##---------- Save the counts
    print('Saving counts...')
    X = __get_array(dat_in.layers["counts"])
    
    if dat_out is not None:
        out_holdout_mask = None if out_dataset is None else out_dataset.holdout_mask
        dat_out.X = __get_array(dat_out.layers["counts"])
        out_dataset_counts = CustomDataset(__get_array(dat_out.X), holdout_portion=holdout_portion, obs_names=dat_out.obs_names, feature_names=dat_out.var_names, holdout_mask=out_holdout_mask, labels=labels_out, x_mean=X_mean, x_std=X_std)
    else:
        out_dataset_counts = None

    __save_file(
        X=X,
        holdout_portion=holdout_portion_used,
        outPath=outPathCounts,
        obs_names=obs_names,
        feature_names=feature_names,
        holdout_mask=holdout_mask,
        modules=modules,
        heldout_data=out_dataset_counts,
        labels = labels,
        original_data_path=filePath,
        x_mean=X_mean,
        x_std=X_std
    )

    return [outPath, outPathCounts]





# Create holdout data
def create_heldout_data_with_cols(
    filePath,
    save_dir,
    holdout_portion=0.2,
    force=False,
    data_cache_path=None,
    correlation_limit=0.9,
    holdout_rows=True,
    seed=0, 
    ignore_pca=False,
    holdout_cols=True,
):
    """
    Creates a heldout data set from the original data set.
    Expects the log transformed data in X and the counts data in .layers['counts']

    will divide the matrix into 4 parts: X0, X1, X2, X3
    X0: training
    X1: heldout rows
    X2: heldout columns
    X3: heldout rows and columns

    Parameters
    ----------
    filePath : str
        Path to the h5ad file where the data is stored in slot X.
    save_dir : str
        Path to the directory where the heldout data set will be saved.
    holdout_portion : float
        Proportion of the data set that will be heldout.
    correlation_limn : float
        The correlation threshold, between 0 and 1, at which to drop highly correlated features.
    holdout_rows: bool
        If True, holdout rows as well as single elements of the matrix.
    seed: int
        Random seed for reproducibility.
    ignore_pca: bool
        If True, does not generate the non-count data.

    Returns
    -------
    outPath : [str, str, str]
        Path to the heldout data set in pickle format (for log, standardized_log, counts)

    """
    print(f'Setting seed to {seed}')
    np.random.seed(seed)
    stem = Path(filePath).stem
    nameStr = f"{stem}_{holdout_portion}_{correlation_limit}"
    outPath = os.path.join(save_dir, f"{nameStr}.pkl")
    outPathCounts = os.path.join(save_dir, f"{nameStr}_counts.pkl")

    if os.path.exists(outPath) and not force:
        print("Holdout data already exists. Skip.")
        return outPath

    # Also skip if the data exists in the data_cache_path directory. Create a symlynk
    print(f"data_cache_path: {data_cache_path}")
    if not force and data_cache_path is not None:
        tmpPath = Path(os.path.join(data_cache_path, os.path.basename(outPath)))
        tmpPathCounts = Path(os.path.join(data_cache_path, os.path.basename(outPathCounts)))
        # print(f"tmpPath: {tmpPath}")
        # print(f"tmpPathCounts: {tmpPathCounts}")
        if tmpPath.exists() and tmpPathCounts.exists():
            print("Holdout data already exists. Creating symlinks.")
            os.symlink(tmpPath, outPath)
            os.symlink(tmpPathCounts, outPathCounts)
            return [outPath, outPathCounts]
        else:
            # raise ValueError(f'The data_cache_path directory does not contain the data. Expected {tmpPath} and {tmpPathCounts}')
            print("Holdout data does not exist. Creating.")
            print(f"tmpPath: {tmpPath}")
            print(f"tmpPathCounts: {tmpPathCounts}")
            
    dat = anndata.read_h5ad(filePath)

    # Check that dat.layers['counts'] exists and is sparse, if not sparse convert to csr matrix
    if 'counts' not in dat.layers.keys():
        raise ValueError('Counts layer not found in data.')
    if not issparse(dat.layers['counts']):
        dat.layers['counts'] = csr_matrix(dat.layers['counts'])

    # Drop highly correlated features
    dat, modules = handle_correlated_features(adata=dat, corlim=correlation_limit)

    holdout_portion_used = holdout_portion if holdout_rows else holdout_portion / 2


    if holdout_rows and holdout_cols:
        dat_in, dat_rowout, dat_colout, dat_out  = __create_holdout_rows_cols(dat=dat, holdout_portion_used=holdout_portion_used)
    elif holdout_rows:
        dat_in, dat_out = __create_holdout_rows(dat=dat, holdout_portion_used=holdout_portion_used)
    else:
        dat_out = None
        dat_in = dat.copy()

    ##---------- Save an standardized version of the data
    # TODO: Computing X mean for the holdout mask separately to avoid data leakage - ignoring now since we test on the heldout rows
    X = __get_array(dat_in.X)
    # Find only the mean and variance of NON-zero elements

    # 1. Create a holdout mask
    # 2. Apply the holdout mask to the training data
    # 3. Compute the mean and std of the non zero elements

    holdout_mask, holdout_row = CustomDataset.create_holdout_mask(holdout_portion=holdout_portion_used, x=X)
    # Set values of X to zero where holdout_mask is True
    print('Computing X_h')
    X_h = (X - X.multiply(holdout_mask))
    # compute sparsity of X
    def cs(X):
        print(1 - X.nnz / (X.shape[0] * X.shape[1]))

    # Now compute the more efficient way
    print('Computing X_mean and X_std')
    X_mean, X_std = __compute_mean_std_sparse(X_h)
    print('Done')


    feature_names = dat_in.var_names.tolist()
    obs_names = dat_in.obs_names.tolist()
    labels = dat_in.obs["labels"].tolist()
    labels_out = None if dat_out is None else dat_out.obs["labels"].tolist()
    print(f'ignore_pca: {ignore_pca}')
    if not ignore_pca:
        # Centering is constly on sparse matrices!
        # If dat_out is not None, create and save a hodout mask for it too and save it as a CustomDataset
        if dat_out is not None:
            X_out = __get_array(dat_out.X)
            X_out = __handle_sparse_normalization(X_out, X_mean, X_std)
            out_dataset = CustomDataset(X_out, holdout_portion=holdout_portion, obs_names=dat_out.obs_names, feature_names=dat_out.var_names, labels=labels_out, holdout_mask=None, x_mean=X_mean, x_std=X_std)
        else:
            out_dataset = None

        ## Standardize using x_mean and x_std
        X = __handle_sparse_normalization(X, X_mean, X_std)

        ##---------- Save the log transformed data
        print('Saving file...')
        #holdout_mask = __save_file(
        holdout_mask0 = __save_file(
            X=X,
            holdout_portion=holdout_portion,
            outPath=outPath,
            holdout_mask=holdout_mask,
            obs_names=obs_names,
            feature_names=feature_names,
            modules=modules,
            heldout_data=out_dataset,
            labels=labels,
            original_data_path=filePath,
            x_mean=X_mean,
            x_std=X_std
        )
    else:
        print('Not saving standardized data... Will create a dummpy file. ')
        out_dataset = None
        #holdout_mask = None
        # create an empty file to indicate that the data was not saved - a hacky way to not have to change the code downstream
        with open(outPath, 'wb') as f:
            pickle.dump([], f)
        
    ##---------- Save the counts
    print('Saving counts...')
    X = __get_array(dat_in.layers["counts"])
    
    if dat_out is not None:
        out_holdout_mask = None if out_dataset is None else out_dataset.holdout_mask
        dat_out.X = __get_array(dat_out.layers["counts"])
        out_dataset_counts = CustomDataset(__get_array(dat_out.X), holdout_portion=holdout_portion, obs_names=dat_out.obs_names, feature_names=dat_out.var_names, holdout_mask=out_holdout_mask, labels=labels_out, x_mean=X_mean, x_std=X_std)
    else:
        out_dataset_counts = None

    __save_file(
        X=X,
        holdout_portion=holdout_portion_used,
        outPath=outPathCounts,
        obs_names=obs_names,
        feature_names=feature_names,
        holdout_mask=holdout_mask,
        modules=modules,
        heldout_data=out_dataset_counts,
        labels = labels,
        original_data_path=filePath,
        x_mean=X_mean,
        x_std=X_std
    )

    return [outPath, outPathCounts]
