#!/usr/bin/env python3
# Code imported from here: https://github.com/zehsilva/prior-predictive-specification/blob/200291ca55469fd7d07df9ba68b82cdd51c31afc/bo_optimization/pmf_sampling.py#L92
import numpy as np
import pandas as pd
import scipy as sp
import os
import pickle
import scipy.sparse as sp

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


def full_biased_corr(data_mat):
    temp = np.tril(np.corrcoef(data_mat), -1).flatten()
    temp = temp[np.nonzero(temp)]
    corr1 = np.nanmean(temp)
    temp = np.tril(np.corrcoef(data_mat, rowvar=False), -1).flatten()
    temp = temp[np.nonzero(temp)]
    corr2 = np.nanmean(temp)
    return corr1, corr2


def sampling_corr(x, ns=10000):
    N, D = x.shape
    vals = np.zeros(shape=(ns, 2))
    # vals2 = np.zeros(shape=(ns,2))
    for i in range(ns):
        sam1 = np.random.choice(N, 1)
        sam2 = np.random.choice(D, 2)
        vals[i, 0] = x[sam1, sam2[0]]
        vals[i, 1] = x[sam1, sam2[1]]
    corr2 = np.corrcoef(vals, rowvar=False)[0, 1]
    for i in range(ns):
        sam1 = np.random.choice(N, 2)
        sam2 = np.random.choice(D, 1)
        vals[i, 0] = x[sam1[0], sam2]
        vals[i, 1] = x[sam1[1], sam2]
    corr1 = np.corrcoef(vals, rowvar=False)[0, 1]
    return (corr1, corr2)


def sampling_corr_cov_empirical_adjust(x, ns=10000):
    N, D = x.shape
    vals = np.zeros(shape=(ns, 2))
    # vals2 = np.zeros(shape=(ns,2))
    mean = np.mean(x)
    v_bias = np.var(x)
    for i in range(ns):
        sam1 = np.random.choice(N, 1)
        sam2 = np.random.choice(D, 2)
        vals[i, 0] = x[sam1, sam2[0]]
        vals[i, 1] = x[sam1, sam2[1]]
    corr2 = np.corrcoef(vals, rowvar=False)[0, 1]
    cov2 = np.cov(vals, rowvar=False)[0, 1]
    for i in range(ns):
        sam1 = np.random.choice(N, 2)
        sam2 = np.random.choice(D, 1)
        vals[i, 0] = x[sam1[0], sam2]
        vals[i, 1] = x[sam1[1], sam2]
    corr1 = np.corrcoef(vals, rowvar=False)[0, 1]
    cov1 = np.cov(vals, rowvar=False)[0, 1]
    v_unbias = 2 * v_bias + (D - 1) / (N * D - 1) * cov1 + (N - 1) / (N * D - 1) * cov2


def recover_k_empirical_simple(e_corr1_m, e_corr2_m, e_mean, e_var):
    res = ((e_mean / e_var) ** 2) * (
            (1 - (e_corr1_m + e_corr2_m)) * e_var / (e_corr1_m * e_corr2_m) - e_mean / (e_corr1_m * e_corr2_m))
    return (res)


def recover_all_gamma(e_corr1_m, e_corr2_m, e_mean, e_var, mean_a=1):
    K = np.ceil(recover_k_empirical_simple(e_corr1_m, e_corr2_m, e_mean, e_var))
    res1 = (1 - (e_corr1_m + e_corr2_m)) / e_corr1_m - e_mean / (e_corr1_m * e_var)
    res2 = (1 - (e_corr1_m + e_corr2_m)) / e_corr2_m - e_mean / (e_corr2_m * e_var)
    a1 = 1. / res1
    b1 = 1. / res2
    a2 = a1 / mean_a
    b2 = K * (a1 / a2) * (b1 / e_mean)  # e_mean = K*(a1/a2)*(b1/b2)
    return (int(K), a1, a2, b1, b2)


def empirical(data_mat, ns=50000):
    corr1, corr2 = sampling_corr(data_mat, ns)
    e_mean = np.mean(data_mat)
    e_var = np.var(data_mat)
    e_k = recover_k_empirical_simple(corr1, corr2, e_mean, e_var)
    return {"e_mean": e_mean, "e_var": e_var, "e_corr_vals": np.array([corr1, corr2]), "e_latent_dim": e_k}


def empirical_v2(data_mat):
    corr1, corr2 = full_biased_corr(data_mat)
    e_mean = np.mean(data_mat)
    e_var = np.var(data_mat)
    e_k = recover_k_empirical_simple(corr1, corr2, e_mean, e_var)
    return {"e_mean": e_mean, "e_var": e_var, "e_corr_vals": np.array([corr1, corr2]), "e_latent_dim": e_k}


def theoretical(shape_a, scale_a, shape_b, scale_b, latent):
    mean_a = shape_a * scale_a
    mean_b = shape_b * scale_b
    std_a = np.sqrt(shape_a) * (scale_a)
    std_b = np.sqrt(shape_b) * (scale_b)
    v0 = (mean_a * mean_b)
    v1 = (mean_a * std_b) ** 2
    v2 = (mean_b * std_a) ** 2
    v3 = (std_a * std_b) ** 2
    varr = v0 + v1 + v2 + v3
    return {"mean": latent * mean_a * mean_b, "var": (latent) * varr, "corr_vals": np.array([v1 / varr, v2 / varr]),
            "cov_vals": (latent) * np.array([varr - v1, varr - v2, varr])}


def findhyper(target_mean, target_var, target_corr1, target_corr2):
    e_k = recover_k_empirical_simple(target_corr1, target_corr2, target_mean, target_var)
    def optim_fun(x):
        theo = theoretical(x[0], x[1], x[2], x[3], e_k)
        return (target_mean - theo['mean']) ** 4 + (target_var - theo['var']) ** 2 - 100000 * theo['mean'] - theo['var']
        +np.abs(target_corr1 - theo['corr_vals'][0]) / target_corr1 + np.abs(
            target_corr2 - theo['corr_vals'][1]) / target_corr2
    x = sp.optimize.minimize(optim_fun, np.array([10, 10, 10, 10]), method='Nelder-Mead', tol=1e-6,
                             options={"maxtiter": 20000})
    print("theoretical achieved: ", theoretical(x.x[0], x.x[1], x.x[2], x.x[3], e_k))
    print("target values: ", target_mean, target_var, target_corr1, target_corr2)
    return x


def bb(uu):
    uu[uu <= 3] = 0
    uu[uu > 3] = 1
    return uu




def ppp(uu):
    """where uu is a sparse matrix"""
    # cast uu between zero and one, then multiply by 10K
    # compute the rows sums, then divide by rowsums
    row_sums = uu.sum(axis=1)
    # divide each row by its sum
    uu = uu / row_sums[:, np.newaxis]
    # multiply by 10K
    uu = uu * 10000
    # then compute log1p
    # convert back to nearest integer
    uu = uu.astype(int)
    return uu

def pp_1(yy):
    yy.counts = ppp(yy.counts)
    yy.vad = ppp(yy.vad)
    yy.train = ppp(yy.train)
    return yy


def count_normalize_(xx):
    """Compute log1p and normalize total"""
    xx = pp_1(xx)
    xx.heldout_data = pp_1(xx.heldout_data)
    return xx



def compute_statistics_v2(data_path, dataset_name, out_dir, binarize=False, count_normalize=False, dat=None):
    """
    Use their implementation
    """
    
    if dat is None:
        # read the pickle file
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        dat = dataset.train.A
    if binarize:
        print(f'Binarizing the data')
        dat = bb(dat)
    elif count_normalize:
        print(f'Count normalizing the data')
        dat = ppp(dat)
    EMPIRICAL_ESTIMATES = empirical_v2(dat)
    K, THETA_SHAPE, THETA_RATE, BETA_SHAPE, BETA_RATE = recover_all_gamma(EMPIRICAL_ESTIMATES['e_corr_vals'][0], EMPIRICAL_ESTIMATES['e_corr_vals'][1], EMPIRICAL_ESTIMATES['e_mean'], EMPIRICAL_ESTIMATES['e_var'], 1.0)
    bd = EMPIRICAL_ESTIMATES['e_mean']/EMPIRICAL_ESTIMATES['e_var'] * np.sqrt( (EMPIRICAL_ESTIMATES['e_corr_vals'][0]*EMPIRICAL_ESTIMATES['e_corr_vals'][1]) / (THETA_SHAPE*BETA_SHAPE) )
    print("[pmf_objectives] Default data: stats=%s, nrows=%s ncols=%s => K=%s, a=%s, b=%s, c=%s, d=%s, b*d=%s" % (EMPIRICAL_ESTIMATES, dat.shape[0], dat.shape[1], K,THETA_SHAPE,THETA_RATE,BETA_SHAPE,BETA_RATE,bd))
    a = THETA_SHAPE
    b = THETA_RATE
    c = BETA_SHAPE
    d = BETA_RATE
    # print all at .2f 
    print('K: {:.2f}'.format(K))
    print('({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(a, b, c, d))
    # write this as a yaml file
    res_dict = {'K': K, 'a': a, 'b': b, 'c': c, 'd': d, 'bd': bd, 'dataset': dataset_name, 'data_path': data_path, 'binarize': binarize}
    # save as pickle
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        pickle_path = os.path.join(out_dir, f'{dataset_name}_prior_matching_v2.pkl')
        with open(pickle_path, 'wb') as file:
            pickle.dump(res_dict, file)
    return res_dict


# add a main function for argparser to run compute_statistics_v2
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute the statistics for the prior matching.')
    parser.add_argument('--data_path', '-d', type=str, help='The path to the data pickle file.')
    parser.add_argument('--dataset_name', '-n', type=str, help='The name of the dataset.', default=None, required=False)
    parser.add_argument('--out_dir', '-o', type=str, help='The path to the output directory.', default=None, required=False)
    parser.add_argument('--binarize', '-b', type=str2bool, help='Whether to binarize the data.', default=False, required=False)
    parser.add_argument('--count_normalize', '-c', type=str2bool, help='Whether to binarize the data.', default=False, required=False)
    args = parser.parse_args()
    compute_statistics_v2(args.data_path, args.dataset_name, args.out_dir, args.binarize, args.count_normalize)
