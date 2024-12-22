# """
#     Utilities for generating synthetic data with structured latent factors
# """
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import scanpy as sc
from scipy.sparse import csr_matrix
import os
import pickle
from utils import ConfigHandler, ExperimentHandler
import os
import numpy as np
from model_factory import ModelHandler
import torch
import pickle


def make_positive(x, exponentiate=False, min=1e-8, max=1e3):
    """Make sure x is positive."""
    if exponentiate:
        return torch.clamp(torch.exp(x), min=min, max=max)
    else:
        return torch.clamp(torch.nn.functional.softplus(x), min=min, max=max)


def compute_rate_shape_for_gamma(mean, var):
    """Given Gamma(a, b) the mean is a/b and the variance is a/b^2"""
    rate = mean**2 / var
    shape = mean / var
    # print(f"Rate: {rate}, Shape: {shape}")
    return rate, shape


def compute_shape_scale_for_gamma(mean, var):
    """Given Gamma(k, theta) the mean is k*theta and the variance is k*theta^2"""
    shape = mean**2 / var
    scale = var / mean
    # print(f"Shape: {shape}, Scale: {scale}")
    return shape, scale


# given the mean and variance, compute the parameters of the negative binomial distribution
def get_params_for_nb(mean, var):
    # return n and p
    # NB's parametrization in numpy is P(N; n, p) = \binom{N+n-1}{N} p^n (1-p)^N
    # wikipedia: NB(k; r, p) = \binom{k+r-1}{k} p^r (1-p)^k
    # the equivalene is (first numpy, then wikipedia): (N is K, n is r, p is p)
    # mean is n(1-p)/p
    # variane is n(1-p)/p^2
    p = mean / var
    # n = p**2 * var / (1 - p)
    n = mean**2 / (var - mean)
    return n, p


def simulate_from_twin_eb(n_latents=None, n_obs=None, n_features=None, n_mix_rows=None, n_mix_columns=None, seed=0):
    """
    TWinEBModel:
    Z, W for row and column-wise r.v.s

    Z_i \in \matchcal{R}^{L}
    z_i | \omega_i ~ Gamma(a_{\omega_i}, b_{\omega_i})
    W_j \in \matchcal{R}^{L}
    w_j | \omega_j ~ Gamma(a_{\omega_j}, b_{\omega_j})

    X_{ij} ~ Poisson(z_i^T w_j)

    ---------------------------------

    Z_i \in \matchcal{R}^{L}
    Z_i ~ \sum_{k=1}^K_rows \omega_k^row Gamma(a_k^row, b_k^row)

    W_j \in \matchcal{R}^{L}
    W_j ~ \sum_{k=1}^K_columns \omega_k^column Gamma(a_k^column, b_k^column)

    1. Sample each \alpha_k^row, \beta_k^row, from a prior uniformly from a positive range [0, 10]
    2. Sample each \alpha_k^column, \beta_k^column, from a prior uniformly from a positive range [0, 10]
    3. Sample each \omega_k^row from a prior Dirichlet(1, 1, ..., 1)
    4. Sample each \omega_k^column from a prior Dirichlet(1, 1, ..., 1)
    5. Sample each z_i from a mixture of gammas with the given parameters
    6. Sample each w_j from a mixture of gammas with the given parameters
    7. Sample X_{ij} ~ Poisson(z_i^T w_j)
    """

    def _inner_sample_mixture(K, L, n_vars, max_alpha=5, max_beta=3, concentration=10):
        """
        Sample a vector of length n_vars by L, from a mixture of K gammas
        """
        #alpha = np.random.uniform(0, max_alpha, size=(K, L))
        alpha = np.random.gamma(.5, 1, size=(K, L))
        # beta = np.random.uniform(0, max_beta, size=(K, L))
        beta = np.random.gamma(.5, 1, size=(K, L))
        # 2. sample the weights of the gamma
        omega = np.random.dirichlet(np.ones(K)*concentration, size=1)
        # 3. sample where n_var from 1...K (the component assingment)
        component = np.random.choice(np.arange(K), size=n_vars, p=omega.squeeze())
        # print(component)
        return np.random.gamma(alpha[component, :], beta[component, :], size=(n_vars, L))
    
    def _inner_sample_mixture_separated(K, L, n_vars, max_var=5, concentration=10):
        """
        Sample a vector of length n_vars by L, from a mixture of K gammas
        Uses Mean and Variance
        """
        # shuffle it for each L 
        means = np.zeros((K, L))
        tmp = np.arange(K) + 1
        for l in range(L):
            means[:, l] = np.random.permutation(tmp)
        vars = np.random.uniform(1e2, max_var, size=(K, L))
        alpha, beta = compute_rate_shape_for_gamma(means, vars)
        # 2. sample the weights of the gamma
        omega = np.random.dirichlet(np.ones(K)*concentration, size=1)
        # 3. sample where n_var from 1...K (the component assingment)
        component = np.random.choice(np.arange(K), size=n_vars, p=omega.squeeze())
        return np.random.gamma(alpha[component, :], beta[component, :], size=(n_vars, L))
        
    
    np.random.seed(seed)
    # 1. Sample Zs
    # Zs = _inner_sample_mixture_separated(n_mix_rows, n_latents, n_obs)
    # # 2. Sample Ws
    # Ws = _inner_sample_mixture_separated(n_mix_columns, n_latents, n_features)
    Zs = _inner_sample_mixture(n_mix_rows, n_latents, n_obs)
    # 2. Sample Ws
    Ws = _inner_sample_mixture(n_mix_columns, n_latents, n_features)
    # 3. Sample Xs
    X = np.random.poisson(np.dot(Zs, Ws.T))
    the_dict = {
        "X": X,
        "row_latents": Zs,
        "col_latents": Ws,
    }
    return the_dict


def simulate_mixture_X(K, N, D, means, vars, seed=0, noise_config=None, verbose=False):
    """
    Simulate each entry from a mixture of Gammas with the Given means and variances
    """

    def _sample_from_mixture(means, vars, the_shape=(1,), components_weights=None):
        n_components = len(means)
        if components_weights is None:
            components_weights = np.ones(n_components) / n_components
        # sample a component
        component = np.random.choice(
            np.arange(n_components), p=components_weights, size=the_shape
        )
        # sample from the component
        a, b = compute_shape_scale_for_gamma(
            np.array(means)[component], np.array(vars)[component]
        )
        return np.random.gamma(a, b, size=the_shape)

    np.random.seed(seed)
    # prepare the latents
    rows_latents_shape = (N, K)
    col_latents_shape = (D, K)

    row_latents = np.zeros(rows_latents_shape)
    col_latents = np.zeros(col_latents_shape)

    # simulate each row and column latent from the mixture of gammas
    # simulate the k-th row latent
    row_latents[:, :] = _sample_from_mixture(means, vars, rows_latents_shape)
    # simulate the k-th column latent
    col_latents[:, :] = _sample_from_mixture(means[::-1], vars[::-1], col_latents_shape)

    X = np.random.poisson(np.dot(row_latents, col_latents.T))
    the_dict = {
        "X": X,
        "row_latents": row_latents,
        "col_latents": col_latents,
    }
    return the_dict


def simulate_structured_X(
    K,
    N,
    D,
    means,
    vars,
    seed=0,
    noise_config=None,
    verbose=False,
    family="poisson",
    prior_family="gamma",
    family_params=None,
    uniform_clusters=False
):
    """
    Parameters:
    ------------
    family_params: dict with keys: 'sigma' if the family (llhood) is normal
        
    """
    def printv(*args):
        if verbose:
            print(*args)

    np.random.seed(seed)
    # prepare the latents
    row_latents = np.zeros((N, K))
    col_latents = np.zeros((D, K))

    if uniform_clusters:
        # a. parition N, into K clusters
        row_clusts = np.random.randint(0, K, N)
        # b. partition D into K clusters
        col_clusts = np.random.randint(0, K, D)
    else:
        # use multinomial sampling
        # sample an unequal simplex
        simplex = np.random.dirichlet(np.arange(1, K + 1))
        row_clusts = np.random.choice(np.arange(K), size=N, p=simplex)
        col_clusts = np.random.choice(np.arange(K), size=D, p=simplex)


    # ensure that the rows from the same cluster are contiguous
    row_clusts = np.sort(row_clusts)
    # ensure that the columns from the same cluster are contiguous
    col_clusts = np.sort(col_clusts)

    # print the number of rows in each row cluster
    printv([(row_clusts == i).sum() for i in range(K)])
    # print the number of columns in each column cluster
    printv([(col_clusts == i).sum() for i in range(K)])

    for i, var in enumerate(vars):
        if prior_family == "gamma":
            a, b = compute_shape_scale_for_gamma(means[i], var)
            # simulate (row_clusts == i).sum() by K IID Gamma(a, b) RVs
            row_latents[row_clusts == i, i] = np.random.gamma(
                a, b, ((row_clusts == i).sum())
            )
        elif prior_family == "normal":
            a, b = means[i], np.sqrt(var)
            row_latents[row_clusts == i, i] = np.random.normal(
                a, b, ((row_clusts == i).sum())
            )

        # print the empirical mean and variance of the row latents
        em_mean = row_latents[row_clusts == i, i].mean()
        em_var = row_latents[row_clusts == i, i].var()
        # round the empirical mean and variance to 3 decimal places
        em_mean, em_var = round(em_mean, 3), round(em_var, 3)
        printv(
            f"Empirical mean: {em_mean}/{means[i]}, Empirical variance: {em_var}/{var}"
        )

    # reverse the order of vars
    for i, var in enumerate(vars[::-1]):
        # for i, var in enumerate(vars):
        if prior_family == "gamma":
            a, b = compute_shape_scale_for_gamma(means[i], var)
            col_latents[col_clusts == i, i] = np.random.gamma(
                a, b, ((col_clusts == i).sum())
            )
        elif prior_family == "normal":
            a, b = means[i], np.sqrt(var)
            col_latents[col_clusts == i, i] = np.random.normal(
                a, b, ((col_clusts == i).sum())
            )
        # print the empirical mean and variance of the col latents
        em_mean = col_latents[col_clusts == i, i].mean()
        em_var = col_latents[col_clusts == i, i].var()
        em_mean, em_var = round(em_mean, 3), round(em_var, 3)
        printv(
            f"Empirical mean: {em_mean}/{means[i]}, Empirical variance: {em_var}/{var}"
        )

    if family == "poisson":
        # X ~ Poisson(row_latents * col_latents) \in R^{N x D}
        X = np.random.poisson(np.dot(row_latents, col_latents.T))
    elif family == "normal":
        X = np.random.normal(np.dot(row_latents, col_latents.T), family_params["sigma"])

    # add noise to the data
    if noise_config is not None:
        # add noise of the family noise_config["family"], and with the given mean and variance
        if noise_config["family"] == "poisson":
            X = X + np.random.poisson(noise_config["rate"], size=X.shape)
        elif noise_config["family"] == "negative_binomial":
            # here just simulate X again, but have the mean as the mean of the negative binomial,
            the_mean = np.dot(row_latents, col_latents.T)
            # change zeros to noise_config['mean']
            the_mean[the_mean == 0] = noise_config["mean"]
            the_var = noise_config["var"]
            # compute n, p for the negative binomial
            n, p = get_params_for_nb(the_mean, the_var)
            X = X + np.random.negative_binomial(n, p, size=X.shape)
        elif noise_config["family"] == "negative_binomial_additive":
            n, p = get_params_for_nb(noise_config["mean"], noise_config["var"])
            X = X + np.random.negative_binomial(n, p, size=X.shape)
        else:
            raise ValueError(f"Unrecognized noise family {noise_config['family']}")

    the_dict = {
        "X": X,
        "row_latents": row_latents,
        "col_latents": col_latents,
        "row_clusts": row_clusts,
        "col_clusts": col_clusts,
    }
    return the_dict


def simulate_structured_X_normal(K, N, D, mean, vars):
    # prepare the latents
    row_latents = np.zeros((N, K))
    col_latents = np.zeros((D, K))

    # a. parition N, into J clusters
    row_clusts = np.random.randint(0, K, N)
    # b. partition D into K clusters
    col_clusts = np.random.randint(0, K, D)

    # ensure that the rows from the same cluster are contiguous
    row_clusts = np.sort(row_clusts)
    # ensure that the columns from the same cluster are contiguous
    col_clusts = np.sort(col_clusts)

    for i in range(len(row_clusts)):
        if i in range(0, 10):
            row_clusts[i] = 0
        elif i in range(10, 60):
            row_clusts[i] = 1
        elif i in range(60, 150):
            row_clusts[i] = 2
        else:
            raise ValueError(f"row_clusts out of range {i} vs {len(row_clusts)}")

    for j in range(len(col_clusts)):
        # 1 - 80, 81 - 160, 161 - 240
        if j in range(0, 80):
            col_clusts[j] = 0
        elif j in range(80, 160):
            col_clusts[j] = 1
        elif j in range(160, 240):
            col_clusts[j] = 2
        else:
            raise ValueError(f"col_clusts out of range {j} vs {len(col_clusts)}")

    # print the number of rows in each row cluster
    print([(row_clusts == i).sum() for i in range(K)])
    # print the number of columns in each column cluster
    print([(col_clusts == i).sum() for i in range(K)])

    # i and k co-vary
    for i, var in enumerate(vars):
        a, b = mean, var
        row_latents[row_clusts == i, i] = np.random.normal(
            a, b, ((row_clusts == i).sum())
        )
        # print the empirical mean and variance of the row latents
        em_mean = row_latents[row_clusts == i, i].mean()
        em_var = row_latents[row_clusts == i, i].var()
        # round the empirical mean and variance to 3 decimal places
        em_mean, em_var = round(em_mean, 3), round(em_var, 3)
        print(f"Empirical mean: {em_mean}/{mean}, Empirical variance: {em_var}/{var}")

    # reverse the order of vars
    for i, var in enumerate(vars[::-1]):
        a, b = mean, var
        col_latents[col_clusts == i, i] = np.random.normal(
            a, b, ((col_clusts == i).sum())
        )
        # print the empirical mean and variance of the col latents
        em_mean = col_latents[col_clusts == i, i].mean()
        em_var = col_latents[col_clusts == i, i].var()
        em_mean, em_var = round(em_mean, 3), round(em_var, 3)
        print(f"Empirical mean: {em_mean}/{mean}, Empirical variance: {em_var}/{var}")

    # X ~ Poisson(row_latents * col_latents) \in R^{N x D}
    X = np.random.normal(np.dot(row_latents, col_latents.T), 2)
    the_dict = {
        "X": X,
        "row_latents": row_latents,
        "col_latents": col_latents,
        "row_clusts": row_clusts,
        "col_clusts": col_clusts,
    }
    return the_dict


def row_norm(X):
    """comptue row sums"""
    row_sums = X.sum(axis=1)
    row_sums[row_sums == 0] = 1
    # noramlize X
    return X / row_sums[:, None]


# given a matrix, cluster its rows and columns and reorder them
def reorder_X(X, method="complete"):
    # cluster the rows and columns
    row_linkage = linkage(X, method=method)
    col_linkage = linkage(X.T, method=method)

    # reorder the rows and columns
    row_idx = dendrogram(row_linkage, no_plot=True)["leaves"]
    col_idx = dendrogram(col_linkage, no_plot=True)["leaves"]

    # reorder the matrix
    X_reordered = X[row_idx, :]
    X_reordered = X_reordered[:, col_idx]
    return X_reordered


def getPicklePath(expPath):
    # read the config file and load the
    exp_handler = ExperimentHandler(expPath)
    return exp_handler.config["picklePath"]


def unpermute(the_array, skip=1, subtract=1):
    the_array = [i[skip:] for i in the_array.copy()]
    the_array = [(int(i) - subtract) for i in the_array]
    return np.argsort(the_array)


def get_unpermuate(expPath, out=False):
    picklePath = getPicklePath(expPath)
    with open(picklePath, "rb") as f:
        data = pickle.load(f)
    data = data.heldout_data if out else data
    unpermute_rows = unpermute(data.obs_names)
    unpermute_cols = unpermute(data.feature_names)
    return unpermute_rows, unpermute_cols


def get_xout_factors(expPath):
    """
    Return original heldout latents

    1. Return the subsampled rows
    2. The rows and columns have been permuted, return them back to the original order
    """
    picklePath = getPicklePath(expPath)
    with open(picklePath, "rb") as f:
        data = pickle.load(f)
    original_data_path = data.original_data_path
    data = data.heldout_data
    # remove o at the begining of the obsnames
    obsnames = [i[1:] for i in data.obs_names]
    obsnames = [(int(i) - 1) for i in obsnames]
    varnames = [i[1:] for i in data.feature_names]
    varnames = [(int(i) - 1) for i in varnames]
    # permuate the obs and vars back
    unpermute_rows = np.argsort(obsnames)
    unpermute_cols = np.argsort(varnames)
    # load the original (unsubsampled) data
    adata = sc.read_h5ad(original_data_path)

    obsnames = np.sort(obsnames)
    # get the original factors
    row_latents_ = adata.obsm["X_row_latents"][obsnames, :].copy()
    col_latents_ = adata.varm["X_col_latents"].copy()
    adata_unpermute_rows = unpermute(adata.obs_names)
    adata_unpermute_cols = unpermute(adata.var_names)
    adata = adata[adata_unpermute_rows, :][:, adata_unpermute_cols]

    x_in = data.train.copy()
    x_in = x_in[unpermute_rows, :][:, unpermute_cols]
    return x_in, row_latents_, col_latents_


def get_xin_factors(expPath):
    """
    Return original latents

    1. The rows (obs) are subsampled, return the true rows
    2. The rows and columns have been permuted, return them back to the original order
    """
    picklePath = getPicklePath(expPath)
    with open(picklePath, "rb") as f:
        data = pickle.load(f)
    # remove o at the begining of the obsnames
    obsnames = [i[1:] for i in data.obs_names]
    obsnames = [(int(i) - 1) for i in obsnames]
    varnames = [i[1:] for i in data.feature_names]
    varnames = [(int(i) - 1) for i in varnames]
    # permuate the obs and vars back
    unpermute_rows = np.argsort(obsnames)
    unpermute_cols = np.argsort(varnames)
    # load the original (unsubsampled) data
    adata = sc.read_h5ad(data.original_data_path)

    obsnames = np.sort(obsnames)
    # get the original factors
    # row_latents_ = adata.obsm["X_row_latents"][obsnames, :]
    row_latents_ = adata.obsm["X_row_latents"][obsnames, :].copy()
    col_latents_ = adata.varm["X_col_latents"].copy()
    adata_unpermute_rows = unpermute(adata.obs_names)
    adata_unpermute_cols = unpermute(adata.var_names)
    adata = adata[adata_unpermute_rows, :][:, adata_unpermute_cols]
    # vv = row_latents_ @  col_latents_.T
    # quick_plot(vv.copy(), outPath)
    # quick_plot(adata.X.A, outPath)

    x_in = data.train.copy()
    x_in = x_in[unpermute_rows, :][:, unpermute_cols]
    return x_in, row_latents_, col_latents_


# def compute_mean_for_log_norma(mu, scale):
#     """Given the params mu, scale, return the mean"""
#     return np.exp(mu + scale**2 / 2)


def extract_latents(expPath):
    """
    Return inferered latents
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"The device is ! {device}")
    model = ModelHandler.load_model(expPath).to(device)
    row_l = model.row_distribution.sample(1).cpu().detach().numpy()
    col_l = model.column_distribution.sample(1).cpu().detach().numpy()
    # return the means too
    # based on the var_family of the model
    # ro_l_mean = compute_mean_for_log_norma(model.row_distribution.location.cpu().detach().numpy(), model.row_distribution.scale().cpu().detach().numpy())
    # col_l_mean = compute_mean_for_log_norma(model.column_distribution.location.cpu().detach().numpy(), model.column_distribution.scale().cpu().detach().numpy())
    # check if the model name has amortized, in that case, pass on the data
    if "Amortized" in model.__class__.__name__:
        # load the data
        picklePath = getPicklePath(expPath)
        with open(picklePath, "rb") as f:
            data = pickle.load(f)
        dat_train = data.train
        ro_l_mean = model.row_mean(dat_train)
    else:
        ro_l_mean = model.row_mean
    # squeeze the factors and loadings
    col_l_mean = model.column_mean
    row_l = row_l.squeeze()
    col_l = col_l.squeeze()
    # Fix permuation
    unpermute_rows, unpermute_cols = get_unpermuate(expPath)
    ro_l_mean = ro_l_mean[:, unpermute_rows]
    col_l_mean = col_l_mean[unpermute_cols, :]
    row_l = row_l[:, unpermute_rows]
    col_l = col_l[unpermute_cols, :]
    return row_l, col_l, ro_l_mean, col_l_mean


def extract_out_latents(expPath):
    """
    Return inferered latents
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"The device is ! {device}")
    model = ModelHandler.load_model(expPath, retrain=True).to(device)
    row_l = model.row_distribution.sample(1).cpu().detach().numpy()
    col_l = model.column_distribution.sample(1).cpu().detach().numpy()
    # return the means too
    # check if the model name has amortized, in that case, pass on the data
    if "Amortized" in model.__class__.__name__:
        # load the data
        picklePath = getPicklePath(expPath)
        with open(picklePath, "rb") as f:
            data = pickle.load(f)
        dat_train = data.train
        ro_l_mean = model.row_mean(dat_train)
    else:
        ro_l_mean = model.row_mean
    # squeeze the factors and loadings
    col_l_mean = model.column_mean
    row_l = row_l.squeeze()
    col_l = col_l.squeeze()
    # Fix permuation
    unpermute_rows, unpermute_cols = get_unpermuate(expPath, out=True)
    ro_l_mean = ro_l_mean[:, unpermute_rows]
    col_l_mean = col_l_mean[unpermute_cols, :]
    row_l = row_l[:, unpermute_rows]
    col_l = col_l[unpermute_cols, :]
    return row_l, col_l, ro_l_mean, col_l_mean


def plot_reconstruction(
    ro_l_mean, col_l_mean, row_latents_, col_latents_, outPath, tag=""
):
    true_rate = row_latents_ @ col_latents_.T
    infered_rate = ro_l_mean.T @ col_l_mean.T
    # compute RMSE between the true and infered rates
    rmse = np.sqrt(np.mean((true_rate - infered_rate) ** 2))
    print(f"RMSE{tag}: {rmse}")
    # also compute MAE
    mae = np.mean(np.abs(true_rate - infered_rate))
    print(f"MAE{tag}: {mae}")

    # Run abs on all
    true_rate = np.abs(true_rate)
    infered_rate = np.abs(infered_rate)
    row_latents_ = np.abs(row_latents_)
    col_latents_ = np.abs(col_latents_)
    ro_l_mean = np.abs(ro_l_mean)
    col_l_mean = np.abs(col_l_mean)

    plt.clf()
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), dpi=300)
    im1 = axes[0, 0].imshow(
        ro_l_mean, cmap="gray_r", interpolation="none", aspect="auto"
    )
    im2 = axes[0, 1].imshow(
        row_latents_.T, cmap="gray_r", interpolation="none", aspect="auto"
    )
    im3 = axes[1, 0].imshow(
        col_l_mean.T, cmap="gray_r", interpolation="none", aspect="auto"
    )
    im4 = axes[1, 1].imshow(
        col_latents_.T, cmap="gray_r", interpolation="none", aspect="auto"
    )
    im5 = axes[2, 0].imshow(
        infered_rate, cmap="gray_r", interpolation="none", aspect="auto"
    )
    im6 = axes[2, 1].imshow(
        true_rate, cmap="gray_r", interpolation="none", aspect="auto"
    )

    axes[0, 0].set_title("row_latents_")
    axes[0, 1].set_title("row_latents (original)")
    axes[1, 0].set_title("col_latents_")
    axes[1, 1].set_title("col_latents (original)")
    axes[2, 0].set_title("row_l.T @ col_l.T")
    axes[2, 1].set_title("row_latents_ @ col_latents_.T (original)")
    # add colorbars
    fig.colorbar(im1, ax=axes[0, 0])
    fig.colorbar(im2, ax=axes[0, 1])
    fig.colorbar(im3, ax=axes[1, 0])
    fig.colorbar(im4, ax=axes[1, 1])
    fig.colorbar(im5, ax=axes[2, 0])
    fig.colorbar(im6, ax=axes[2, 1])
    # reduce the space between the plots
    fig.subplots_adjust(hspace=0.3)
    # set title for the plot
    fig.suptitle(f"RMSE: {rmse} - MAE: {mae}")
    plt.savefig(outPath, bbox_inches="tight")
    plt.close()
    return rmse, mae


def viz_latents_holl_exp_path(expPath, fig_dir, use_mean=True):
    """
    Visualizes the heldout rows
    """
    # extract original factors
    x_in, row_latents_, col_latents_ = get_xout_factors(expPath)
    # extract reconstructed factors
    row_l, col_l, ro_l_mean, col_l_mean = extract_out_latents(expPath)
    row_l.shape, col_l.shape, ro_l_mean.shape, col_l_mean.shape
    # plot the factors
    rmse, mae = None, None
    if use_mean:
        outPath = os.path.join(fig_dir, f"means_factors_out.png")
        rmse, mae = plot_reconstruction(
            ro_l_mean, col_l_mean, row_latents_, col_latents_, outPath, " out"
        )
    else:
        outPath = os.path.join(fig_dir, f"factors_out.png")
        rmse, mae = plot_reconstruction(
            row_l, col_l, row_latents_, col_latents_, outPath, " out"
        )

    # save rmse
    tmp = {"rmse_out": float(rmse), "mae_out": float(mae)}
    configHandler = ConfigHandler(expPath=expPath)
    configHandler.write_updated_config(**tmp)


def viz_latents_exp_path(expPath, fig_dir, use_mean=True):
    # extract original factors
    x_in, row_latents_, col_latents_ = get_xin_factors(expPath)
    # extract reconstructed factors
    row_l, col_l, ro_l_mean, col_l_mean = extract_latents(expPath)
    row_l.shape, col_l.shape, ro_l_mean.shape, col_l_mean.shape
    # plot the factors
    rmse, mae = None, None
    if use_mean:
        outPath = os.path.join(fig_dir, f"means_factors.png")
        rmse, mae = plot_reconstruction(
            ro_l_mean, col_l_mean, row_latents_, col_latents_, outPath
        )
    else:
        outPath = os.path.join(fig_dir, f"factors.png")
        rmse, mae = plot_reconstruction(
            row_l, col_l, row_latents_, col_latents_, outPath
        )

    # save rmse
    tmp = {"rmse": float(rmse), "mae": float(mae)}
    configHandler = ConfigHandler(expPath=expPath)
    configHandler.write_updated_config(**tmp)


# var-NONE:
# Heldout llhood: -0.560562331944004
# ELBO:  -9648760.295070538
# var-.5:
# Heldout llhood: -0.5566434427626422
# ELBO:  -9714711.613914184
# Heldout llhood: -0.5583052121085453
# ELBO:  -9651355.148534453
