## Load the resutls and generate w from them, then k-means them
import argparse
import scanpy as sc
from model_factory import ModelHandler
from utils import (
    BatchHandler,
    ExperimentHandler,
    sparse_tensor_from_sparse_matrix,
    str2bool,
    check_columns,
    runCMD,
)
from utils import ConfigHandler
import torch
import anndata
import os
import pandas as pd
import pickle
import numpy as np
import torch
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import yaml
from scipy.stats import entropy
from scipy.cluster import hierarchy
from sklearn.metrics.cluster import adjusted_rand_score
import sys
import matplotlib.pyplot as plt
import seaborn as sns


disney_movies = [
    "aladdin (",
    "little mermaid",
    "lion king",
    "beauty and the beast",
    "bambi",
    "dumbo (",
    "cinderella (",
    "alice in wonderland",
    "peter pan",
    "lady and the tramp",
    "sleeping beauty",
    "one hundred and one dalmatians",
    "the sword in the stone",
    "robin hood (",
    "jungle book, the",
    "pocahontas",
    "toy story",
    "hercules",
    "tarzan (",
]
horror_movies = [
    "exorcist",
    "shining",
    "alien: resurrection",
    "alien (",
    "aliens (",
    "alien³",
    "interview with the vampire",
    "jacob",
    "nightmare on elm street",
    "thing, the",
    "fly, the (1986",
    "fly, the (1958",
    "birds",
    "texas chain saw massacre",
    "scream (",
]
horror_movies_excempt = ["Boy Who Could Fly, The (1986)", "Next Best Thing, The (2000)"]

scifi_movies = [
    "star wars",
    "star trek",
    "fantastic planet",
    "2001",
    "terminator",
    "jurassic",
]

# used for lolipop plots
TEST_MOVIES = [
    "aladdin",
    "toy story",
    "star wars",
    "terminator 2",
    "terminator,",
    "matrix",
    "harry potter",
    "lord of the rings",
    "the godfather",
    "jaws",
    "shining",
    "the lion king",
    "the little marmaid",
    "american beauty",
    'alien: resurrection',
    'jurassic park',
]

# NB: request a GPU
# bsub -q gpuqueue -R A100 -n 1 -gpu "num=1"  -W 1:00 -Is -R "rusage[mem=10]" /bin/bash

# run nmf


def compute_nmf(
    X=None,
    n_components=None,
    random_state=0,
    max_iter=2000,
    beta_loss="frobenius",
    solver="cd",
):
    # beta_loss = kullback-leibler
    nmf = NMF(
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
        beta_loss=beta_loss,
        solver=solver,
    )
    Z_nmf = nmf.fit_transform(X)
    W_nmf = nmf.components_
    X_inv = nmf.inverse_transform(Z_nmf)
    return Z_nmf, W_nmf, X_inv

def compute_mean_for_log_normal(mu, scale):
    """Given the params mu, scale, return the mean"""
    # check that all elements in scale are positive
    assert np.all(scale > 0), f"scale must be > 0, but it is {scale}"
    return np.exp(mu + scale**2 / 2)


def create_factors(expPath, nSamples=1):
    # if cuda is available, set the device to cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # model = ModelHandler.load_model(expPath, device='cpu').to(device)
    print(f"The device is ! {device}")
    model = ModelHandler.load_model(expPath).to(device)
    
    factors = None
    for i in range(nSamples):
        tmp = model.column_distribution.sample(1)
        factors = (tmp + factors) if factors is not None else tmp
    # average over samples
    factors = factors / nSamples

    # create an adata object from the factors
    # if device is cuda, use .cpu too
    if device == torch.device("cuda"):
        factors = factors.cpu()
    adata = anndata.AnnData(X=factors.squeeze().detach().numpy())
    # save the adata object
    adata.write_h5ad(os.path.join(expPath, "factors.h5ad"))
    # also look at the mean of the factors
    # the_mean = model.qv_distribution.location
    # the_loc = model.column_distribution.location.cpu().detach().numpy()
    # the_scale = model.column_distribution.scale().cpu().detach().numpy()
    # the_mean = compute_mean_for_log_normal(the_loc, the_scale)
    the_mean = model.column_mean
    # if device == torch.device("cuda"):
    #     the_mean = the_mean.cpu()
    adata = anndata.AnnData(X=the_mean.squeeze())
    # save the adata object
    adata.write_h5ad(os.path.join(expPath, "factors_mean.h5ad"))


# Download the adata object
def download_original_data():
    with open(
        "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)
        adata = anndata.AnnData(X=data.train)
        adata.write_h5ad(
            os.path.join(
                "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/",
                "data_train.h5ad",
            )
        )


# download_original_data()
def read_movies(dataset):
    """Read the original data and extract the movie names"""
    # TODO: check if you're on ceto, lilac, or local
    # check if "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/ml-1m.h5ad" exists
    if dataset == "ml-100k" or dataset == "ml-1m":
        if os.path.exists("/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/ml-1m.h5ad"):
            h5adPath = "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/ml-1m.h5ad"
            featureNamesPath = "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0_feature_names.csv"
        elif os.path.exists("/Users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/ml-1m.h5ad"):
            h5adPath = "/Users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/ml-1m.h5ad"
            featureNamesPath = "/Users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0_feature_names.csv"
        elif os.path.exists("/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/ml-1m.h5ad"):
            h5adPath = "/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/ml-1m.h5ad"
            featureNamesPath = "/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0_feature_names.csv"
        else:
            raise ValueError("Unknown machine")
    elif dataset == "goodreads":
        featureNamesPath = None
        if os.path.exists("/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads.h5ad"):
            h5adPath = "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads.h5ad"
        elif os.path.exists("/Users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads.h5ad"):
            h5adPath = "/Users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads.h5ad"
        elif os.path.exists("/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads.h5ad"):
            h5adPath = "/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads.h5ad"
        else:
            raise ValueError("Unknown machine")
    elif dataset == 'goodreads_5k':
        featureNamesPath = None
        if os.path.exists("/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads_5k.h5ad"):
            h5adPath = "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads_5k.h5ad"
        elif os.path.exists("/Users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads_5k.h5ad"):
            h5adPath = "/Users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads_5k.h5ad"
        elif os.path.exists("/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads_5k.h5ad"):
            h5adPath = "/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/goodreads_5k.h5ad"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # load the original data
    bdata = sc.read(h5adPath)
    uu = bdata.var
    # just keep rows of uu that are in df.feature_names
    # change feature_names to strings
    if dataset == "goodreads":
        # the order is contigous for the goodreads dataset
        uu = uu[['title']].copy()
        # rename original_title to movie_name
        uu.rename(columns={'title': 'movie_name'}, inplace=True)
        # add feature_indx as the index
        uu['feature_indx'] = uu.index.values#
        #uu.sort_values(by='feature_indx', inplace=True)
        # DO NOT SORT BY feature_indx, they are kept the same as the original dataset
        return uu
    elif dataset == 'goodreads_5k':
        uu = uu[['title_year']].copy()
        # rename goodreads_5k to movie_name
        uu.rename(columns={'title_year': 'movie_name'}, inplace=True)
        # add feature_indx as the index
        uu['feature_indx'] = uu.index.values
        # DO NOT SORT BY feature_indx, they are kept the same as the original dataset
        #uu.sort_values(by='feature_indx', inplace=True)
        return uu
    else:
        df = pd.read_csv(featureNamesPath)
    df["feature_names"] = df["feature_names"].astype(str)
    # set df['feature_names'] to index
    df.set_index("feature_names", inplace=True)
    # join df and uu
    df = df.join(uu, on="feature_names", how="inner")
    df.reset_index(inplace=True)
    # sort df by feature_indx
    df.sort_values(by="feature_indx", inplace=True)
    return df


def prepare_factors(expPath, ignore_cashed=True, means=False): 
    """Adds the movie names to the factors object"""
    file_name = "factors_mean.h5ad" if means else "factors.h5ad"
    file_path = os.path.join(expPath, file_name)
    if os.path.exists(file_path) & (ignore_cashed is False):
        adata = sc.read(file_path)
        return adata
    # load the factors
    adata = sc.read(file_path)
    adata.layers["counts"] = adata.X.copy()
    if "gene_names" in adata.obs.columns:
        adata.obs["movie_names"] = adata.obs["gene_names"].values
    else:
        dataset = getDatasetNameFromExpPath(expPath)
        df = read_movies(dataset)
        adata.obs["movie_names"] = df["movie_name"].values
    # save the adata object
    adata.write_h5ad(file_path)
    return adata


def quick_cluster(adata, expPath, ignore_cashed=True, means=False, resolution=0.5):
    """
    Cluster using scanpy and save the results
    """
    file_name = "factors_mean.h5ad" if means else "factors.h5ad"
    file_path = os.path.join(expPath, file_name)
    if os.path.exists(file_path) & (ignore_cashed is False):
        adata = sc.read(file_path)
        return adata
    # cluster and pca and etc...
    adata.X = adata.layers["counts"].copy()
    # scale the data
    sc.pp.scale(adata)
    # pca
    sc.pp.pca(adata)
    # neighbors
    sc.pp.neighbors(adata)
    # umap
    print("Computing UMAP...")
    sc.tl.umap(adata)
    # leiden
    print("Computing Leiden...")
    sc.tl.leiden(adata, resolution=resolution, key_added=f"leiden_r{resolution}")
    adata.write(file_path)
    return adata


def extract_clusters(assignments, labels):
    """Extract clusters from user preferences or item assignments via K-means.
    Always returns n_attributes clusers.
    Input:
        assignments: a tensor/array of shape (n_nodes, n_attributes) for user
            preferences or item assignments
        labels: a dataframe with two columns, the first one is the node id (int) and
            the second one is the node label (str)
    """
    # compute similarities
    n_attributes = assignments.shape[1]
    # normalize assignments
    if isinstance(assignments, torch.Tensor):
        assignments = assignments.detach().numpy()
    assignments = np.diag(np.linalg.norm(assignments, axis=1) ** (-1)) @ assignments
    # set all NaNs to 0
    # count the number of nans in assignments
    n_nans = np.sum(np.isnan(assignments))
    if n_nans > 0:
        print(f"Warning: {n_nans} NaNs in assignments")
    assignments[np.isnan(assignments)] = 0
    # run K-means on similarities
    kmeans = KMeans(n_clusters=n_attributes, random_state=0, n_init=10).fit(assignments)
    # compute distance of each assignment from centroids
    centroids = kmeans.cluster_centers_
    distances = np.zeros((n_attributes, assignments.shape[0]))
    for i in range(n_attributes):
        distances[i, :] = np.linalg.norm(assignments - centroids[i, :], axis=1)
    # extract cluster indices
    # clusters[i] contains the indexes of the nodes in cluster i
    clusters = {}
    for i in range(n_attributes):
        clusters[i] = np.where(kmeans.labels_ == i)[0]
    # sort indices by increasing distance from centroid
    clusters = [
        clusters[k][np.argsort(distances[k, clusters[k]])] for k in range(n_attributes)
    ]
    # convert the first column of labels to int
    labels[labels.columns[0]] = labels[labels.columns[0]].astype(int)
    # extract cluster labels
    clusters = [
        [labels[labels.iloc[:, 0] == int(x)].iloc[:, 1].values[0] for x in clusters[k]]
        for k in range(n_attributes)
    ]
    return clusters, kmeans.labels_


def leiden_to_df(adata, clust_col=None, n_top=None):
    """
    In each cluster, print the top 10 movies
    """
    assert clust_col is not None, "clust_col must be provided"
    assert clust_col in adata.obs.columns, f"{clust_col} not in adata.obs.columns"
    dt = None
    for i, cluster in enumerate(adata.obs[clust_col].unique()):
        # print(f'Cluster {cluster}')
        if n_top is None:
            movies = adata[adata.obs[clust_col] == cluster].obs.movie_names
        else:
            movies = adata[adata.obs[clust_col] == cluster].obs.movie_names[:10]
        # concatenate movies as a single string
        tmp = {"cluster": cluster, "movie_names": movies.str.cat(sep=", ")}
        # convert to pandas and ignore the index
        tmp = pd.DataFrame(tmp, index=[0])
        dt = tmp if dt is None else pd.concat([dt, tmp], axis=0, ignore_index=True)
    return dt


def clusts_to_dt(clusts, n_top=None):
    """
    Convert the clusters list obj to a dataframe.
    """
    # in each cluster, print the top 10 movies
    dt = None
    for i in range(len(clusts)):
        if n_top is None:
            movies = clusts[i]
        else:
            movies = clusts[i][:n_top]
        # print(movies)
        tmp = {"cluster": i, "movie_names": ", ".join(movies)}
        tmp = pd.DataFrame(tmp, index=[0])
        dt = tmp if dt is None else pd.concat([dt, tmp], axis=0, ignore_index=True)
    return dt


def handle_kmeans(adata, expPath):
    assignments = adata.X.copy()
    labels = adata.obs.movie_names.reset_index()
    # get sorted clusters
    clusts, kmeans_labels = extract_clusters(assignments=assignments, labels=labels)
    # save as a pickle file
    with open(os.path.join(expPath, "clusters_kmeans.pkl"), "wb") as f:
        pickle.dump(clusts, f)
    # also save the labels
    with open(os.path.join(expPath, "clusters_kmeans_labels.pkl"), "wb") as f:
        pickle.dump(kmeans_labels, f)
    dt_full = clusts_to_dt(clusts, n_top=None)
    dt_10 = clusts_to_dt(clusts, n_top=10)
    # save them
    dt_full.to_csv(os.path.join(expPath, "clusters_kmeans.tsv"), index=False, sep="\t")
    dt_10.to_csv(os.path.join(expPath, "clusters_kmeans_10.tsv"), index=False, sep="\t")
    return dt_10, clusts, kmeans_labels


def handle_neighbours(adata, expPath):
    # cluster movies and show the top 10 movies in each cluster
    adata = quick_cluster(adata, expPath)
    # save the results
    dt = leiden_to_df(adata, clust_col="leiden_r0.5")
    dt.to_csv(os.path.join(expPath, "clusters_leiden.csv"), index=False)
    # just the top 20
    dt = leiden_to_df(adata, clust_col="leiden_r0.5", n_top=10)
    dt.to_csv(os.path.join(expPath, "clusters_leiden_top_20.csv"), index=False)
    return dt


def summarize_factors(expPath, ignore_leiden=True, nSamples=1, means=False):
    create_factors(expPath, nSamples=nSamples)
    # Movie by factor object
    adata = prepare_factors(expPath, means=means)
    if ignore_leiden:
        # print('Ignoring Leiden')
        dt_leiden = None
    else:
        dt_leiden = handle_neighbours(adata, expPath)
    dt_kmeans, clusts_kmeans, labels_kmeans = handle_kmeans(adata, expPath)
    return dt_leiden, dt_kmeans, clusts_kmeans, labels_kmeans


def bb(uu, cutoff=3):
    """
    Binarize movie ratings.
    """
    assert cutoff is not None, "cutoff must be provided"
    assert cutoff >= 0, f"cutoff must be >= 0, but it is {cutoff}"
    if cutoff == 0:
        uu[uu > 0] = 1
        return uu
    uu[uu <= cutoff] = 0
    uu[uu > cutoff] = 1
    return uu


def summarize_factors_nmf(
    expPath, ignore_leiden=True, binarize=True, n_components=20, use_cache=False
):
    # Movie by factor object
    _ = handle_nmf(
        expPath, binarize=binarize, n_components=n_components, use_cache=use_cache
    )
    adata = prepare_factors(expPath)
    if ignore_leiden:
        # print('Ignoring Leiden')
        dt_leiden = None
    else:
        dt_leiden = handle_neighbours(adata, expPath)
    dt_kmeans, clusts_kmeans, labels_kmeans = handle_kmeans(adata, expPath)
    return dt_leiden, dt_kmeans, clusts_kmeans, labels_kmeans


def get_xin(binarize=True, dataset=None):
    assert dataset is not None, "dataset must be provided"
    if dataset in ["ml-100k", "ml-1m"]:
        cutoff = 3
        if os.path.exists("/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0_counts.pkl"):
            picklePath = "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0_counts.pkl"
        elif os.path.exists("/Users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0_counts.pkl"):
            picklePath = "/Users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0_counts.pkl"
        elif os.path.exists("/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0_counts.pkl"):
            picklePath = "/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/movie_lens/processed/ml-1m_0.2_1.0_counts.pkl"
        else:
            raise ValueError("Unknown machine")
    elif dataset == "goodreads":
        cutoff = 0
        # /work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_0.2_1.0_counts.pkl
        if os.path.exists("/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_0.2_1.0_counts.pkl"):
            picklePath = "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_0.2_1.0_counts.pkl"
        elif os.path.exists("/Users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_0.2_1.0_counts.pkl"):
            picklePath = "/Users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_0.2_1.0_counts.pkl"
        elif os.path.exists("/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_0.2_1.0_counts.pkl"):
            picklePath = "/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_0.2_1.0_counts.pkl"
        else:
            raise ValueError("Unknown machine")
    elif dataset == 'goodreads_5k':
        if os.path.exists("/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_5k_0.2_1.0_counts.pkl"):
            picklePath = "/data/De-identified Authorlab/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_5k_0.2_1.0_counts.pkl"
        elif os.path.exists("/Users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_5k_0.2_1.0_counts.pkl"):
            picklePath = "/Users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_5k_0.2_1.0_counts.pkl"
        elif os.path.exists("/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_5k_0.2_1.0_counts.pkl"):
            picklePath = "/work/De-identified Author/users/De-identified Authors/projects/rnaseq-pfm/data/goodreads/processed/goodreads_5k_0.2_1.0_counts.pkl"
        else:
            raise ValueError("Unknown machine")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    with open(picklePath, "rb") as f:
        data = pickle.load(f)
    x_in = data.train.copy()
    if binarize:
        x_in = bb(x_in.A, cutoff=cutoff)
    else:
        x_in = x_in.A
    return x_in


def get_picklePath(expPath):
    # read the config file and load the
    exp_handler = ExperimentHandler(expPath)
    picklePath = exp_handler.config['picklePath']
    # return the basename for this
    return os.path.basename(picklePath)

def getDatasetName(picklePath):
    for ii in ["ml-100k", "ml-1m", "goodreads_5k", "goodreads"]:
        if ii in picklePath:
            return ii
    return None

def getCutOffForDataset(datasetName):
    if datasetName in ["ml-100k", "ml-1m"]:
        return 3
    elif datasetName in ["goodreads", "goodreads_5k"]:
        return 0
    else:
        raise ValueError(f"Unknown dataset: {datasetName}")

def getCutoffFromExpPath(expPath):
    picklePath = get_picklePath(expPath)
    datasetName = getDatasetName(picklePath)
    return getCutOffForDataset(datasetName)


def getDatasetNameFromExpPath(expPath):
    picklePath = get_picklePath(expPath)
    return getDatasetName(picklePath)

def handle_nmf(
    expPath,
    binarize=False,
    n_components=20,
    use_cache=False,
    seed=1000,
    beta_loss="kullback-leibler",
    solver="mu",
    max_iter=30000,
):
    """Generate NMF"""
    if use_cache:
        adata_path = os.path.join(expPath, "factors.h5ad")
        if os.path.exists(adata_path):
            print("Using cached factors")
            return sc.read(adata_path)
    dataset = getDatasetNameFromExpPath(expPath)
    x_in = get_xin(binarize=binarize, dataset=dataset)
    np.random.seed(seed)
    z_nmf, w_nmf, x_nmf = compute_nmf(
        X=x_in,
        n_components=n_components,
        random_state=seed,
        beta_loss=beta_loss,
        solver=solver,
        max_iter=max_iter,
    )
    adata = anndata.AnnData(X=w_nmf.T)
    adata.write_h5ad(os.path.join(expPath, "factors.h5ad"))
    return adata


def compute_relative_entropy(some_vector):
    """
    Compute relative entropy for a given vector of repeated objects.
    if some_vectors has n_unique unique elements, each repeated p_i times, then compute:
    -sum(p_i * log(p_i)) / log(n_unique)
    """
    N = len(np.unique(some_vector))
    if N == 1:
        # return 1.0
        # all are the same
        N = len(some_vector)
        p_is = np.ones(N) / N
    else:
        p_is = np.unique(some_vector, return_counts=True)[1]
    # normalize p_is to sum 1
    p_is = p_is / np.sum(p_is)
    max_entropy = np.log(N)
    entropy = -np.sum(p_is * np.log(p_is))
    normalized_entropy = entropy / max_entropy
    # return rounded to 3 digits
    return np.round(normalized_entropy, 3)


def get_clusts(clusts, ref_movies, except_movies=None, silent=True):
    """
    Find the cluster for each movie in ref_movies that is in clusts and not in except_movies. 
    """
    _clusts = []
    for i in range(len(clusts)):
        movies = clusts[i]
        for movie in movies:
            for dm in ref_movies:
                if dm in movie.lower() and (
                    except_movies is None or movie not in except_movies
                ):
                    if silent is False:
                        print(f"Cluster {i} has {dm} in {movie}")
                    _clusts.append(i)
    return _clusts


def compute_clust_loss(clusts, weighted=False):
    if weighted is False:
        return (len(np.unique(clusts)) - 1) / len(clusts)
    else:
        # comptue the weighted mean
        raise NotImplementedError

def check_genere(clusts, ref_movies, except_movies=None, silent=True):
    # find the ref_movies
    _clusts = get_clusts(clusts, ref_movies, except_movies=except_movies, silent=silent)
    # report the variance
    loss = compute_clust_loss(_clusts)
    entropy_loss = compute_relative_entropy(_clusts)
    if silent is False:
        print(f"Loss: {loss} -- Entropy: {entropy_loss}")
    return loss, entropy_loss


def check_disney_movies(clusts, silent=True):
    return check_genere(clusts, disney_movies, silent=silent)


def check_horror_movies(clusts, silent=True):
    return check_genere(clusts, horror_movies, horror_movies_excempt, silent=silent)


def check_scifi_movies(clusts, silent=True):
    return check_genere(clusts, scifi_movies, silent=silent)


# def check_disney_movies(clusts):
#     disney_clusts = []
#     # find the disney movies
#     for i in range(len(clusts)):
#         movies = clusts[i]
#         for movie in movies:
#             for dm in disney_movies:
#                 if dm in movie.lower():
#                     print(f'Cluster {i} has {dm} in {movie}')
#                     disney_clusts.append(i)
#     # report the variance
#     loss = (len(np.unique(disney_clusts)) - 1)/len(clusts)
#     print(f'Loss: {loss}')
#     return loss

# vizualize the factors
# 1. scale
# 2. pca
# 3. add kmeans clustering
# 4. add the disney movies
# 4. visuaze as a scatter plot, with two panels, the clusetrs, and then then disney movies
def viz_factors(expPath):
    sc.settings.figdir = expPath
    adata = sc.read(os.path.join(expPath, "factors.h5ad"))
    # add clustering
    with open(os.path.join(expPath, "clusters_kmeans_labels.pkl"), "rb") as f:
        clusts = pickle.load(f)
    adata.obs["cluster"] = clusts
    movies = adata.obs["movie_names"].values
    is_disney = []
    for movie in movies:
        found_disney = False
        for dm in disney_movies:
            if dm in movie.lower():
                is_disney.append(True)
                found_disney = True
                break
        if not found_disney:
            is_disney.append(False)
    adata.obs["is_disney"] = is_disney
    # convert is_disney to category
    adata.obs["is_disney"] = adata.obs["is_disney"].astype("category")
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    # scale, pca and then plot pca using scanpy
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    # set scanpy figdir
    sc.pl.pca(adata, color=["cluster", "is_disney"], ncols=2, save="nmf_bin_pca.png")


def check_genre(expPath, clusts=None, silent=True):
    if clusts is None:
        with open(os.path.join(expPath, "clusters_kmeans.pkl"), "rb") as f:
            clusts = pickle.load(f)
    else:
        assert expPath is None, "expPath must be None if clusts is provided"
    loss_1, _ = check_horror_movies(clusts, silent=silent)
    loss_2, _ = check_disney_movies(clusts, silent=silent)
    loss_3, _ = check_scifi_movies(clusts, silent=silent)
    return loss_1, loss_2, loss_3


def check_single_cluster(expPath, clusts=None, silent=True):
    """
    Check that all movie categories are not lumped in a single cluster
    """
    if clusts is None:
        with open(os.path.join(expPath, "clusters_kmeans.pkl"), "rb") as f:
            clusts = pickle.load(f)
    else:
        assert expPath is None, "expPath must be None if clusts is provided"
    ref_movies_dicts = {'disney': disney_movies, 'horror': horror_movies, 'scifi': scifi_movies}
    all_clusts = {}
    for key in ref_movies_dicts:
        ref_movies = ref_movies_dicts[key]
        if key == 'horror':
            except_movies = horror_movies_excempt
        else:
            except_movies = None
        _clusts = get_clusts(clusts, ref_movies, except_movies=except_movies, silent=silent)
        all_clusts[key] = _clusts
    # Hack: pick 10 movies from each genre. Compare their clusters  
    n_movies = 10
    n_genre = len(ref_movies_dicts.keys())
    loss = 0
    for i in range(n_movies):
        the_clusts = []
        for key in all_clusts:
            the_clusts.append(all_clusts[key][i])
        # loss  would be zero if all three genres are in different clusters
        loss += n_genre - len(np.unique(the_clusts))
    loss = loss / (n_movies * (n_genre - 1))
    return loss
        


def cat_config(expPath):
    # read the config.yaml file and pretty prints some of its keys
    with open(os.path.join(expPath, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    keys = ["var_fam_scale", "init", "factor_model", "latent_dim", "max_steps"]
    for key in keys:
        if key in config:
            print(f"{key}: {config[key]}")



def ml_score_exp(expPath, outPath=None, silent=True, means=False):
    """
    Given an expPath, generates the factors, and clustering of them, and then computes factor clustering quality scores.

    ml_scores:
        the simple loss for 
    """
    if outPath is None:
        outPath = expPath
    else:
        os.makedirs(os.path.dirname(outPath), exist_ok=True)

    df_leiden, df_kmeans, clusts, labels_kmeans = summarize_factors(expPath, means=means)
    res = list(check_genre(expPath, silent=silent))
    res += [check_single_cluster(expPath, silent=silent)]
    tmp = {}
    for i, val in enumerate(['horror', 'disney', 'scifi', 'mixing_loss']):
        key_name = f'ml_score_{val}' if means is False else f'ml_score_mean_{val}'
        tmp[key_name] = res[i]

    # Log the distances in the config file and tensorboard        
    configHandler = ConfigHandler(expPath=expPath)
    configHandler.write_updated_config(**tmp)
    return tmp


if __name__ == '__main__':
    """
    Runs the scoring from command line.

    Args:
        See help.
        
    Example usage:
        
    """
    # TODO: add score batch
    task_name = sys.argv[1]
    if task_name == 'score_exp':
        parser = argparse.ArgumentParser(description='Score movielens based on facators.')
        parser.add_argument('--exp', '-e', type=str, help='The path to the experiment directory.')
        parser.add_argument('--out_path', '-o', type=str, help='The path to output directory.', default=None, required=False)
        parser.add_argument('--silent', '-s', type=str2bool, help='Silent or not.', default=True, required=False)
        args = parser.parse_args(sys.argv[2:])
        ml_score_exp(expPath=args.exp, outPath=args.out_path, silent=args.silent)
    else:
        raise ValueError(f"Unknown task: {task_name}")



# 1. find the most popular factors (sum per factor the movie scores)
# 2. within the most popular factors, find the most popular movies (in each factor, sort by the movie, and report the top 10)
# a. generate the mean factor matrix
# b. compute the sum of each factor
# c. emit the top 10 movies in each factor, based on popularity of the factors
def handle_manual_test(expPath, ntopMovies=20):
    create_factors(expPath, nSamples=1)
    factor = prepare_factors(expPath, means=True)
    # handle_factor(factor.X, ntopMovies=20, outDir=expPath)
    # should be n movies by n latents
    assert factor.shape[0] > factor.shape[1], f"factor.shape[0] > factor.shape[1]; {factor.shape}"
    # compute the sum of each factor, and put with the factor index in a dataframe
    factor_sums = np.sum(factor.X, axis=0)
    df = pd.DataFrame({'factor_index': np.arange(factor.shape[1]), 'factor_sum': factor_sums})
    # sort by factor_sum
    df.sort_values(by='factor_sum', inplace=True, ascending=False)
    # in each of the top10 factors, sort movies by their popularity, and report the top 10
    object_names = factor.obs['movie_names'].values.copy()
    # assert that book_names are the same as object_names
    # assert len(np.where(book_names != object_names)[0]) == 0, "book_names != object_names"
    res = dict()
    for i in range(df.shape[0]):
        ff = factor.X[:, df.iloc[i, 0]]
        # sort the movies by their popularity
        ff = np.argsort(ff)[::-1]
        # report the top n
        top_movies = object_names[ff[:ntopMovies]]
        # if top_movies is not a list, convert it to a list
        if not isinstance(top_movies, str):
            top_movies = top_movies.tolist()
        res[i] = top_movies
    # convert res to a data.frame
    res = pd.DataFrame(res).T
    # save this as a tsv data.frame
    res.to_csv(os.path.join(expPath, 'top_factors.tsv'), sep='\t', index=False)

    
def NGDC():
    # normalized discounted cumulative gain
    # see: https://github.com/AmazingDD/daisyRec/blob/dev/daisy/utils/metrics.py
    # what happens with a recommendation metric like NDCG    
    pass


def load_sorted_factor(expPath):
    """
    Returns sorted movies by latents, factors sorted by their popularity
    """
    factor = sc.read(os.path.join(expPath, "factors_mean.h5ad"))
    factor.shape
    # find the popularity of each factor (i.e,. column sum)
    factor.var["popularity"] = factor.X.sum(axis=0)
    # sort factor by popularity
    factor = factor[:, np.argsort(factor.var["popularity"].values)]
    return factor


def find_movie(movie, factor):
    for i, m in enumerate(factor.obs.movie_names.values):
        if movie.lower() in m.lower():
            print(i, m)
            return i, m
    return None, None


def get_movie_dict(movies, factor):
    the_moviies = {}
    for movie in movies:
        movie_idx, movie_name = find_movie(movie, factor)
        if movie_idx is not None:
            the_moviies[movie_name] = movie_idx
    return the_moviies


def plot_lolipop(factor, the_moviies, filePath, nCols=4, figsize=(15, 15), dpi=300):
    """
    Plot multi-panel lolipop plot for the given movies
    """
    nRows = int(np.ceil(len(the_moviies.keys()) / nCols))
    plt.clf()
    fig, axes = plt.subplots(nRows, nCols, figsize=figsize, sharex=False, sharey=True, )
    for i, (movie_name, movie_idx) in enumerate(the_moviies.items()):
        row_indx = int(np.floor(i / nCols))
        col_indx = i % nCols
        x = factor.X[movie_idx]
        # for each point, connect it to the xaxis using a vertical line
        for j in range(len(x)):
            _ = axes[row_indx, col_indx].plot([factor.var_names[j], factor.var_names[j]], [0, x[j]], color='grey', linestyle='-', linewidth=1)
        axes[row_indx, col_indx].scatter(factor.var_names, x)
        axes[row_indx, col_indx].set_title(movie_name)
        axes[row_indx, col_indx].set_xticklabels(factor.var_names, rotation=90)
        # add axis labels
        axes[row_indx, col_indx].set_xlabel('Factors (sorted by popularity)')
        axes[row_indx, col_indx].set_ylabel('E[LogNormal-VI]')
    # remove subplots that are not used
    for i in range(nCols*nRows):
        row_indx = int(np.floor(i / nCols))
        col_indx = i % nCols
        if i >= len(the_moviies.keys()):
            axes[row_indx, col_indx].axis('off')
    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    plt.savefig(filePath, dpi=dpi, bbox_inches='tight')
    plt.close()
    return nCols,nRows


def _order_by_cluster(factor, method='ward'):
    """
    Order movies in the factor by their cluster
    """
    Z = hierarchy.linkage(factor.X, method=method)
    return hierarchy.dendrogram(Z, no_plot=True)['leaves']


def plot_heatmap(factor, filePath, figsize=(15, 15), dpi=300, cmap='Greys'):
    """Plot factors as a heatmap"""
    order = _order_by_cluster(factor)
    plt.clf()
    plt.figure(figsize=figsize)
    sns.heatmap(factor.X[order, :], cmap=cmap, xticklabels=True, yticklabels=False)
    # xtick labels should be factor.var_names
    plt.xticks(np.arange(factor.shape[1]), factor.var_names)
    plt.savefig(filePath, dpi=dpi, bbox_inches='tight')
    plt.close()


def viz_factor(expPath):
    """
    Visualize the (mean) factor
    """
    factor = load_sorted_factor(expPath)
    # use hierarchical clustering on the movies to find the order of the movies
    filePath = os.path.join(expPath, 'factor_heatmap_ordered.png')
    plot_heatmap(factor, filePath)
    # plot the lolipop plot for a pre-defined set of movies
    the_moviies = get_movie_dict(TEST_MOVIES, factor)
    outPath = os.path.join(expPath, 'factor_lolipop.png')
    nCols, nRows = plot_lolipop(factor, the_moviies, outPath, nCols=4, figsize=(16, 10), dpi=300)



def compute_DCG(rels):
    """
    DCG_p = \sum_{i=1}^p \frac{2^{rel_i} - 1}{log(i+1)}
    """
    p = rels.shape[0]
    i = np.arange(p) + 1
    return np.sum((2**rels - 1) / np.log2(i + 1))

def compute_NDCG(rels, best_rels):
    """
    nDCG_p = \frac{DCG_p}{IDCG_p}
    where IDCG is ideal discounted cumulative gain:
        IDCG_p = \sum_{i=1}^{|REL_p|}  \frac{rel_i}{log(i+1)}
        REL_p: list of relevant documents in the corpus up to position p
    """
    dcg = compute_DCG(rels)
    idcg = compute_DCG(best_rels)
    return dcg / idcg

def handle_NDCG(holdout_idx, X_pred, X_obs):
    # elmentwise multiplication of the two matrices
    rel_mat = X_pred * X_obs.A
    ndcgs = np.zeros(rel_mat.shape[0])
    for i in range(rel_mat.shape[0]):
        # get the column indeces of the non-zero elements in this row from holdout_idx
        row_idx = np.where(holdout_idx[0] == i)
        col_idx = holdout_idx[1][row_idx[0]]
        # now get the values of the rel_mat
        rel_vals = rel_mat[i, col_idx]
        # sort them by the X_pred (so put all X_pred 1s over X_pred 0s)
        pred_vals = X_pred[i, col_idx]
        rel_vals = rel_vals[np.argsort(pred_vals)][::-1]
        #pred_vals[np.argsort(pred_vals)][::-1]
        best_rels = np.squeeze(np.sort(X_obs[i, col_idx].A))[::-1]
        # compute the NDCG
        ndcgs[i] = compute_NDCG(rel_vals, best_rels)
    avg_ndcg = np.nanmean(ndcgs)
    print(f"The average NDCG is: {avg_ndcg}")
    return ndcgs


def quick_bin(yy, cutoff=None):
    if cutoff is None:
        raise ValueError("cutoff must be provided")
    yy.counts = bb(yy.counts, cutoff=cutoff)
    yy.vad = bb(yy.vad, cutoff=cutoff)
    yy.train = bb(yy.train, cutoff=cutoff)
    return yy

def binarize_df(xx, cutoff=None):
    if cutoff is None:
        raise ValueError("cutoff must be provided")
    xx = quick_bin(xx, cutoff=cutoff)
    xx.heldout_data = quick_bin(xx.heldout_data, cutoff=cutoff)
    return xx


def fetch_holdout_data(expPath):
    # load the original data
    exp_handler = ExperimentHandler(expPath)
    dataset, labels = exp_handler.load_data()
    cutoff = getCutoffFromExpPath(expPath)
    # binarize the dataset
    dataset = binarize_df(dataset, cutoff)
    out_dataset = dataset.heldout_data
    X_obs = out_dataset.counts
    # find the indexes of the heldout data 
    holdout_mask = out_dataset.holdout_mask # (1 if it is heldout, 0 otherwise)
    # extract all 1 indexes using sparse matrix
    holdout_idx = holdout_mask.nonzero()
    return X_obs, holdout_idx


def handle_NDCG_exp(expPath, n_samples=10, bin_cutoff=0, use_mean=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"The device is ! {device}")
    model = ModelHandler.load_model(expPath, retrain=True).to(device)
    X_obs, holdout_idx = fetch_holdout_data(expPath)
    if use_mean:
        z = model.row_mean
        w = model.column_mean
        rate = (z.T @ w.T)
        rate = (rate > bin_cutoff).astype(int)
        ndcgs = handle_NDCG(holdout_idx, rate, X_obs)
        row_avg = ndcgs
        the_means = row_avg
    else:
        the_means = np.zeros((n_samples, X_obs.shape[0]))
        for i in range(n_samples):
            # X_gen = model.generate_data(1, None, None)[0].cpu().numpy()
            # X_gen = (X_gen > bin_cutoff).astype(int)
            #ndcgs = handle_NDCG(holdout_idx, X_gen, X_obs)
            z = model.row_distribution.sample(1).squeeze()
            w = model.column_distribution.sample(1).squeeze()
            the_rate = (z.T @ w.T).detach().cpu().numpy()
            the_rate = (the_rate > bin_cutoff).astype(int)
            ndcgs = handle_NDCG(holdout_idx, the_rate, X_obs)
            the_means[i, :] = ndcgs
            # compute the average per row, then a grand average
            row_avg = np.nanmean(the_means, axis=0)
    grand_avg = np.nanmean(row_avg)
    print(f"The grand average is: {grand_avg}")
    # Log the grand average in the config file and tensorboard        
    tmp = {'ndcg': float(grand_avg)}
    configHandler = ConfigHandler(expPath=expPath)
    configHandler.write_updated_config(**tmp)
    return the_means
