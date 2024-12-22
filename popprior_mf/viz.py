#!/usr/bin/env python
"""
    Set of visualizations methods for the VI in factor models.

    @Author: Sohrab Salehi sohrab.salehi@columbia.edu
"""


# TODO: locally generate data from the model

import argparse
import os
from pickle import TRUE
import sys
from pathlib import Path
import matplotlib
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap as umap
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import paired_distances

from model_factory import ModelHandler
from utils import BatchHandler, ExperimentHandler, sparse_tensor_from_sparse_matrix, str2bool, check_columns, runCMD
from utils import ConfigHandler


def plot_resutls(merge_table_path, save_dir):
    """
    Plot the results of the model
    Inputs:
        merge_table_path: path to the merge table (a csv with sample_id, model, yaml_path)
        save_dir: path to the save directory
    """
    merge_table = pd.read_csv(merge_table_path)
    # Assert that the merge_table has the required columns
    check_columns(merge_table, ["sample_id", "model", "yaml_path"])

    print(merge_table)

    # For path in yaml_path column, load the yaml file and add heldout_llhood to the table
    for i in range(len(merge_table)):
        try:
            yaml_path = os.path.join(merge_table.iloc[i]["yaml_path"], "config.yaml")
            with open(yaml_path, "r") as file:
                config = yaml.load(file, yaml.Loader)
                merge_table.at[i, "heldout_llhood"] = config["heldout_llhood"]
        except:
            print(f"Could not find yaml file {merge_table.iloc[i]['yaml_path']}")
            merge_table.at[i, "heldout_llhood"] = np.nan

    merge_table.to_csv(os.path.join(save_dir, "plot_data.csv"), index=False)
    # Create a plots directory if it doesn't exist
    plots_dir = os.path.join(save_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    def plot_heldout_llhood_single(merge_table, save_dir):
        """Gather all plots in one view"""
        n_samples = len(merge_table.sample_id.unique())
        n_models = len(merge_table.model.unique())
        figure, axis = plt.subplots(
            n_samples,
            n_models,
            squeeze=False,
            figsize=(10, 10),
            sharex=True,
            sharey=True,
        )
        i, j = 0, 0
        for sample_id in merge_table.sample_id.unique():
            sample_table = merge_table[merge_table.sample_id == sample_id]
            j = 0
            for model in sample_table.model.unique():
                model_table = sample_table[sample_table.model == model]
                model_table = model_table.sort_values(by="latent_dim")
                # Plot the hold_outlikelihood per latent dimension
                axis[i, j].plot(
                    model_table.latent_dim, model_table.heldout_llhood, "o-"
                )
                axis[i, j].set_title("{} {}".format(sample_id, model))
                axis[i, j].set_xlabel("Latent dimension")
                axis[i, j].set_ylabel("Heldout likelihood")
                j += 1
            i += 1

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "heldout_llhood_single.png"))
        plt.close()

    def plot_heldout_llhood(merge_table, save_dir):
        """Plot each facet in a file"""
        for sample_id in merge_table.sample_id.unique():
            sample_table = merge_table[merge_table.sample_id == sample_id]
            for model in sample_table.model.unique():
                model_table = sample_table[sample_table.model == model]
                model_table = model_table.sort_values(by="latent_dim")
                # Plot the hold_outlikelihood per latent dimension
                plt.figure(figsize=(10, 10))
                plt.plot(model_table.latent_dim, model_table.heldout_llhood, "o-")
                plt.title(f"{sample_id} {model}")
                plt.xlabel("Latent dimension")
                plt.ylabel("Held-out loglikelihood")
                plt.savefig(
                    os.path.join(save_dir, f"{sample_id}_{model}_heldout_llhood.png")
                )
                plt.close()

    plot_heldout_llhood_single(merge_table, plots_dir)
    plot_heldout_llhood(merge_table, plots_dir)


def load_factors(path, model):
    """
    Loads the factors from the given path.

    Args:
        path: the path to the factors.
        model: the model name.

    Returns:
        a numpy array, the transposed of the original array
    """
    try:
        if 'PMF' in model:
            z = np.loadtxt(os.path.join(path, 'qv_loc'))
        else:
            z = np.loadtxt(os.path.join(path, 'qw_loc'))
        
        if 'Amortized' in model:
            raise NotImplementedError('Loading factors in amortized models are not supported yet.')
    except Exception as e:
        print(e)
        return None
    return z

def load_embedding(path, model):
    """
    Loads the embedding from the given path.

    Args:
        path: the path to the embedding.
        model: the model name.
        
    Returns:
        a numpy array, the transposed of the original array
    """
    try:
        if 'PMF' in model:
            z = np.loadtxt(os.path.join(path, 'qu_loc')).T
        else:
            z = np.loadtxt(os.path.join(path, 'qz_loc')).T
        
        if 'Amortized' in model:
            z = z.T
        
    except Exception as e:
        print(e)
        return None
    return z


def plot_scatter(df, title='', pointSize=5.0, outPath=None):
    """
    Given a dataframe, plots a scatter plot.

    Args:
        df: a dataframe with columns 'x' and 'y' and 'label'.
        title: the title of the plot.
        pointSize: the size of the dots.
        outPath [optional]: the path to the output file. If None, the plot is not saved.

    Returns:
        the snv_plot object.
    """
    check_columns(df, ['x', 'y', 'label'])
    plt.clf()
    sns_plot = sns.scatterplot(data=df, x='x', y="y", hue='label', s=pointSize)
    #sns_plot.set(xticklabels=[], title=title)
    sns_plot.set(title=title)
    if outPath is not None:
        if not os.path.exists(os.path.dirname(outPath)):
            os.makedirs(os.path.dirname(outPath))
        sns_plot.get_figure().savefig(outPath)
    return sns_plot


def plot_joint_scatter(scalars):
    """
    Given a list of scalars, plots a joint scatter plot (!!Incomplete)
    """
    check_columns(scalars, ['step', 'value', 'tag', 'model', 'latent'])

    _ = sns.relplot(data=scalars, x="step", y="value", hue="tag", col="model", row='latent')


def transform2D(x, use_pca=False, n_components=None):
    """
    Runs PCA (optionally) then passes it through UMAP 

    Args:
        x: the embedding to transform, as a numpy array [n_samples, n_features]
        use_pca: whether to use PCA. If true, n_components is the number of components to use.
        n_components: the number of components to use. If None, will use the square root of the number of components.
    
    Returns:
        a numpy array [n_samples, 2]
    """
    if x.shape[1] == 2:
        print('Already 2D. Will pass.')
        return x
    if use_pca:
        if n_components is None:
            # Ensure 2 <= n_components <= x.shape[1]
            n_components = np.minimum(x.shape[1], np.maximum(2, int(np.sqrt(x.shape[1]))))
        pca = PCA(n_components=n_components)
        x = pca.fit_transform(x)

    return umap.UMAP().fit_transform(x)


def plot_embedding(path, model, title, labels, outPath, **kwargs):
    """
    Loads, transforms, and plots the embedding for the given path and model.

    Args:
        path: the path to the embedding.
        model: the model name (used to select the correct embedding).
        title: the title of the plot.
        labels: the labels of the samples (used to color the points) Should be the same size as the n_samples in the embedding.
        outPath: the path to the output file. If None, the plot is not saved.
        **kwargs: additional arguments to pass to the plot_scatter function.

    Returns:
        the snv_plot object.
    """
    z = load_embedding(path, model)
    if z is None:
        return

    embedding = transform2D(z, **kwargs)
    embedding = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'label': labels})
    plot_scatter(df=embedding, title=title, outPath=outPath)


def viz_pseudo_obs(ref_dat, outPath, expPath, model, latent, dataset, use_pca, is_global=False):
    """
    Plots the pseudo-observations for the given model and latent.

    Args:
        ref_dat: the dataset to combine with the pseudo-observations (Latent for EB or Data embedding for AmortizedEB)
        outPath: path to save the plot.
        expPath: path to the experiment folder to load the pseudo-observations from.
        model: the model name.
        latent: the latent dimesion value.
        dataset: the dataset object. 
        use_pca: whether to use PCA for the embedding.
    """
    print("Plotting pseudo_obs") 
    # TODO: also plot the local
    tag = 'local' if not is_global else 'global'
    pseudo_obs = np.loadtxt(os.path.join(expPath, f'pseudo_obs_loc_{tag}'))
    if len(pseudo_obs.shape) == 1: 
        pseudo_obs = pseudo_obs[None, :]
    new_dat = np.concatenate([ref_dat, pseudo_obs], axis=0)
    #_label = np.repeat(np.array(['data', 'pseudo_obs']), [dataset.counts.shape[0], pseudo_obs.shape[0]])
    _label = np.repeat(np.array(['data', 'pseudo_obs']), [ref_dat.shape[0], pseudo_obs.shape[0]])
    embedding = transform2D(new_dat, use_pca=use_pca)
    dt = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'label': _label})
    title = f"pseudo_obs ({tag}): {model} - {latent}"
    plot_scatter(df=dt, title=title, outPath=outPath)


def viz_batch(batchPath, use_pca=False):
    """
    Plots the summaryWriter, and various embeddings for each experiment in the batch.

    Args:
        batchPath: the path to the batch.
        use_pca: whether to use PCA. Default is False since PPCA is a type of PCA.

    Returns:
        None
    """
    batch_handler = BatchHandler(batchPath)
    scalars, paths = batch_handler.summarize_batch()
    scalars.model.unique()

    # Summary plots for elbo, lllhood, lpior, and entropy
    script_dir = Path( __file__ ).parent.absolute()
    cmdStr = f"Rscript {script_dir}/plots.R viz_elbo -i {batchPath}/summary.csv.gz -o {batchPath}/plots"
    runCMD(cmdStr)
    
    # Plot the original data
    print('Using PCA for data visualization')
    dataset, labels = batch_handler.load_data()
    embedding = transform2D(dataset.counts, use_pca=True)
    embedding = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'label': labels})
    plot_scatter(df=embedding, title=f"data", outPath=os.path.join(batchPath, 'plots', 'data.png'))

    # Plot the computed embeddings
    for indx in range(paths.shape[0]):
        print(f"Plotting embeddings {indx}/{paths.shape[0]}")
        path, model, latent = paths.iloc[indx][['paths', 'model', 'latent']]
        title = f"{model} - {latent}"
        outPath = os.path.join(batchPath, 'plots', 'embeddings', f"{model}_{latent}.png")
        if not os.path.exists(os.path.dirname(outPath)):
            os.makedirs(os.path.dirname(outPath))
        plot_embedding(path, model, title, labels, outPath)
        
    # Plot u_k for amortizd 
    amortized_paths = paths[paths.model == 'AmortizedPPCAEB']
    for indx in range(amortized_paths.shape[0]):
        print(f"Plotting pseudo_obs {indx}/{amortized_paths.shape[0]}")    
        path, model, latent = amortized_paths.iloc[indx][['paths', 'model', 'latent']]
        outPath = os.path.join(batchPath, 'plots', 'u_k', f'u_k_{model}_{latent}.png')
        viz_pseudo_obs(dataset.counts, outPath, path, model, latent, dataset, use_pca)

    # Plot z for prior for EB 
    eb_paths = paths[paths.model == 'PPCAEB']
    for indx in range(eb_paths.shape[0]):
        print(f"Plotting pseudo_obs {indx}/{eb_paths.shape[0]}")    
        path, model, latent = eb_paths.iloc[indx][['paths', 'model', 'latent']]
        outPath = os.path.join(batchPath, 'plots', 'u_k_eb', f'u_k_{model}_{latent}.png')
        viz_pseudo_obs(load_embedding(path, model), outPath, path, model, latent, dataset, use_pca)


def viz_batch_all(batchPath, use_pca=False):
    """
    Plots the summaryWriter, and various embeddings for each experiment in the batch.

    Args:
        batchPath: the path to the batch.
        use_pca: whether to use PCA. Default is False since PPCA is a type of PCA.

    Returns:
        None
    """
    batch_handler = BatchHandler(batchPath)
    scalars, paths = batch_handler.summarize_batch()
    scalars.model.unique()

    for indx in range(paths.shape[0]):
        print(f"Plotting embeddings {indx}/{paths.shape[0]}")
        path, _, _ = paths.iloc[indx][['paths', 'model', 'latent']]
        viz_exp(expPath=path, use_pca=use_pca, plot_data=(indx == 0))


def plot_sim_data(expPath, dataset, labels, use_data=True, num_data=1):
    """
    Projects and plots simulated data in the space of the original data.
    
    Args:
        expPath: the path to the experiment folder.
        dataset: the dataset object data matrix (e.g., counts)
        labels: the labels for the data (classes e.g., UU, UTTU)
    Returns:
        None

    Note
    """
    # Fix device to CPU
    device = torch.device('cpu')
    assert num_data > 0, "num_data must be greater than 0 but {num_data} given."
    model = ModelHandler.load_model(expPath).to(device)
    if dataset.heldout_data is not None:
        #compute_list = [[dataset.counts, 'counts', dataset.la
        # els], [dataset.heldout_data.counts, 'data_out', dataset.heldout_data.obs['labels'].values]]
        compute_list = [[dataset.counts, 'counts', dataset.labels], [dataset.heldout_data.counts, 'data_out', dataset.heldout_data.labels]]
        #compute_list = [[dataset.heldout_data.counts, 'data_out', dataset.heldout_data.labels]]
    else:
        compute_list = [[dataset.counts, 'counts', dataset.labels]]
    for data, tag, labels in compute_list:
        # data, tag, lables = compute_list[0]
        if use_data:
            print('Using data for embedding!')
            #x_input = torch.from_numpy(data).double()
            x_input = sparse_tensor_from_sparse_matrix(data).to(device)
            obs_indices = np.arange(data.shape[0])
        else:
            x_input = None
            obs_indices = None

        #if (tag == 'data_out') and ('Amortized' not in model.__class__.__name__):
        try:
            if tag == 'data_out':
                model = ModelHandler.load_model(expPath, retrain=True).to(device)
        except FileNotFoundError:
            print('No retrained model found to visualize. ')
            continue
        # print device for x_input and obs_incices
        X_gen = model.generate_data(1, x_input, obs_indices)[0]
        n_gen = num_data - 1
        for i in tqdm(range(n_gen)):
            with torch.no_grad():
                X_gen += model.generate_data(1, x_input, obs_indices)[0]

        X_gen = X_gen / (n_gen+1)
        
        # Compute the distance between the data and the generated data
        dists = paired_distances(data, X_gen, metric='euclidean')
        avgFullDist = np.mean(dists)

        _label = np.repeat(np.array(['data', 'predicted']), [data.shape[0], X_gen.shape[0]])
        _tp = np.tile(labels, 2)
        
        # Plot just the simulated data
        print('Fitting PCA to simulated data...')
        pca = PCA(n_components=model.latent_dim)
        x_fit = pca.fit(X_gen).transform(X_gen)
        embedding = transform2D(x_fit, use_pca=False)
        embedding = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'label': labels})
        plot_scatter(df=embedding, title=f"Simulated data ({tag})", outPath=os.path.join(expPath, 'plots', f'sim_data_{tag}.png'))

        # Plot overlayed
        pca = PCA(n_components=model.latent_dim)
        pca.fit(data.A)
        o1 = pca.transform(data.A)
        o2 = pca.transform(X_gen)
        x_comb = np.row_stack((o1, o2))
        # Compute the distance between the data and the generated data in the PCA space
        dists = paired_distances(o1, o2, metric='euclidean')
        avgPCADist = np.mean(dists)
        embedding = transform2D(x_comb, use_pca=False)
        embedding = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'label': _label, 'tp': _tp})
        plt.clf()
        sns_plot = sns.scatterplot(data=embedding, x='x', y="y", hue='tp', style='label', s=5, markers=['o', 's'])  
        sns_plot.get_figure().suptitle(f"data, predicted_data ({tag})")
        sns_plot.get_figure().savefig(os.path.join(expPath, 'plots', f'data_predicted_data_{tag}.png'))

        # Plot without the labels
        plt.clf()
        sns_plot = sns.scatterplot(data=embedding, x='x', y="y", hue='label', s=5)  
        sns_plot.get_figure().suptitle(f"data, predicted_data, without labels")
        sns_plot.set(title=f"Avg. full dist: {avgFullDist:.2f}, Avg. PCA dist: {avgPCADist:.2f}")
        sns_plot.get_figure().savefig(os.path.join(expPath, 'plots', f'data_predicted_data_no_label_{tag}.png'))

        # Log the distances in the config file and tensorboard        
        configHandler = ConfigHandler(expPath=expPath)
        configHandler.write_updated_config(**{f'avgFullDist_{tag}': float(avgFullDist), f'avgPCADist_{tag}': float(avgPCADist)})
        
        summary_writer = SummaryWriter(expPath)
        summary_writer.add_scalar(f'fit/avg_full_dist_{tag}', avgFullDist)
        summary_writer.add_scalar(f'fit/avg_pca_dist_{tag}', avgPCADist)


def viz_exp(expPath, outPath=None, use_pca=False, plot_data=True, num_data=1, viz_elbo=True, lightweight=False):
    """
    Plots the summaryWriter, and various embeddings for the experiment.

    Args:
        expPath: the path to the experiment.
        use_pca: whether to use PCA. Default is False since PPCA is a type of PCA.
        lightweight: whether to only plot the simulated data on top of the real data.

    Returns:
        None
    """
    if outPath is None:
        outPath = expPath
    else:
        if not os.path.exists(os.path.dirname(outPath)):
            os.makedirs(os.path.dirname(outPath))

    exp_handler = ExperimentHandler(expPath)
    dataset, labels = exp_handler.load_data()
    # scalars = exp_handler.summarize_exp()
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    sns.set_style("ticks")    

    # Plot simulated data side by side the original data
    plot_sim_data(expPath, dataset, labels, num_data=num_data)

    if lightweight:
        return(0)


    # Summary plots for elbo, lllhood, lpior, and entropy
    if viz_elbo:
        print('Plotting Elbo and other measures')
        script_dir = Path( __file__ ).parent.absolute()
        cmdStr = f"Rscript {script_dir}/plots.R viz_single -i {expPath}/summary.csv.gz -o {expPath}/plots"
        runCMD(cmdStr)


    # Plot the original data
    if plot_data:
        print('Using PCA for data visualization')
        embedding = transform2D(dataset.counts.A, use_pca=True)
        embedding = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'label': labels})
        plot_scatter(df=embedding, title=f"data", outPath=os.path.join(expPath, 'plots', 'data.png'))

    # Plot the computed embeddings
    print("Plotting embeddings")
    model, latent = exp_handler.config['factor_model'], exp_handler.config['latent_dim']
    title = f"{model} - {latent}"
    outPath = os.path.join(expPath, 'plots', 'embeddings', f"{model}_{latent}.png")
    if not os.path.exists(os.path.dirname(outPath)):
        os.makedirs(os.path.dirname(outPath))
    plot_embedding(expPath, model, title, labels, outPath)

    if 'Amortized' in exp_handler.config['factor_model'] and 'EB' in exp_handler.config['factor_model']:
        # Plot u_k for amortizd 
        outPath = os.path.join(expPath, 'plots', 'u_k', f'u_k_{model}_{latent}.png')
        viz_pseudo_obs(dataset.counts, outPath, expPath, model, latent, dataset, use_pca)
    elif 'PPCAEB' in exp_handler.config['factor_model'] or 'PMFEB' in exp_handler.config['factor_model']:
        # Plot z for prior for EB 
        outPath = os.path.join(expPath, 'plots', 'u_k_eb', f'u_k_{model}_{latent}.png')        
        viz_pseudo_obs(load_embedding(expPath, model), outPath, expPath, model, latent, dataset, use_pca)

    # for the twin models, plot pseudo obs for the global variables too
    if 'Twin' in exp_handler.config['factor_model']:
        # Plot u_k for amortizd 
        # factors are (L by D)
        outPath = os.path.join(expPath, 'plots', 'psi_h_eb', f'psi_h_{model}.png')
        viz_pseudo_obs(ref_dat=load_factors(expPath, model), outPath=outPath, expPath=expPath, model=model, latent=latent, dataset=dataset, use_pca=use_pca, is_global=True)

    # TODO: viz the pseoudo_obs initialization vs the final peuso obs values
    return(0)


    


# dedicated function to viz the optimization path
def get_summary_writer(expPath, key=None):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(expPath)
    event_acc.Reload()
    if key is None:
        print(event_acc.Tags())
        raise ValueError('Please provide a key from above')    
    return pd.DataFrame(event_acc.Scalars(key))


def get_summary_writer_multi_keys(expPath, keys=None):
    """Load multiple keys from the summary writer"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(expPath)
    event_acc.Reload()
    if keys is None:
        print(event_acc.Tags())
        raise ValueError('Please provide a key from above')    
    df = pd.DataFrame()
    for key in keys:
        tmp = pd.DataFrame(event_acc.Scalars(key))
        tmp['key'] = key
        df = pd.concat([df, tmp], axis=0)
    return df



def viz_optimization_path(expPath, outPath=None, save=False, figsize=(10, 5)):
    if outPath is None:
        outPath = os.path.join(expPath, 'plots', 'elbo.png')
    
    df = get_summary_writer(expPath, key='elbo/elbo')
    if save:
        outDir = os.path.join(expPath, 'tables')
        os.makedirs(outDir, exist_ok=True)
        df['expPath'] = os.path.basename(expPath)
        df.to_csv(os.path.join(outDir, 'elbo.csv'), index=False, compression='gzip')
    
    # ensure the dir exists
    os.makedirs(os.path.dirname(outPath), exist_ok=True)

    plt.clf()
    # plot side by side, the elbo and then the last 50 percent of the iterations
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # plot the full elbo
    sns.lineplot(x='step', y='value', data=df, ax=axes[0])
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('ELBO')
    axes[0].set_title('Batch ELBO over training steps (cadence = 10)')
    # plot the last 50 percent of the iterations
    # compute and report the variance of the ELBO for the last 50 percent of the iterations
    half_df = df.iloc[int(df.shape[0]/2):, :]
    elbo_var = np.var(half_df.value)
    sns.lineplot(x='step', y='value', data=half_df, ax=axes[1])
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('ELBO')
    axes[1].set_title('Last 50% of Steps - Var[ELBO]: {:.2f}'.format(elbo_var))
    plt.tight_layout()
    fig.savefig(outPath, dpi=300, bbox_inches='tight')
    plt.close()


def _inner_grad_trace_plot(outPath, figsize, the_names, df, color_dict, key='grad_'):
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Dictionary to hold the line objects
    line_dict = {}
    for name in the_names:
        df_sub = df[df.key == f'grad_var/{key}{name}']
        sns.lineplot(x='step', y='value', data=df_sub, ax=axes[0], marker='o', markersize=5, linewidth=0.5, linestyle='dashed', color=color_dict[name])
        # Retrieve the last line object and add it to the dictionary
        line_dict[name] = axes[0].lines[-1]
        # Plot the last 50%
        half_df = df_sub.iloc[int(df_sub.shape[0]/2):, :]
        sns.lineplot(x='step', y='value', data=half_df, ax=axes[1], marker='o', markersize=5, linewidth=0.5, linestyle='dashed', color=color_dict[name])
    # Setting labels and titles
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Gradient Trace')
    axes[0].set_title(f'Gradient Trace (cadence = 10)')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Gradient Trace')
    axes[1].set_title('Last 50% of Steps')
    # Creating the legend from the line objects
    fig.legend(handles=line_dict.values(), labels=line_dict.keys(), loc='upper right', bbox_to_anchor=(1.1, 1))
    # Saving the figure
    fig.savefig(outPath, dpi=300, bbox_inches='tight')
    plt.close()

def viz_grad_trace(expPath, outPath=None, save=False, figsize=(10, 5), ignore_total=True):
    """Viz all keys on the same plot"""
    the_names = ['trace', 'row_location', 'row_scale', 'col_location', 'col_scale']
    if ignore_total:
        the_names = the_names[1:]
    # Add names for the E[||g||] and Var[||g||], i.e., grad_var/grad_norm_mean_{} and grad_var/grad_norm_var_{}
    the_keys = [f'grad_var/grad_{name}' for name in the_names]
    the_keys = the_keys + [f'grad_var/grad_norm_mean_{name}' for name in the_names]
    the_keys = the_keys + [f'grad_var/grad_norm_var_{name}' for name in the_names]
    df = get_summary_writer_multi_keys(expPath, keys=the_keys)
    
    if save:
        outDir = os.path.join(expPath, 'tables')
        os.makedirs(outDir, exist_ok=True)
        df['expPath'] = os.path.basename(expPath)
        df.to_csv(os.path.join(outDir, 'grads.csv.gz'), index=False, compression='gzip')
    outDir = os.path.join(expPath, 'plots')
    os.makedirs(outDir, exist_ok=True)
    

    color_vals = sns.color_palette("husl", len(the_names))
    colors = [matplotlib.colors.rgb2hex(color) for color in color_vals]
    color_dict = dict(zip(the_names, colors))

    # Your existing color setup
    color_vals = sns.color_palette("husl", len(the_names))
    colors = [matplotlib.colors.rgb2hex(color) for color in color_vals]
    color_dict = dict(zip(the_names, colors))

    for k in ['grad_', 'grad_norm_mean_', 'grad_norm_var_']:
        outPath = os.path.join(outDir, f'{k}all_grads.png')
        _inner_grad_trace_plot(outPath, figsize, the_names, df, color_dict, key=k)
    


def viz_params(expPath, outPath=None, save=False, figsize=(10, 5)):
    """
    For each named params of the variational family, (rows, and columns), plot a histogram
    """
    if outPath is None:
        outPath = os.path.join(expPath, 'plots', 'params')
    os.makedirs(outPath, exist_ok=True)
    # set the device to cuda if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the model
    model = ModelHandler.load_model(expPath, retrain=True).to(device)
    # in row and column distributions, find the named params, then plot the histogram
    
    def plot_hist(param, param_name, outPath, tag=''):
        pp = param.detach().cpu().numpy()
        plt.clf()
        plt.hist(pp, bins=100)
        # set title
        plt.title(f"{tag} {param_name}")
        # save the plot
        plt.savefig(os.path.join(outPath, f'{param_name}_{tag}.png'))
        plt.close()

    for p in model.row_distribution.named_parameters():
        # detach, then create histogram
        print(p[0])
        plot_hist(p[1], p[0], outPath, tag='row')
    
    for p in model.column_distribution.named_parameters():
        # detach, then create histogram
        print(p[0])
        plot_hist(p[1], p[0], outPath, tag='col')


    





if __name__ == '__main__':
    """
    Runs the plotting function for a full batch.

    Args:
        See help.
        
    Example usage:
        python3 viz.py viz_exp -e /Users/De-identified Authors/projects/rnaseq-pfm/results/rna_small_HBCKI_202208-08-134841.263597AmortizedPMF
        python3 -mpdb -c 'c' viz.py viz_batch -b ../results/deliverables/compare_parallel/2022-Jul-21-08-20-02-elc9g6hN4F/lx33_uu_uttu_var_sub    
        python3 -mpdb -c 'c' viz.py viz_exp -e ../results/rna_small_08CKU_202208-02-151126.745800AmortizedPPCAEB_full/
    """

    task_name = sys.argv[1]
    if task_name == 'viz_batch':
        parser = argparse.ArgumentParser(description='Visualize the results of the experiments organized in a batch.')
        parser.add_argument('--batch', '-b', type=str, help='The path to the batch directory.')
        parser.add_argument("--pca", '-p', type=str2bool, nargs='?', const=True, default=False,  help="Use PCA before using UMAP.")
        args = parser.parse_args(sys.argv[2:])
        viz_batch(batchPath=args.batch, use_pca=args.pca)
    elif task_name == 'viz_exp':
        parser = argparse.ArgumentParser(description='Visualize model results from a single experiment.')
        parser.add_argument('--exp', '-e', type=str, help='The path to the experiment directory.')
        parser.add_argument('--out_path', '-o', type=str, help='The path to output directory.', default=None, required=False)
        parser.add_argument("--pca", '-p', type=str2bool, nargs='?', const=True, default=False,  help="Use PCA before using UMAP.")
        parser.add_argument("--num_data", '-n', type=int, nargs='?', default=1,  help="Number of data points to simulate for predicted_data plotting.")
        parser.add_argument("--viz_elbo", '-v', type=str2bool, nargs='?', default=True,  help="Whether to plot the elbo and other measures.")
        parser.add_argument("--viz_data", '-d', type=str2bool, nargs='?', default=True,  help="Whether to plot the data.")
        parser.add_argument("--light", '-l', type=str2bool, nargs='?', const=True, default=False,  help="Lightweight mode - plot just the simulated data and data.")
        args = parser.parse_args(sys.argv[2:])
        viz_exp(expPath=args.exp, outPath=args.out_path, use_pca=args.pca, num_data=args.num_data, viz_elbo=args.viz_elbo, plot_data=args.viz_data)
    elif task_name == 'viz_elbo':
        # parse for viz_optimization_path
        parser = argparse.ArgumentParser(description='Visualize the optimization path.')
        parser.add_argument('--exp', '-e', type=str, help='The path to the experiment directory.')
        parser.add_argument('--out_path', '-o', type=str, help='The path to output directory.', default=None, required=False)
        parser.add_argument("--save", '-s', type=str2bool, nargs='?', const=True, default=False,  help="Whether to save the elbo table.")
        args = parser.parse_args(sys.argv[2:])
        viz_optimization_path(expPath=args.exp, outPath=args.out_path, save=args.save)
    elif task_name == 'viz_grad_trace':
        # parse for viz_grad_trace
        parser = argparse.ArgumentParser(description='Visualize the gradient trace path.')
        parser.add_argument('--exp', '-e', type=str, help='The path to the experiment directory.')
        parser.add_argument('--out_path', '-o', type=str, help='The path to output directory.', default=None, required=False)
        parser.add_argument("--save", '-s', type=str2bool, nargs='?', const=True, default=True,  help="Whether to save the elbo table.")
        args = parser.parse_args(sys.argv[2:])
        viz_grad_trace(expPath=args.exp, outPath=args.out_path, save=args.save)
    elif task_name == 'viz_batch_all':
        parser = argparse.ArgumentParser(description='Visualize the results of the experiments organized in a batch.')
        parser.add_argument('--batch', '-b', type=str, help='The path to the batch directory.')
        parser.add_argument("--pca", '-p', type=str2bool, nargs='?', const=True, default=False,  help="Use PCA before using UMAP.")
        args = parser.parse_args(sys.argv[2:])
        viz_batch_all(batchPath=args.batch, use_pca=args.pca)
    elif task_name == 'plot_results':
        parser = argparse.ArgumentParser(description='Plot the results of the model.')
        parser.add_argument('--merge_table', '-m', type=str, help='The path to the merge table.')
        parser.add_argument('--save_dir', '-s', type=str, help='The path to the save directory.')
        args = parser.parse_args(sys.argv[2:])
        plot_resutls(merge_table_path=args.merge_table, save_dir=args.save_dir)
    elif task_name == 'viz_params':
        parser = argparse.ArgumentParser(description='Viz the params of the variational family')
        parser.add_argument('--exp', '-e', type=str, help='The path to the experiment directory.')
        parser.add_argument('--out_path', '-o', type=str, help='The path to output directory.', default=None, required=False)
        parser.add_argument("--save", '-s', type=str2bool, nargs='?', const=True, default=False,  help="Whether to save the elbo table.")
        args = parser.parse_args(sys.argv[2:])
        viz_params(expPath=args.exp, outPath=args.out_path, save=args.save)
    else:
        raise ValueError(f"Unknown task: {task_name}")




    



