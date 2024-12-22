#!/usr/bin/env python

"""
This script will setup the model, run and evaluate it and save the results.
Intended to be used within a nextflow workflow.

Workflow
    For each dataset:
        1. Create a masked dataset with the given heldout proportion
    For each parameter:
        2. Configure and run the model and compute heldout-loglikelihood
        3. Gather per data, per parameter loglikelihood into one csv
        4. Plot a per data, per param plot

Inputs:
    Path to the h5ad file
    Output directory
    is_testing

Outputs:
    The heldout logliklihood

Example:        
    argv = ['../data/lx33_uu_500.h5ad', '.', '2', '10', 'False', '100', '0.2']
    main(argv)

NB: To add another input argument:
    - See main_run_model_setup_args() in utils.py
    - Adjust the nextflow workflow

@Author: Sohrab Salehi
"""


import copy
import datetime
from scipy import sparse


# TODO: move all prints to logger
from tqdm import tqdm
import torch
import numpy as np
import pickle
import os
import sys
import shutil
import time
import yaml
import inspect

from torch.utils.tensorboard import SummaryWriter
from utils import (
    ConfigHandler,
    ExperimentHandler,
    main_run_model_setup_args,
    main_data_setup_args,
    main_plot_setup_args,
    FileNameUtils,
    sparse_tensor_from_sparse_matrix,
    get_missing_index,
    save_dataset,
)

from data_utils import apply_preprocessing, setup_data_loader

from setup_data import create_heldout_data, pickPicklePath
from model_factory import (
    ModelHandler,
    TRAINED_MODEL_TAG,
    RETRAINED_MODEL_TAG,
    RETRAINED_MODEL_DIR,
)

from training_utils import (
    CheckPointManager,
    GlobalTrainer,
    init_params,
)


def none_or_float(value):
    """Helper function for parsing floats"""
    if value == "None":
        return None
    else:
        return float(value)


def none_or_int(value):
    """Helper function for parsing ints"""
    if value == "None":
        return None
    else:
        return int(value)


def train_model(
    model,
    dataset,
    batch_size,
    retrain,
    use_batch_sampler,
    device,
    log_cadence,
    summary_writer,
    config,
    init=None,
    init_exp_dir=None,
    **kwargs,
):
    """
    Train the model using specified optimizer and a scheduler.
    """
    # Status variables
    epoch = 0
    train_loss = 0.0

    init_params(model, dataset, row_only=retrain, init=init) if (
        init != "xavier" and init is not None
    ) else None
    data_loader = setup_data_loader(
        dataset=dataset,
        use_batch_sampler=use_batch_sampler,
        batch_size=batch_size,
        dataset_kwargs={},
    )

    train_tag = RETRAINED_MODEL_TAG if retrain else TRAINED_MODEL_TAG
    start_time = time.time()
    epoch_len = len(data_loader)

    if init_exp_dir is not None and not retrain:
        print("Initializing model from the given checkpoint...")
        checkPointPath = os.path.join(init_exp_dir, f"model_checkpoint_{train_tag}.pt")
        CheckPointManager.load_checkpoint(model, checkPointPath)

    kwargs["min_loss"] = 1e10
    kwargs["model"] = model
    kwargs["train_tag"] = train_tag
    kwargs["retrain"] = retrain
    kwargs["epoch_len"] = epoch_len

    globalTrainer = GlobalTrainer(**kwargs)

    print(f"Number of epochs: {globalTrainer.n_epoch}")
    print(f"Epoch length: {epoch_len}")

    USE_VALIDATION = False
    if USE_VALIDATION:
        # Prepare data for validation loss
        with torch.no_grad():
            vad_torch, missing_indexes = _prepare_data_for_heldout_loglikelihood(
                dataset.vad, dataset.holdout_mask, device=device
            )

    pbar = tqdm(range(globalTrainer.n_epoch))
    for epoch in pbar:
        if epoch > 1:
            train_str = globalTrainer.switchTrainer.get_train_str()

            pbar.set_description(
                f"Train: {train_str} ELBO: {-train_loss:.2f} TOL: {globalTrainer.earlyStopping.tol_indx} ({duration:.2f} sec) LR: {lr_str} CPE: {globalTrainer.checkPointManager.check_point_epoch}"
            )
            globalTrainer.switchTrainer(model)

        train_loss = 0.0
        for step, (datapoints_indices, x_train, holdout_mask) in enumerate(data_loader):
            datapoints_indices = datapoints_indices.to(device)
            holdout_mask = holdout_mask.to(device)
            x_train = x_train.to(device)

            loss = globalTrainer.trainer(
                step, epoch, summary_writer, datapoints_indices, x_train, holdout_mask
            )

            train_loss += loss.item()

            globalTrainer.customScheduler.warm_up_step(epoch=epoch, step=step)

            if step == 0 or step % log_cadence == log_cadence - 1:
                duration = (time.time() - start_time) / (step + 1)
                lr_str = ""
                for param_group in globalTrainer.optimizers[0].param_groups:
                    # lr_str += f"{param_group['lr']:.4f} "
                    lr_str += f"{param_group['lr']:.2e} "

                summary_writer.add_scalar("loss/loss", loss, step + epoch * epoch_len)

                # Handle batch specific schedulers
                globalTrainer.customScheduler.batch_step()

        train_loss = train_loss / len(data_loader)
        summary_writer.add_scalar("loss/train_loss", loss, epoch)

        globalTrainer.switchTrainer.update_status(epoch)

        # Compute an actual valid loss, based on the mean sample
        if USE_VALIDATION:
            with torch.no_grad():
                valid_loss = -model.compute_heldout_loglikelihood(
                    vad_torch,
                    missing_indexes,
                    n_monte_carlo=1,
                    subsample_zeros=True,
                    write_llhood=False,
                ).item()
        else:
            # Just use the training loss as the validation loss
            valid_loss = train_loss

        globalTrainer.checkPointManager(epoch, valid_loss, train_loss)

        if globalTrainer.earlyStopping(valid_loss, globalTrainer.optimizers):
            break

        globalTrainer.customScheduler(epoch=epoch, valid_loss=valid_loss)

    if epoch == epoch_len - 1:
        print(f"Maximum epoch: {epoch} reached. Stop training.")

    # Save the final model
    torch.save(
        model.state_dict(),
        os.path.join(globalTrainer.param_save_dir, f"model_final_{train_tag}.pt"),
    )

    # Restore the best checkpoint if it exists
    model, train_loss = globalTrainer.checkPointManager.restore_check_point(
        config, model, train_loss
    )

    torch.save(
        model.state_dict(),
        os.path.join(globalTrainer.param_save_dir, f"model_trained_{train_tag}.pt"),
    )

    return model, train_loss, globalTrainer.earlyStopping.optimization_diverged


def setup_output_dir(save_dir, factor_model):
    """ """
    if save_dir == ".":
        param_save_dir = "out"  # to handle nextflow
    else:
        param_save_dir = os.path.join(
            save_dir,
            FileNameUtils.get_file_name(generic_name="EXP", suffix=factor_model),
        )

        if os.path.exists(param_save_dir):
            print("Deleting old log directory at {}".format(param_save_dir))
            shutil.rmtree(param_save_dir)
        if not os.path.exists(param_save_dir):
            os.makedirs(param_save_dir)
    return param_save_dir


def _get_args_for_setup_row_vars(model, kwargs):
    legal_args = inspect.signature(model._setup_row_vars).parameters.keys()
    # add items for ppca
    kwargs["scale"] = kwargs["row_prior_scale"]
    kwargs["init_loc"] = None
    # add items for pmf
    kwargs["concentration"] = kwargs["row_prior_concentration"]
    kwargs["rate"] = kwargs["row_prior_rate"]
    # add init_scale and fixed scale
    kwargs["init_scale"] = kwargs["var_fam_init_scale"]
    kwargs["fixed_scale"] = kwargs["var_fam_scale"]
    # add prior family and pseudo_var_family
    kwargs["family"] = kwargs["var_family"]
    # kwargs['pseudo_var_family'] = kwargs['pseudo_var_family']
    tmp = dict(filter(lambda x: x[0] in legal_args, kwargs.items()))
    return tmp


def _prepare_data_for_heldout_loglikelihood(vad, holdout_mask, device, use_cache=False):
    """ """
    enforce_cpu = False
    if enforce_cpu:
        print("Putting on CPU!!")
        vad_torch = sparse_tensor_from_sparse_matrix(vad).to(torch.device("cpu"))
    else:
        vad_torch = sparse_tensor_from_sparse_matrix(vad).to(device)
    torch.cuda.empty_cache()
    if use_cache:
        # Save missing indexes to disk
        cache_name = "missing_indexes.pkl"
        # cache_name = None
        if cache_name is not None:
            if os.path.exists(cache_name):
                print("WARNING: Loading missing indexes from disk")
                missing_indexes = pickle.load(open(cache_name, "rb"))
        else:
            missing_indexes = get_missing_index(
                X=vad, holdout_mask=holdout_mask, prime_number=vad.shape[1]
            ).T  # [2, n_missing]
            # save to disk
            if cache_name is not None:
                pickle.dump(missing_indexes, open(cache_name, "wb"))
    else:
        missing_indexes = get_missing_index(
            X=vad, holdout_mask=holdout_mask, prime_number=vad.shape[1]
        ).T  # [2, n_missing]

    # If missing index is empty, just return 0, 0
    if missing_indexes.shape == (0,) or missing_indexes.shape == (0, 0):
        missing_indexes = np.zeros(shape=(2, 1))

    # Assert that the missing indexes are within the bounds of vad (no overflow)
    assert (
        missing_indexes[0, :] < vad.shape[0]
    ).all(), "missing_indexes[0, :] < vad.shape[0]"
    assert (
        missing_indexes[1, :] < vad.shape[1]
    ).all(), "missing_indexes[1, :] < vad.shape[1]"
    return vad_torch, missing_indexes


# TODO: refactor and breakdown into smaller functions
def run_model(**kwargs):
    """
    Configure and train the model and compute held-out loglikelihood
    """

    # check if kwargs dictionary has a key called 'picklePath'
    if "seed" not in kwargs:
        kwargs["seed"] = 0

    # Keep track of total time
    start_time = time.time()

    print(f'Setting seed to {kwargs["seed"]}')
    torch.manual_seed(kwargs["seed"])

    # Load the pickle dataset
    picklePath = pickPicklePath(kwargs["picklePath"], kwargs["factor_model"])
    print(f"Loading dataset from {picklePath}")
    with open(picklePath, "rb") as f:
        dataset = pickle.load(f)

    dataset = apply_preprocessing(
        dataset, kwargs["factor_model"], kwargs["binarize_data"]
    )

    # Set up the output directory
    param_save_dir = setup_output_dir(kwargs["save_dir"], kwargs["factor_model"])
    print(f"Saving to {param_save_dir}")

    # breakpoint()
    # save_dataset(dataset, tag='training', param_save_dir=param_save_dir)

    num_datapoints, data_dim = dataset.counts.shape
    summary_writer = SummaryWriter(param_save_dir)
    # set the device from the kwargs
    if kwargs["device"] == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Print a summary of the dataset (the number of rows and columns)
    print(f"Number of datapoints: {num_datapoints}")
    print(f"Data dimension: {data_dim}")

    config = kwargs.copy()
    config["device"] = f"{device}"
    config["num_datapoints"] = num_datapoints
    config["data_dim"] = data_dim
    config["summary_writer"] = summary_writer
    config["param_save_dir"] = param_save_dir
    config["picklePath"] = picklePath
    config["start_time"] = start_time

    model = ModelHandler.model_factory(config)
    # In pytorch2...
    # model = torch.compile(model)

    # Train the model and save the results
    # Do not save the summary writer
    tmp_config = config.copy()
    tmp_config.pop("summary_writer", None)
    with open(os.path.join(param_save_dir, "config.yaml"), "w") as file:
        yaml.dump(tmp_config, file)

    model, elbo, optim_diverged = train_model(
        model=model,
        dataset=dataset,
        batch_size=kwargs["batch_size"],
        max_steps=kwargs["max_steps"],
        row_learning_rate=kwargs["row_learning_rate"],
        column_learning_rate=kwargs["column_learning_rate"],
        mixture_learning_rate=kwargs["mixture_learning_rate"],
        use_warmup=kwargs["use_warmup"],
        optimizer=kwargs["optimizer"],
        scheduler=kwargs["scheduler"],
        scheduler_patience=kwargs["scheduler_patience"],
        device=device,
        log_cadence=kwargs["log_cadence"],
        tolerance=kwargs["tolerance"],
        summary_writer=summary_writer,
        param_save_dir=param_save_dir,
        retrain=False,
        config=config,
        init=kwargs["init"],
        save_checkpoint=kwargs["save_checkpoint"],
        restore_best_model=kwargs["restore_best_model"],
        train_mode_switch=kwargs["train_mode_switch"],
        use_custom_scheduler=kwargs["use_custom_scheduler"],
        use_batch_sampler=kwargs["use_batch_sampler"],
        schedule_free_epochs=kwargs["schedule_free_epochs"],
        elbo_mode=kwargs["elbo_mode"],
        track_grad_var=kwargs["track_grad_var"],
        n_elbo_particles=kwargs["n_elbo_particles"],
        clipGradients=kwargs["clipGradients"],
        max_gradient_norm=kwargs["max_gradient_norm"],
        stopping_loss_threshold=kwargs["stopping_loss_threshold"],
        init_exp_dir=kwargs["init_exp_dir"],
    )

    # add timestamp
    config["train_delta_time"] = time.time() - start_time
    print("Train delta_time: ", config["train_delta_time"])

    # add last train elbo
    config["train_elbo"] = elbo
    config["optim_diverged"] = optim_diverged

    if "summary_writer" in config.keys():
        config.pop("summary_writer", None)

    # Compute the log likelihood on heldout data (heldout elements) using masked enteries
    print("Computing heldin log likelihood")
    if optim_diverged:
        # don't waste time training the lest
        kwargs["n_llhood_samples"] = 1
        kwargs["max_steps"] = 2

    # print(torch.cuda.memory_summary())
    with torch.no_grad():
        vad_torch, missing_indexes = _prepare_data_for_heldout_loglikelihood(
            dataset.vad, dataset.holdout_mask, device=device
        )

        heldout_llhood = model.compute_heldout_loglikelihood(
            vad_torch,
            missing_indexes,
            n_monte_carlo=kwargs["n_llhood_samples"],
            subsample_zeros=kwargs["subsample_zeros"],
        )

        # release the memory of vad_torch adn missing_indexes
        del vad_torch
        del missing_indexes
        torch.cuda.empty_cache()
    print(f"heldout_llhood: {float(heldout_llhood)}")

    # print validation time
    config["validation_delta_time"] = time.time() - start_time
    print("Validation delta_time: ", config["validation_delta_time"])

    # config["heldout_llhood"] = float(heldout_llhood.detach().numpy())
    config["heldout_llhood"] = float(heldout_llhood.cpu().numpy())
    with open(os.path.join(param_save_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file)

    # Compute the log likelihood on heldout data using compelte dropped rows
    if kwargs["run_hold_out_rows"]:
        if dataset.heldout_data is not None:
            compute_heldout_rows_llhood(
                kwargs, dataset, param_save_dir, summary_writer, device, config, model
            )
            test_delta_time = time.time() - start_time
            print("Test delta_time: ", test_delta_time)
            config["test_delta_time"] = test_delta_time
            with open(os.path.join(param_save_dir, "config.yaml"), "w") as file:
                yaml.dump(config, file)
        else:
            print(
                "No row-heldout data was provided. Skipping heldout rows log likelihood computation."
            )
    else:
        print("ERROR!")
        print("Skipping heldout rows log likelihood computation")

    print("ELBO: ", -elbo)

def impute_model(**kwargs):
    """
    Configure and train the model and compute held-out loglikelihood
    """
    # check if kwargs dictionary has a key called 'picklePath'
    if "seed" not in kwargs:
        kwargs["seed"] = 0

    # Keep track of total time
    start_time = time.time()

    print(f'Setting seed to {kwargs["seed"]}')
    torch.manual_seed(kwargs["seed"])

    # Load the pickle dataset
    picklePath = pickPicklePath(kwargs["picklePath"], kwargs["factor_model"])
    print(f"Loading dataset from {picklePath}")
    with open(picklePath, "rb") as f:
        dataset = pickle.load(f)

    dataset = apply_preprocessing(
        dataset, kwargs["factor_model"], kwargs["binarize_data"]
    )

    # Set up the output directory
    param_save_dir = setup_output_dir(kwargs["save_dir"], kwargs["factor_model"])
    print(f"Saving to {param_save_dir}")

    num_datapoints, data_dim = dataset.counts.shape
    summary_writer = SummaryWriter(param_save_dir)
    # set the device from the kwargs
    if kwargs["device"] == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print the device
    print(f"Using device: {device}")

    config = kwargs.copy()
    config["device"] = f"{device}"
    config["num_datapoints"] = num_datapoints
    config["data_dim"] = data_dim
    config["summary_writer"] = summary_writer
    config["param_save_dir"] = param_save_dir
    config["picklePath"] = picklePath
    config["start_time"] = start_time

    model = ModelHandler.model_factory(config)

    # Do not save the summary writer
    tmp_config = config.copy()
    tmp_config.pop("summary_writer", None)
    with open(os.path.join(param_save_dir, "config.yaml"), "w") as file:
        yaml.dump(tmp_config, file)

    model, elbo, optim_diverged = train_model(
        model=model,
        dataset=dataset,
        batch_size=kwargs["batch_size"],
        max_steps=kwargs["max_steps"],
        row_learning_rate=kwargs["row_learning_rate"],
        column_learning_rate=kwargs["column_learning_rate"],
        mixture_learning_rate=kwargs["mixture_learning_rate"],
        use_warmup=kwargs["use_warmup"],
        optimizer=kwargs["optimizer"],
        scheduler=kwargs["scheduler"],
        scheduler_patience=kwargs["scheduler_patience"],
        device=device,
        log_cadence=kwargs["log_cadence"],
        tolerance=kwargs["tolerance"],
        summary_writer=summary_writer,
        param_save_dir=param_save_dir,
        retrain=False,
        config=config,
        init=kwargs["init"],
        save_checkpoint=kwargs["save_checkpoint"],
        restore_best_model=kwargs["restore_best_model"],
        train_mode_switch=kwargs["train_mode_switch"],
        use_custom_scheduler=kwargs["use_custom_scheduler"],
        use_batch_sampler=kwargs["use_batch_sampler"],
        schedule_free_epochs=kwargs["schedule_free_epochs"],
        elbo_mode=kwargs["elbo_mode"],
        track_grad_var=kwargs["track_grad_var"],
        n_elbo_particles=kwargs["n_elbo_particles"],
        clipGradients=kwargs["clipGradients"],
        max_gradient_norm=kwargs["max_gradient_norm"],
        stopping_loss_threshold=kwargs["stopping_loss_threshold"],
        init_exp_dir=kwargs["init_exp_dir"],
    )

    # add timestamp
    config["train_delta_time"] = time.time() - start_time
    print("Train delta_time: ", config["train_delta_time"])

    if "summary_writer" in config.keys():
        config.pop("summary_writer", None)

    # Compute the log likelihood on heldout data (heldout elements) using masked enteries
    print("Computing MAE and MSE")

    # Comptue RMSE for the masked values
    # NOTE: This is not meant for performance
    # Using two methods
    # 1. First compute the posterior mean, then report one RMSE
    # 2. For each sampled X, compute a RMSE, then report its mean and svd
    # Sample 100 datasets from the model, find the masked values, compute the RMSE per sampled X
    # find zero indeces of the mask
    heldout_indx = np.array(dataset.holdout_mask.A, dtype=bool)

    n_rmse = kwargs["n_llhood_samples"]
    rmses = []
    maes = []
    imputed_sum = np.zeros_like(dataset.counts[heldout_indx])
    x_true = np.array(dataset.counts[heldout_indx])
    def _compute_natural_mean(a, b):
        return -a / (2*b)
    
    row_mean = model.row_distribution.location
    col_mean = model.column_distribution.location
    the_mean = row_mean.T.matmul(col_mean.T)
    # convert to numpy
    the_mean = the_mean.detach().cpu().numpy()  

    rmse = np.sqrt(np.mean((the_mean[heldout_indx] - x_true) ** 2))
    mae = np.mean(np.abs(the_mean[heldout_indx] - x_true))
    # find non-zero elements of x_true
    non_zero_indx = np.where(x_true != 0)
    # compute the mean and sd of the RMSEs
    non_zero_mean = the_mean[heldout_indx][non_zero_indx[1]]
    rmse_non_zero = np.sqrt(np.mean((non_zero_mean - x_true[non_zero_indx]) ** 2))
    mae_non_zero = np.mean(np.abs(non_zero_mean - x_true[non_zero_indx]))
    # print both non zero ones
    print("RMSE_non_zero is ", rmse_non_zero)
    print("MAE_non_zero is ", mae_non_zero)
    

    # what is the mean of model.row_distribution.eta1/
    for i in tqdm(range(n_rmse)):
        x = model.generate_data(1, None, None)[0].cpu().numpy()
        # keep the sums of the imputed values
        imputed_sum += x[heldout_indx]
        # compute the RMSE using numpy
        rmse = np.sqrt(np.mean((x[heldout_indx] - x_true) ** 2))
        mae = np.mean(np.abs(x[heldout_indx] - x_true))
        rmses.append(rmse)
        maes.append(mae)
    # compute the posterior mean
    imputed_sum = np.array(imputed_sum / n_rmse)
    rmse_point_estimate = np.sqrt(np.mean((imputed_sum - x_true) ** 2))
    mae_point_estimate = np.mean(np.abs(imputed_sum - x_true))

    # compute the mean and sd of the RMSEs
    rmse_mean = np.mean(rmses)
    rmse_sd = np.std(rmses)

    # compute the mean and sd of the MAEs
    mae_mean = np.mean(maes)
    mae_sd = np.std(maes)

    # write all in the config
    config["rmse_point_estimate"] = float(rmse_point_estimate)
    config["rmse_mean"] = float(rmse_mean)
    config["rmse_sd"] = float(rmse_sd)
    config["mae_point_estimate"] = float(mae_point_estimate)
    config["mae_mean"] = float(mae_mean)
    config["mae_sd"] = float(mae_sd)

    config["rmse_non_zero"] = float(rmse_non_zero)
    config["mae_non_zero"] = float(mae_non_zero)

    # print the mae point estimates
    print("MAE_point_estimate is ", mae_point_estimate)

    # print the rmse point estimates
    print("RMSE_point_estimate is ", rmse_point_estimate)

    # print the mae mean
    print("MAE_mean is ", mae_mean)

    # print the rmse mean
    print("RMSE_mean is ", rmse_mean)

    # print validation time
    config["validation_delta_time"] = time.time() - start_time
    print("Validation delta_time: ", config["validation_delta_time"])

    with open(os.path.join(param_save_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file)


def compute_heldout_rows_llhood(
    kwargs, dataset, param_save_dir, summary_writer, device, config, model
):
    """
    Reconfigures the model to compute the heldout rows log likelihood.

    Cleanup all row variables, including:
    - All the embeddings (loadings)
    """
    out_dataset = dataset.heldout_data

    if out_dataset.counts.shape[0] == 0:
        print("No rows were heldout. Skipping heldout rows log likelihood computation.")
        return None
    # Update the dataset
    model.num_datapoints = out_dataset.counts.shape[0]

    # TODO: check that the row prior parameters are identical before and after resetting
    # row_prior_params_before = torch.cat([p.flatten().clone().detach() for p in model.row_distribution.prior.parameters() if p.requires_grad])

    # For Twin (not TwinPlus), copy the row prior's VampPrior values, reset, then init them with those values again
    # if model class name ends at Twin
    CLONE_PESEUDO_OBS = True
    # TODO: movie this in the the vamprior itself
    if CLONE_PESEUDO_OBS:
        pseudo_obs_copy = None
        if model.__class__.__name__.endswith(
            "Twin"
        ) or model.__class__.__name__.endswith("EB"):
            if "Natural" in model.__class__.__name__ :
                if hasattr(model.row_distribution.prior, "eta1"):
                    # pseudo_obs_copy = [model.row_distribution.prior.eta1.clone().detach().cpu(),  model.row_distribution.prior.eta2.clone().detach().cpu()]
                    pseudo_obs_copy = [
                        model.row_distribution.prior.eta1.clone().detach().cpu(),
                        model.row_distribution.prior.eta2.clone().detach().cpu(),
                        model.row_distribution.prior.mix_vals.clone().detach().cpu(),
                        model.row_distribution.prior.log_e0.clone().detach().cpu(),
                    ]
                else:
                    # get the params from the prior 
                    pseudo_obs_copy = model.row_distribution.prior.get_params()
            else:
                pseudo_obs_copy = [
                    model.row_distribution.prior.pseudo_obs.clone().detach().cpu(),
                    model.row_distribution.prior.log_scale.clone().detach().cpu(),
                ]

    tmp = _get_args_for_setup_row_vars(model, kwargs)
    model._setup_row_vars(**tmp)

    if CLONE_PESEUDO_OBS:
        if pseudo_obs_copy is not None:
            model.row_distribution.prior.init_pseudo_obs(pseudo_obs_copy)
            # row_prior_params_post = torch.cat([p.flatten().clone().detach() for p in model.row_distribution.prior.parameters() if p.requires_grad])
            # # check they are identical before and post
            # assert torch.allclose(row_prior_params_before, row_prior_params_post), "row_prior_params_before != row_prior_params_post"

    # check if model.row_distribution.prior has a parameters function,
    if hasattr(model.row_distribution.prior, "parameters"):
        for p in model.row_distribution.prior.parameters():
            p.requires_grad = False

    # model._setup_row_vars(None, concentration=kwargs['row_prior_concentration'], rate=kwargs['row_prior_rate'])

    # Fix column variable parameters
    for p in model.column_distribution.parameters():
        # TODO: check that this includes all involved variables
        p.requires_grad = False

    ll_param_save_dir = os.path.join(param_save_dir, RETRAINED_MODEL_DIR)
    ll_summary_writer = SummaryWriter(ll_param_save_dir)
    model.summary_writer = ll_summary_writer

    new_config = config.copy()
    new_config["summary_writer"] = ll_summary_writer
    new_config["num_datapoints"] = model.num_datapoints
    new_config["param_save_dir"] = ll_param_save_dir
    new_config["retraining"] = True

    # TODO: Ensure that pseudo_obs in AmortizedPMFEB resembled the data

    ## Train on a much sparser dataset than the originanl 20%, but still evaluate on the exact same
    # 1. make a copy of the out_dataset
    # 2. mask n_new_mask perent of its ones
    # TODO: this is a hack
    train_out_dataset = out_dataset
    masking_factor = kwargs["masking_factor"]

    the_vad, the_holdout_mask = out_dataset.vad, out_dataset.holdout_mask
    do_overhaul = True
    vad_fraction = 0.3  # fraction of entries kept for validation
    if masking_factor > 0:
        # ensure that it is below 1
        assert (
            masking_factor < 1
        ), f"masking_factor must be below 1, but is {masking_factor}"
        # make a copy of the out_dataset
        new_out_dataset = copy.deepcopy(out_dataset)
        if do_overhaul:
            print("Overhauling the test dataset...")
            # outputs:
            # 1. (new_vad, new_vad_holdout_mask)
            # 2. (new_train, new_train_holdout_mask)
            # 1. first pick your validation dataset at vad_fraction of the counts
            # 2. then, from the rest of it, filter masking_factor for training
            # change the masked items in the test set, so that only non-zeros's are masked

            def _create_masked(counts, vad_fraction, rnd_seed=0):
                """
                vad_fraction: fraction of the non-zero elements to be masked
                Output:
                    - train matrix, holdoutmask, vad matrix
                non_zero_indexes: to force picking from these set
                """
                np.random.seed(rnd_seed)
                new_train = counts.copy()
                non_zero_indexes = new_train.nonzero()
                n_new_mask = int(vad_fraction * non_zero_indexes[0].shape[0])
                random_indexes = np.random.choice(
                    non_zero_indexes[0].shape[0], n_new_mask, replace=False
                )
                # pick zero-out indexes
                rnd_rows, rnd_cols = (
                    non_zero_indexes[0][random_indexes],
                    non_zero_indexes[1][random_indexes],
                )
                new_train[rnd_rows, rnd_cols] = 0
                new_holdout = sparse.csr_matrix(
                    (np.ones(n_new_mask), (rnd_rows, rnd_cols)), shape=new_train.shape
                )
                new_vad = counts.multiply(new_holdout)
                return new_train, new_holdout, new_vad

            # pick the validation set
            new_train, the_holdout_mask, the_vad = _create_masked(
                new_out_dataset.counts, vad_fraction, rnd_seed=0
            )
            # TODO: sanity check the_holdout_mask
            # new_train.nonzero()[0].shape
            new_train, new_holdout, _ = _create_masked(
                new_train, masking_factor, rnd_seed=1
            )

            # set entries of vad to zero in new_holdout
            new_holdout = new_holdout + the_holdout_mask

            new_out_dataset.train = new_train
            new_out_dataset.holdout_mask = new_holdout
        else:
            # change the holdout mask and train
            # 1. find non-zero indexes in new_out_dataset.train
            non_zero_indexes = new_out_dataset.train.nonzero()
            # randomly selet masking_factor of them
            n_new_mask = int(masking_factor * non_zero_indexes[0].shape[0])
            # randomly select n_new_mask indexes
            random_indexes = np.random.choice(
                non_zero_indexes[0].shape[0], n_new_mask, replace=False
            )
            # Set random_indexes to zero in both train and holdout_mask
            new_out_dataset.train[
                non_zero_indexes[0][random_indexes], non_zero_indexes[1][random_indexes]
            ] = 0
            new_out_dataset.holdout_mask[
                non_zero_indexes[0][random_indexes], non_zero_indexes[1][random_indexes]
            ] = 0
            # assert that they have fewer non-zero elements than their orignal counterparts
            assert (
                new_out_dataset.train.nonzero()[0].shape[0]
                < out_dataset.train.nonzero()[0].shape[0]
            ), "new_out_dataset.train has more non-zero elements than out_dataset.train"

        train_out_dataset = new_out_dataset

    # now outdataset
    def quick_plot(mat, filePath):
        import matplotlib.pyplot as plt

        plt.clf()
        # use the gray colormap
        plt.imshow(mat.todense(), cmap="gray_r", aspect="auto")
        # add a colorbar
        plt.colorbar()
        plt.savefig(filePath)

    # TODO: plot what is given to the model for training
    def __test():
        quick_plot(
            train_out_dataset.train,
            os.path.join(param_save_dir, "train_out_dataset.png"),
        )
        quick_plot(
            out_dataset.train,
            os.path.join(param_save_dir, "train_out_dataset_orig.png"),
        )
        quick_plot(
            train_out_dataset.vad,
            os.path.join(param_save_dir, "train_out_dataset_vad_orig.png"),
        )
        # if the model is twin, visualize the prior mixture values
        # if 'Twin' in model.__class__.__name__:

        # print the fraction of non-zero elements in the train_out_dataset
        print(
            "Fraction of non-zero elements in the train_out_dataset: ",
            train_out_dataset.train.nonzero()[0].shape[0]
            / train_out_dataset.train.shape[0]
            / train_out_dataset.train.shape[1],
        )
        print(
            "Fraction of non-zero elements in the out_dataset: ",
            out_dataset.train.nonzero()[0].shape[0]
            / out_dataset.train.shape[0]
            / out_dataset.train.shape[1],
        )

    # clone the columwise parameters
    train_model(
        model=model.to(device).double(),
        dataset=train_out_dataset,
        batch_size=kwargs["batch_size"],
        max_steps=int(
            np.ceil(kwargs["max_steps"] / 2)
        ),  # spend half as much time on this
        use_warmup=kwargs["use_warmup"],
        row_learning_rate=kwargs["row_learning_rate"],
        column_learning_rate=kwargs["column_learning_rate"],
        mixture_learning_rate=kwargs[
            "mixture_learning_rate"
        ],  # should this be set to zero?
        optimizer=kwargs["optimizer"],
        scheduler=kwargs["scheduler"],
        scheduler_patience=kwargs["scheduler_patience"],
        device=device,
        log_cadence=kwargs["log_cadence"],
        tolerance=kwargs["tolerance"],
        summary_writer=ll_summary_writer,
        param_save_dir=ll_param_save_dir,
        retrain=True,
        config=new_config,
        save_checkpoint=kwargs["save_checkpoint"],
        restore_best_model=kwargs["restore_best_model"],
        train_mode_switch=kwargs["train_mode_switch"],
        use_custom_scheduler=kwargs["use_custom_scheduler"],
        use_batch_sampler=kwargs["use_batch_sampler"],
        schedule_free_epochs=kwargs["schedule_free_epochs"],
        elbo_mode=kwargs["elbo_mode"],
        track_grad_var=kwargs["track_grad_var"],
        n_elbo_particles=kwargs["n_elbo_particles"],
        clipGradients=kwargs["clipGradients"],
        max_gradient_norm=kwargs["max_gradient_norm"],
        stopping_loss_threshold=kwargs["stopping_loss_threshold"],
    )
    # TODO: sanity check: ensure that the global variables are the same as the initialization
    # col_params_post =torch.cat([p.flatten().clone().detach() for p in model.column_distribution.parameters()])

    # check that the column parameters are the same
    # assert torch.allclose(
    #     col_params_pre, col_params_post), "Column parameters changed during heldout rows training"

    # Compute the log likelihood on heldout data using removed rows
    print("Computing heldout log likelihood on heldout-rows...")

    vad_torch, missing_indexes = _prepare_data_for_heldout_loglikelihood(
        vad=the_vad, holdout_mask=the_holdout_mask, device=device
    )
    # vad_torch, missing_indexes = _prepare_data_for_heldout_loglikelihood(
    #     vad=out_dataset.vad, holdout_mask=out_dataset.holdout_mask, device=device
    # )

    heldout_llhood_rows = model.compute_heldout_loglikelihood(
        vad_torch,
        missing_indexes,
        n_monte_carlo=kwargs["n_llhood_samples"],
        subsample_zeros=kwargs["subsample_zeros"],
    )
    print("heldout_llhood_rows: ", float(heldout_llhood_rows))

    config["heldout_llhood_rows"] = float(heldout_llhood_rows.cpu().numpy())
    # Update the original config
    with open(os.path.join(param_save_dir, "config.yaml"), "w") as file:
        yaml.dump(config, file)

    summary_writer.add_scalar(f"llhood_test/llhood", heldout_llhood_rows)
    summary_writer.close()

    # Save the llhood config
    new_config.pop("summary_writer", None)

    with open(os.path.join(ll_param_save_dir, "config.yaml"), "w") as file:
        yaml.dump(new_config, file)


if __name__ == "__main__":
    """
    Main function, pick among one of the three main tasks.
    Synopsis:
        python driver.py [setup_data|run_model|plot_results] [options]

    See below for details of each action.


    Example usage:
        ./driver.py setup_data -i ../data/lx95_uu_uttu_1000_highly_var.h5ad -o ../data/ -p .1 -f True
        ./driver.py setup_data -i ../data/lx95_uu_uttu_all_highly_var.h5ad -o ../data/ -p .2 -f True -l .8

        python3 driver.py run_model --configPath ../pipelines/test_config.yaml
        ./driver.py run_model -i ../data/lx95_uu_uttu_1000_highly_var_0.1.pkl -o ../results -l 32 -s 1 -t False -m 500 -f AmortizedPPCAEB_full -b 200 -r 0.01 -tol 3 -a 1.0 -k 10 -n 1 -svd 1.0 -c 10
    """
    task_name = sys.argv[1]
    if task_name == "setup_data":
        arg_parser = main_data_setup_args()
        args = arg_parser.parse_args(sys.argv[2:])
        create_heldout_data(
            args.filePath,
            args.saveDir,
            holdout_portion=args.holdoutPortion,
            force=args.force,
            data_cache_path=args.cacheDir,
            correlation_limit=args.correlationCutOff,
            holdout_rows=args.holdoutRows,
            ignore_pca=args.ignore_pca,
            seed=args.seed,
        )
        sys.exit()
    elif task_name == "run_model":
        start_time = time.time()
        if sys.argv[2] == "--configPath":
            config_path = sys.argv[3]
            ch = ConfigHandler(config_path)
            args = ch.parse_config()
        else:
            arg_parser = main_run_model_setup_args()
            args = arg_parser.parse_args(sys.argv[2:])
        run_model(**vars(args))
        delta_time = time.time() - start_time
        print("Total time: ", delta_time)
        sys.exit()
    elif task_name == "retrain_model":
        """
        Given an expPath, retrain or not switch, and new set of optimization params,
        1. load the model (or the retrained)
        2. update the optimization params
        3. re measure performance
        """
        import argparse

        arg_parser = argparse.ArgumentParser(
            description="Rerun a model from a checkpoint directory."
        )
        # 1. string traint_tag
        # 2. exp_dir
        arg_parser.add_argument(
            "--exp_dir", "-e", type=str, help="The path to the experiment directory."
        )
        arg_parser.add_argument(
            "--train_tag",
            "-t",
            type=str,
            help="Which model to rerun",
            default=TRAINED_MODEL_TAG,
        )
        arg_parser.add_argument(
            "--mixture_learning_rate",
            "-mlr",
            type=none_or_float,
            help="Learning rate for the mixture variables",
            default=None,
        )
        arg_parser.add_argument(
            "--column_learning_rate",
            "-clr",
            type=none_or_float,
            help="Learning rate for the column variables",
            default=None,
        )
        arg_parser.add_argument(
            "--row_learning_rate",
            "-rlr",
            type=none_or_float,
            help="Learning rate for the rate variables",
            default=None,
        )
        arg_parser.add_argument(
            "--max_steps",
            "-m",
            type=none_or_int,
            help="Maximum number of steps",
            default=None,
        )
        args = arg_parser.parse_args(sys.argv[2:])
        retrain_model_from_check_point(**vars(args))
    elif task_name == "impute_model":
        start_time = time.time()
        if sys.argv[2] == "--configPath":
            config_path = sys.argv[3]
            ch = ConfigHandler(config_path)
            args = ch.parse_config()
        else:
            arg_parser = main_run_model_setup_args()
            args = arg_parser.parse_args(sys.argv[2:])
        impute_model(**vars(args))
        delta_time = time.time() - start_time
        print("Total time: ", delta_time)
        sys.exit()
    else:
        raise ValueError("Invalid task name.")
