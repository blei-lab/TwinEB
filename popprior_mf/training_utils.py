# """
#     Training utils
#     @Author: De-identified Author
# """

import torch
import numpy as np
from gradient_utils import GradientTracer, GradientVarianceMonitor
from model_factory import ModelHandler, str_to_class
from scipy.sparse import coo_matrix, csr_matrix, vstack
from sklearn.decomposition._nmf import _initialize_nmf
import os

from utils import kmeans_init, transform_nmf, transform_pca

################################################################
## Optimization utilities
################################################################


def _setup_optimization(
    model,
    row_learning_rate,
    column_learning_rate,
    mixture_learning_rate,
    optim="Adam",
    scheduler="",
    scheduler_step_size=1,
    scheduler_patience=3,
):
    """
    Setup the optimizer and its scheduler for each model.

    Args:
        model: the initialized model
        learning_rate: initial learning rate

    Returns:
        optimizer: the optimizer to use for the modesl
        scheduler: the scheduler to use for the model
    """

    optimizer = config_optimization(
        model=model,
        row_learning_rate=row_learning_rate,
        column_learning_rate=column_learning_rate,
        mixture_learning_rate=mixture_learning_rate,
        optimizer=optim,
    )
    if scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            # optimizer, "min", factor=0.1, patience=10, verbose=True
            optimizer,
            "min",
            factor=0.99,
            # min_lr=1e-5, ## added dec 30
            # patience=3,
            patience=scheduler_patience,
            verbose=False,
        )
    elif scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=0.99, verbose=False
        )
    elif scheduler == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-5,
            max_lr=np.maximum(row_learning_rate, column_learning_rate),
            step_size_up=100,
            cycle_momentum=False,
        )
    else:
        print(f"Could not find scheduler {scheduler}, defaulting to None.")
        scheduler = None

    return [optimizer], [scheduler]


def config_optimization_row_column(
    model, row_learning_rate, column_learning_rate, weight_decay=1e-5, optimizer="Adam"
):
    # TODO: check the optimizer exists
    optimClass = str_to_class(f"{optimizer}")
    optimizer = optimClass(
        params=[
            {
                "params": [
                    p
                    for p in model.row_distribution.parameters()
                    if p.requires_grad == True
                ]
            },
            {
                "params": [
                    p
                    for p in model.column_distribution.parameters()
                    if p.requires_grad == True
                ],
                "lr": column_learning_rate,
            },
        ],
        lr=row_learning_rate,
        weight_decay=weight_decay,
    )
    # TODO: set the params of the mixture separately

    return optimizer


def config_optimization(
    model,
    row_learning_rate,
    column_learning_rate,
    mixture_learning_rate,
    weight_decay=1e-5,
    optimizer="Adam",
):
    # Divide the params into 3 groups: 1. row_prior_params and column_prior_params, 2. all other row params, 3. all other column parmas
    optimClass = str_to_class(f"{optimizer}")

    def filter_params(big_grp, small_grp):
        """
        Removes params that are in the small group from the big group.
        Based on the shapes.
        """
        res = []
        for p in big_grp:
            found = False
            for pp in small_grp:
                if p.shape == pp.shape:
                    found = True
                    break
            if found == False:
                res.append(p)
        return res

    # check if Twin is in the name of the model
    if "TwinXXX" in model.__class__.__name__:
        print('Warning: using old optimization scheme for twin')
        # Find the row_prior specific params
        row_prior_params = [
            p
            for p in model.row_distribution.prior.parameters()
            if p.requires_grad == True
        ]
        # Find the column_prior specific params
        column_prior_params = [
            p
            for p in model.column_distribution.prior.parameters()
            if p.requires_grad == True
        ]
        row_params = [
            p for p in model.row_distribution.parameters() if p.requires_grad == True
        ]
        column_params = [
            p for p in model.column_distribution.parameters() if p.requires_grad == True
        ]
        # Now filter out
        row_params = filter_params(row_params, row_prior_params)
        column_params = filter_params(column_params, column_prior_params)

        # Now setup the optimizer, where there are 3 differnt learning rates. one for the row_prior and column_prior, one for the row, and one for the column
        optimizer = optimClass(
            params=[
                {
                    "params": row_prior_params + column_prior_params,
                    "lr": mixture_learning_rate,
                },
                {
                    "params": row_params,
                    "lr": row_learning_rate,
                },
                {
                    "params": column_params,
                    "lr": column_learning_rate,
                },
            ],
            lr=mixture_learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optimClass(
            params=[
                {
                    "params": [
                        p
                        for p in model.row_distribution.parameters()
                        if p.requires_grad == True
                    ]
                },
                {
                    "params": [
                        p
                        for p in model.column_distribution.parameters()
                        if p.requires_grad == True
                    ],
                    "lr": column_learning_rate,
                },
            ],
            lr=row_learning_rate,
            weight_decay=weight_decay,
        )

    # for param_group in optimizer.param_groups:
    #     print(f"Learning rate for param group: {param_group['lr']}")

    return optimizer


def init_params(model, dataset, row_only=False, init="nmf"):
    """
    Initialize the parameters of the model, using vanila PCA or NMF

    Args:
        model (torch.nn.Module): The model to initialize.
        dataset (torch.utils.data.Dataset): The dataset to use for initialization.

    Returns:
        col_vars, row_vars: init_loc for the column vars and location variables.
    """
    # raise NotImplementedError("This function is not used anymore.")
    # Initialize using PCA of NMF
    if "PCA" in model.__class__.__name__:
        col_vars, row_vars = transform_pca(dataset.train, model.latent_dim)
    elif "PMF" in model.__class__.__name__:
        if init == "nmf":
            col_vars, row_vars = transform_nmf(
                dataset.train.A,
                model.latent_dim,
                max_iter=15000,
                beta_loss="kullback-leibler",
                solver="mu",
            )
            print("Initing param for PMF...")
        elif init in ["nndsvda", "nndsvd", "nndsvdar", "random"]:
            # col_var.shape, row_vars.shape: (n_features, n_components), (n_components, n_obs)
            W, H = _initialize_nmf(
                dataset.train.A, model.latent_dim, init=init, random_state=0
            )
            col_vars = H.T
            row_vars = W.T
        elif init == "kmeans":
            col_vars, row_vars = kmeans_init(
                dataset.train.A, model.latent_dim, n_neighbors=0
            )
        else:
            raise ValueError(f"Unrecognized init method {init}")
    else:
        raise ValueError(f"Model {model.__class__.__name__} not supported.")

    if row_only is False:
        model.init_column_vars(col_vars)

    model.init_row_vars(row_vars)
    return col_vars, row_vars


################################################################
## Data loader utilities
################################################################


def sparse_coo_to_tensor(coo: coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)


## Set of three methods that DO NOT use a BatchSampler


def sparse_batch_collate(batch: list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    idxs, data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch)
    return torch.tensor(idxs), data_batch, targets_batch


## Set of three methods that USE a BATCH SAMPLER


def sparse_batch_collate_batch(batch: list):
    """
    Uses a BATCH sampler
    """
    idxs, data_batch, targets_batch = batch[0]
    if type(data_batch[0]) == csr_matrix:
        data_batch = data_batch.tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = targets_batch.tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch)
    return torch.tensor(idxs), data_batch, targets_batch


################################################################
## Training schedule utilities
################################################################


class CustomTrainer:
    """
    Supports training with gradient monitoring checkpoints.
    Advances one step (one batch) in the current epoch.
    """

    def __init__(
        self,
        model,
        optimizers,
        max_steps=None,
        n_epoch=None,
        epoch_len=None,
        retrain=False,
        elbo_mode="parallel",
        track_grad_var=False,
        n_elbo_particles=3,
        clip_gradients=True,
        max_gradient_norm=2.0,
    ):
        """

        Parameters:
        ----------
        elbo_mode: either parallel (default) or sequential. Must be sequential if tracing gradients.

        """
        self.model = model
        self.optimizers = optimizers
        self.track_grad_var = track_grad_var
        self.max_steps = max_steps
        self.retrain = retrain
        self.n_epoch = n_epoch
        self.epoch_len = epoch_len
        self.n_elbo_particles = n_elbo_particles
        self.clip_gradients = clip_gradients
        self.max_gradient_norm = max_gradient_norm
        self.elbo_mode = elbo_mode
        assert self.elbo_mode in [
            "parallel",
            "sequential",
        ], f"elbo_mode must be in parallel or sequential, but was {self.elbo_mode}"
        if track_grad_var == True:
            assert (
                self.elbo_mode == "sequential"
            ), f"elbo_mode has to be sequential if tracing gradients."

        self._init_grad_tracer()

    def _init_grad_tracer(self):
        self.gradTracer = GradientTracer(
            model=self.model,
            optimizers=self.optimizers,
            epoch_len=self.epoch_len,
            n_epoch=self.n_epoch,
            max_steps=self.max_steps,
        )


    def _get_all_params(self):
        return [p for p in self.model.parameters() if p.requires_grad == True]


    def _compute_sequential_elbo(
        self, datapoints_indices, x_train, holdout_mask, step, epoch, summary_writer
    ):
        """
        Fit the larger datasets in memory by using n_samples = 1 and instead forcing monte_carlo sampling here
        """
        S = self.n_elbo_particles
        aggregate = (0, 0, 0)
        mean_elbo = 0
        for _ in range(S):
            elbo = 0
            elbo = self.model(
                datapoints_indices,
                x_train,
                holdout_mask,
                step + epoch * self.epoch_len,
                self.n_epoch * self.epoch_len,
            )
            loss = -elbo
            loss.backward(retain_graph=True)
            all_params = self._get_all_params()
            all_grads = torch.cat([p.grad.flatten() for p in all_params])
            aggregate = GradientVarianceMonitor.update(aggregate, all_grads)
            mean_elbo += elbo.item()
        mean_elbo /= S
        grad_mean, grad_var, grad_sample_var = GradientVarianceMonitor.finalize(
            aggregate
        )

        with torch.no_grad():
            all_params = self._get_all_params()
            last_index = 0
            for p in all_params:
                p.grad = grad_mean[last_index : last_index + p.numel()].reshape(p.shape)
                last_index += p.numel()

            # record the ELBO
            summary_writer.add_scalar(
                "elbo/mean_elbo", mean_elbo, step + epoch * self.epoch_len
            )

        return loss

    def __call__(
        self, step, epoch, summary_writer, datapoints_indices, x_train, holdout_mask
    ):
        """
        Takes one gradient step for a mini-batch and returns the loss.
        Can track the variance of gradients at certain checkpionts.
        """
        for optim in self.optimizers:
            optim.zero_grad()

        if self.track_grad_var & (not self.retrain):
            self.gradTracer(
                summary_writer=summary_writer,
                step=step,
                epoch=epoch,
                datapoints_indices=datapoints_indices,
                x_train=x_train,
                holdout_mask=holdout_mask,
            )

        if self.elbo_mode == "sequential":
            # print(self.elbo_mode)
            loss = self._compute_sequential_elbo(
                datapoints_indices=datapoints_indices,
                x_train=x_train,
                holdout_mask=holdout_mask,
                step=step,
                epoch=epoch,
                summary_writer=summary_writer,
            )
        else:
            elbo = self.model(
                datapoints_indices,
                x_train,
                holdout_mask,
                step + epoch * self.epoch_len,
                self.n_epoch * self.epoch_len,
            )
            loss = -elbo
            loss.backward(retain_graph=True)

        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(
                # TODO: what should this norm be?
                self.model.parameters(),
                max_norm=self.max_gradient_norm,
                norm_type=2,
            )

        for optim in self.optimizers:
            optim.step()

        return loss


class SwitchTrainer:
    """
    Switches between row, column, and joint training.

    Waits for WARM_UP_EPOCHS before starting to switch.
    """

    def __init__(
        self,
        train_both=True,
        train_column=False,
        train_mode="both",
        warm_up_epochs=1000,
        train_mode_switch=True,
    ):
        self.train_both = train_both
        self.train_column = train_column
        self.train_mode = train_mode
        self.WARM_UP_EPOCHS = warm_up_epochs
        self.train_mode_switch = train_mode_switch

    def update_status(self, epoch):
        if self.train_mode_switch:
            if epoch < self.WARM_UP_EPOCHS:
                self.train_mode = "both"
            else:
                self.train_mode = "single"

                # if train_mode_switch:
                if epoch > 0 and epoch % 100 == 0:
                    # switch train_mode
                    # train_mode = "both" if train_mode == "single" else "single"
                    pass

                if self.train_mode == "single":
                    self.train_both = False
                    # swtich train_global, if epoch is divisible by 20
                    if epoch % 10 == 0:
                        self.train_column = (
                            True if self.train_column is False else False
                        )
                else:
                    # train mode is both
                    self.train_both = True

    def get_train_str(self):
        train_str = "both" if self.train_both else "single"
        if train_str == "single":
            train_str = "column" if self.train_column else "row   "
        return train_str

    def __call__(self, model):
        if self.train_mode_switch:
            if self.train_both:
                for p in model.row_distribution.parameters():
                    p.requires_grad = True
                for p in model.column_distribution.parameters():
                    p.requires_grad = True
            else:
                if self.train_column:
                    # Train the column parameters for the fixed row parameters for one epoch
                    # disable grad for the row parameters
                    for p in model.row_distribution.parameters():
                        p.requires_grad = False
                    # enable grad for the column parameters
                    for p in model.column_distribution.parameters():
                        p.requires_grad = True
                else:
                    # Train the row parameters for the fixed column parameters for one epoch
                    # disable grad for the column parameters
                    for p in model.column_distribution.parameters():
                        p.requires_grad = False
                    # enable grad for the row parameters
                    for p in model.row_distribution.parameters():
                        p.requires_grad = True


class EarlyStopping:
    def __init__(
        self, prev_loss=1e10, min_loss=1e10, tolerance=10, stopping_loss_threshold=0
    ):
        self.prev_loss = prev_loss
        self.min_loss = min_loss
        self.tolerance = tolerance
        self.tol_indx = 0
        self.optimization_diverged = False
        self.stopping_loss_threshold = stopping_loss_threshold
        self.early_stop = False

    def __call__(self, valid_loss, optimizers=None):
        # Check for early stopping
        if valid_loss - self.prev_loss < self.stopping_loss_threshold:
            self.tol_indx = 0
        else:
            self.tol_indx += 1
        # If the train_loss has gotten orders of magnitude more than the previous one, it has diverged
        if np.abs(valid_loss) > 1000 * np.abs(self.prev_loss):
            print(
                f"Train loss has diverged {valid_loss} vs {self.prev_loss}. Stopping training."
            )
            self.optimization_diverged = True
            self.early_stop = True

        self.prev_loss = valid_loss
        
        if self.tol_indx >= self.tolerance:
            print(
                f"Validation loss has gone up for {self.tol_indx} consecutive epochs. Will stop training."
            )
            self.early_stop = True
        
        if optimizers is not None:
            # if learning rate is close to zero, stop training
            min_lr = 1e-8
            lrs = []
            for param_group in optimizers[0].param_groups:
                lrs.append(param_group['lr'])
            if max(lrs) < min_lr:
                print(f"Maximum learning rate ({max(lrs)}) is below the threshold {min_lr}. Stop training.")
                self.early_stop = True

        return self.early_stop


class CustomScheduller:
    def __init__(
        self,
        schedulers,
        optimizers,
        schedule_free_epochs,
        use_custom_scheduler,
        use_warmup,
        epoch_len,
        orig_lrs=None,
    ):
        self.schedulers = schedulers
        self.optimizers = optimizers
        self.schedule_free_epochs = schedule_free_epochs
        self.use_custom_scheduler = use_custom_scheduler
        self.use_warmup = use_warmup
        self.epoch_len = epoch_len
        # Custom scheduller status
        self.schedule_free_epochs_indx = 0
        self.start_scheduler = False
        if orig_lrs is None:
            self._init_orig_lrs()

    def _init_orig_lrs(self):
        # Keep track of original learning rates
        self.orig_lrs = []
        for param_group in self.optimizers[0].param_groups:
            self.orig_lrs.append(param_group["lr"])

    def warm_up_step(self, epoch, step):
        """
        Warm up the learning rate for the first 1/(.5*(1-beta2)) steps
        """
        if self.use_warmup:
            if not self.start_scheduler:
                if len(self.optimizers) > 1:
                    raise ValueError("Cannot warm up with multiple optimizers.")

                if self.optimizers[0].__class__.name == "Adam":
                    beta2 = self.optimizers[0].param_groups[0]["betas"][1]
                else:
                    beta2 = 0.99
                w_t = np.minimum(
                    1.0, 0.5 * (1 - beta2) * (1 + step + epoch * self.epoch_len)
                )

                self.optimizers[0].param_groups[0]["lr"] = self.orig_lrs[0] * w_t
                self.optimizers[0].param_groups[1]["lr"] = self.orig_lrs[1] * w_t
                # print(f'w_t = {w_t}, lr = {optimizers[0].param_groups[0]["lr"]}')
                if w_t == 1.0:
                    self.start_scheduler = True
                    print("###################")
                    print("########## Starting the scheduler... ############")

    def schuduler_step_amortized(
        self,
        epoch,
        loss,
    ):
        """
        Start small (1/10th) for 10 epochs
        Then start increasing until the original lr is reached
        """
        if epoch > 1 and not self.start_scheduler:
            for i in range(len(self.optimizers[0].param_groups)):
                beta2 = self.optimizers[0].param_groups[i]["betas"][1]
                cur_lr = self.optimizers[0].param_groups[i]["lr"]
                # optimizers[0].param_groups[i]["lr"] = np.minimum(
                #     #orig_lrs[i], orig_lrs[i] * 1.05
                #     #orig_lrs[i], cur_lr * 1.05
                #     orig_lrs[i], cur_lr * .5*(1-beta2)*epoch
                # )

            if self.optimizers[0].param_groups[0]["lr"] == self.orig_lrs[i]:
                self.start_scheduler = True
                print("###################")
                print("########## Starting the scheduler... ############")

        if self.start_scheduler:
            self.schedule_free_epochs_indx += 1
            print(f"schedule_free_epochs_indx: {self.schedule_free_epochs_indx}")

        if (
            self.start_scheduler
            and self.schedule_free_epochs_indx > self.schedule_free_epochs
        ):
            print("Scheduler step")
            for scheduler in self.schedulers:
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    scheduler.step(loss)
                else:
                    scheduler.step()

    def __call__(self, epoch, valid_loss):
        if epoch > self.schedule_free_epochs:
            for scheduler in self.schedulers:
                if scheduler is not None:
                    if self.use_custom_scheduler:
                        (start_scheduler,) = self.schuduler_step_amortized(
                            epoch=epoch,
                            start_scheduler=start_scheduler,
                            loss=valid_loss,
                        )
                    else:
                        if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                            scheduler.step(valid_loss)
                        elif scheduler.__class__.__name__ != "CyclicLR":
                            scheduler.step()

    def batch_step(self):
        """
        Have gone over one mini-batch.
        """
        for scheduler in self.schedulers:
            if scheduler is not None:
                if scheduler.__class__.__name__ == "CyclicLR":
                    scheduler.step()


class CheckPointManager:
    """
    Soft wrapper around torch.save
    """

    def __init__(
        self,
        model,
        save_checkpoint,
        min_loss,
        train_tag,
        param_save_dir,
        optimizers,
        restore_best_model=True,
    ):
        self.model = model
        self.save_checkpoint = save_checkpoint
        self.min_loss = min_loss
        self.check_point_epoch = 0
        self.train_tag = train_tag
        self.param_save_dir = param_save_dir
        self.optimizers = optimizers
        self.restore_best_model = restore_best_model

    @staticmethod
    def load_checkpoint(model, path):
        if not os.path.exists(path):
            raise FileExistsError(f"Cannot find checkpoint at {path}")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    def restore_check_point(self, config, model, train_loss):
        if (
            os.path.exists(
                os.path.join(
                    self.param_save_dir, f"model_checkpoint_{self.train_tag}.pt"
                )
            )
            and self.restore_best_model
        ):
            print("Restoring the best checkpoint")
            model = ModelHandler.model_factory(config)
            checkpoint = torch.load(
                os.path.join(
                    self.param_save_dir, f"model_checkpoint_{self.train_tag}.pt"
                )
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            # print the best validation and training loss
            print(f"Best training loss: {checkpoint['train_loss']}")
            print(f"Best validation loss: {checkpoint['valid_loss']}")

            train_loss = checkpoint["train_loss"]

        return model, train_loss

    def __call__(self, epoch, valid_loss, train_loss):
        if self.save_checkpoint:
            if valid_loss < self.min_loss:
                self.min_loss = valid_loss
                if epoch > 0:
                    self.check_point_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizers[0].state_dict(),
                            "train_loss": train_loss,
                            "valid_loss": valid_loss,
                        },
                        os.path.join(
                            self.param_save_dir, f"model_checkpoint_{self.train_tag}.pt"
                        ),
                    )


class GlobalTrainer:
    """
    A omni-class for training

    """

    def __init__(
        self,
        model,
        max_steps,
        epoch_len,
        retrain,
        elbo_mode,
        track_grad_var,
        n_elbo_particles,
        clipGradients,
        max_gradient_norm,
        train_mode_switch,
        save_checkpoint,
        min_loss,
        train_tag,
        param_save_dir,
        restore_best_model,
        optimizer,
        scheduler,
        use_custom_scheduler,
        scheduler_patience,
        use_warmup,
        schedule_free_epochs,
        tolerance,
        stopping_loss_threshold,
        row_learning_rate,
        column_learning_rate,
        mixture_learning_rate,
    ):
        self.model = model
        self.max_steps = max_steps
        self.epoch_len = epoch_len
        self.retrain = retrain
        self.elbo_mode = elbo_mode
        self.track_grad_var = track_grad_var
        self.n_elbo_particles = n_elbo_particles
        self.clipGradients = clipGradients
        self.max_gradient_norm = max_gradient_norm
        self.train_mode_switch = train_mode_switch
        self.save_checkpoint = save_checkpoint
        self.min_loss = min_loss
        self.train_tag = train_tag
        self.param_save_dir = param_save_dir
        self.restore_best_model = restore_best_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_custom_scheduler = use_custom_scheduler
        self.use_warmup = use_warmup
        self.schedule_free_epochs = schedule_free_epochs
        self.scheduler_patience = scheduler_patience
        self.tolerance = tolerance
        self.stopping_loss_threshold = stopping_loss_threshold
        self.row_learning_rate = row_learning_rate
        self.column_learning_rate = column_learning_rate
        self.mixture_learning_rate = mixture_learning_rate

        self.n_epoch = self.max_steps // self.epoch_len

        self._init_optim_scheduler()
        self._initSwitchTrainer()
        self._initCustomTrainer()
        self._initCheckPointManager()
        self._initCustomScheduler()
        self._initEarlyStopping()

    def _init_optim_scheduler(self):
        self.optimizers, self.schedulers = _setup_optimization(
            model=self.model,
            row_learning_rate=self.row_learning_rate,
            column_learning_rate=self.column_learning_rate,
            mixture_learning_rate=self.mixture_learning_rate,
            optim=self.optimizer,
            scheduler=self.scheduler,
            scheduler_patience=self.scheduler_patience,
        )

    def _initCustomScheduler(self):
        self.customScheduler = CustomScheduller(
            schedulers=self.schedulers,
            optimizers=self.optimizers,
            schedule_free_epochs=self.schedule_free_epochs,
            use_custom_scheduler=self.use_custom_scheduler,
            use_warmup=self.use_warmup,
            epoch_len=self.epoch_len,
        )

    def _initEarlyStopping(self):
        self.earlyStopping = EarlyStopping(
            tolerance=self.tolerance,
            stopping_loss_threshold=self.stopping_loss_threshold,
            min_loss=self.min_loss,
        )

    def _initCheckPointManager(self):
        self.checkPointManager = CheckPointManager(
            model=self.model,
            optimizers=self.optimizers,
            save_checkpoint=self.save_checkpoint,
            min_loss=self.min_loss,
            train_tag=self.train_tag,
            param_save_dir=self.param_save_dir,
            restore_best_model=self.restore_best_model,
        )

    def _initSwitchTrainer(self):
        self.switchTrainer = SwitchTrainer(train_mode_switch=self.train_mode_switch)

    def _initCustomTrainer(self):
        self.trainer = CustomTrainer(
            model=self.model,
            optimizers=self.optimizers,
            max_steps=self.max_steps,
            n_epoch=self.n_epoch,
            epoch_len=self.epoch_len,
            retrain=self.retrain,
            elbo_mode=self.elbo_mode,
            track_grad_var=self.track_grad_var,
            n_elbo_particles=self.n_elbo_particles,
            clip_gradients=self.clipGradients,
            max_gradient_norm=self.max_gradient_norm,
        )
