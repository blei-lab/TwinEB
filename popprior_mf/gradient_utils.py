#   Utilities for monitoring gradient variance
#   @Authors: De-identified Author

import numpy as np
import torch


class GradientVarianceMonitor:
    @staticmethod
    def update(existing_aggregate, new_value):
        # Welfard's algorithm for computing a running mean and variance
        (count, mean, M2) = existing_aggregate
        count += 1
        delta = new_value - mean
        mean += delta / count
        delta2 = new_value - mean
        M2 += delta * delta2
        return (count, mean, M2)

    # Retrieve the mean, variance and sample variance from an aggregate
    @staticmethod
    def finalize(existing_aggregate):
        (count, mean, M2) = existing_aggregate
        if count < 2:
            return float("nan")
        else:
            (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
            return (mean, variance, sample_variance)

    # Just the mean
    @staticmethod
    def udpate_mean(existing_aggregate, new_value):
        (count, mean) = existing_aggregate
        count += 1
        delta = new_value - mean
        mean += delta / count
        return (count, mean)

    @staticmethod
    def finalize_mean(existing_aggregate):
        (count, mean) = existing_aggregate
        if count < 2:
            return float("nan")
        else:
            return mean



class GradientTracer():

    def __init__(self, model, optimizers, n_epoch, epoch_len, max_steps=None, n_particles=1000):
        """
        Parameters:
        ------------
        n_particles: num of samples to evaluate the ELBO and its gradient at.
        """
        self.model = model
        self.optimizers = optimizers
        self.epoch_len = epoch_len
        self.n_epoch = n_epoch
        self.n_particles = n_particles
        self.max_steps = max_steps


    def grab_params(self):
        return [
            self.model.row_distribution.mean_parameters(),
            self.model.row_distribution.variance_parameters(),
            self.model.column_distribution.mean_parameters(),
            self.model.column_distribution.variance_parameters(),
        ]


    def param_index_set(self, all_params):
        """
        For each fo the params in grab_params, return the index set
        """
        index_set = []
        last_index = 0
        for i, p in enumerate(all_params):
            # set the gradient for each model parameter
            index_set.append( (last_index, last_index + p.numel()) )
            last_index += p.numel()
        return index_set


    def _get_grad_checkpoints(self):
        return np.unique(
            self.max_steps * np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
            #self.max_steps * np.array([0.01, 0.1])
        ).astype(int)


    def __call__(
        self,
        summary_writer,
        step,
        epoch,
        datapoints_indices,
        x_train,
        holdout_mask,
        do_step=False,
        verbose=False
    ):
        """
        Computes and tracks the mean and variance of gradients in an online fashion.

        Will use the mean to update the gradients of the model.
        Will save the variance to tensorboard.
        NB: Will zero the gradients at the end if do_step is False

        By default, tracks a model specific designation of mean and variance parameters of the variational family.
        TODO: add tracing for the prior params

        TODO: move all of this to a gradienttracker class
        On specific epochs (use percentages of the total number of epochs), compute the gradient variance, on the rest use do not, and just use S = 10
        Every 10% of the epochs, compute the gradient variance, compute using 10% of n_epochs
        1. compute a list of epochs to compute the gradient variance
        2. if the epoch is in this list, compute the gradient variance
        """
        def printv(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        grad_steps = self._get_grad_checkpoints()
        true_step = step + epoch * self.epoch_len
        if true_step not in (grad_steps):
            return None
        
        
        # compute the backward multiple times
        #     # assert that the number of particles is 1
        assert (
            self.model.num_samples == 1
        ), f"Number of particles must be 1 for calculating the grad variance, but is {self.model.num_samples}"
        # TODO: also report the average norm2 of the gradient
        aggregate = (0, 0, 0)
        aggregate_norms = (0, 0, 0)
        all_params = self.grab_params()
        index_set = self.param_index_set(all_params)
        the_names = ['row_location', 'row_scale', 'col_location', 'col_scale']

        # Debug: clone the params of the model before
        # all_params_before = torch.cat([p.flatten().detach() for p in model.parameters()])
        aggregate_mean_elbo = (0, 0)
        for s in range(self.n_particles):
            if s % 100 == 0:
                print(f"Computing gradient variance, sample {s}")
            elbo = self.model(
                datapoints_indices,
                x_train,
                holdout_mask,
                step + epoch * self.epoch_len,
                self.n_epoch * self.epoch_len,
            )
            loss = -elbo
            loss.backward(retain_graph=True)
            with torch.no_grad():
                # # collect all parameter variances in a one dimensional array
                # all_grads = torch.cat([p.grad.flatten() for p in model.parameters()])
                # all_grads = torch.cat([p.grad.flatten() for name, p in model.named_parameters()])
                # all_names = [name for name, p in model.named_parameters()]
                # aggregate = GradientVarianceMonitor.update(aggregate, all_grads)

                # Collect the parameters in order
                all_params = self.grab_params()
                all_grads = torch.cat([p.grad.flatten() for p in all_params])
                aggregate = GradientVarianceMonitor.update(aggregate, all_grads)
                # Compute the norm of the gradient (one fro each param group separetely)
                norms = torch.empty(len(the_names))
                for i, name in enumerate(the_names):        
                    norms[i] = torch.linalg.vector_norm(all_grads[index_set[i][0]:index_set[i][1]])
                aggregate_norms = GradientVarianceMonitor.update(aggregate_norms, norms)
                aggregate_mean_elbo = GradientVarianceMonitor.udpate_mean(aggregate_mean_elbo, elbo.item())


        # Check that the params of the model have not changed
        # all_params_after = torch.cat([p.flatten().detach() for p in model.parameters()])
        # assert torch.all(all_params_before == all_params_after), "The parameters of the model have changed during the gradient variance computation."


        # Compute the mean and variance and sample variance of the gradients
        with torch.no_grad():
            grad_mean, grad_var, grad_sample_var = GradientVarianceMonitor.finalize(
                aggregate
            )

            # Compute the mean of the norms (each is a 4 dim array, one dim per param group)
            grad_norm_mean, grad_norm_var, grad_norm_sample_var = GradientVarianceMonitor.finalize(
                aggregate_norms
            )

            aggregate_mean_elbo = GradientVarianceMonitor.finalize_mean(aggregate_mean_elbo)

            summary_writer.add_scalar(
                "grad_var/grad_mean_elbo", aggregate_mean_elbo, step + epoch * self.epoch_len
            )

            # Compute the trace of the variance-covariance matrix of the gradients
            grad_trace = np.sum(grad_sample_var.detach().cpu().numpy())

            print(f"grad_trace: {grad_trace}")

            # record the sample_variance_g to tensorboard
            summary_writer.add_scalar(
                "grad_var/grad_trace", grad_trace, step + epoch * self.epoch_len
            )

            # Compute the mean for the mean-params and the mean for the variance-params
            all_params = self.grab_params()
            index_set = self.param_index_set(all_params)
            
            for i, name in enumerate(the_names):        
                grad_trace = np.sum(grad_sample_var[index_set[i][0]:index_set[i][1]].detach().cpu().numpy())
                summary_writer.add_scalar(f"grad_var/grad_{name}", grad_trace, step + epoch * self.epoch_len)

            # Record the average and variance mean for the mean and var params
            for i, name in enumerate(the_names):        
                summary_writer.add_scalar(f"grad_var/grad_norm_mean_{name}", grad_norm_mean[i].detach().cpu().numpy(), step + epoch * self.epoch_len)
                summary_writer.add_scalar(f"grad_var/grad_norm_var_{name}", grad_norm_sample_var[i].detach().cpu().numpy(), step + epoch * self.epoch_len)


        if do_step:
            # now manually set the gradient of each param as the mean of the gradients
            with torch.no_grad():
                last_index = 0
                for i, p in enumerate(self.model.parameters()):
                    # set the gradient for each model parameter
                    p.grad = grad_mean[last_index : last_index + p.numel()].reshape(p.shape)
                    last_index += p.numel()

            use_gradient_clipping = False
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=2.0, norm_type=2
                )

            for optim in self.optimizers:
                optim.step()
        else:
            # zero the gradients
            for p in self.model.parameters():
                p.grad = None
            loss = None

        return loss
