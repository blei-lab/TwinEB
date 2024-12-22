"""
    The Probabilistic Principal Component Analysis (PPCA) using variational inference
    
    Based on the codebase of the following paper:
    L. Zhang, Y. Wang, A. Ostropolets, J.J. Mulgrave, D.M. Blei, and G. Hripcsak. 
    The Medical Deconfounder: Assessing Treatment Effects with Electronic Health Records. 
    In Proceedings of the 4th Machine Learning for Healthcare Conference, volume 106 of Proceedings of Machine Learning Research, pages 490-512, Ann Arbor, Michigan, 2019
    https://github.com/zhangly811/Deconfounder/blob/master/inst/python/main.py
"""

import numpy as np
import torch
import tqdm
from scipy import stats
import os
from model_skeleton import ObservedDistribution
from model_variational_family_natural import VariationalFamilyNatural
from tqdm import tqdm

from utils import (
    sparse_tensor_from_sparse_matrix,
    streaming_logsumexp,
    find_diff,
    get_missing_index,
)


class PPCANatural(ObservedDistribution):
    """Implementatoin of Probabilistic Principal Component Analysis (PPCA) class."""

    is_log_space = True

    def __init__(
        self,
        device,
        num_datapoints,
        data_dim,
        latent_dim,
        stddv_datapoints,
        num_samples,
        print_steps,
        summary_writer,
        init_row_loc=None,
        init_col_loc=None,
        annealing_factor=1.0,
        row_prior_scale=1.0,
        column_prior_scale=1.0,
        prior_family=None,
        var_fam_scale=None,
        var_fam_init_scale=None,
        var_family="normal",
        regularize_prior=False,
    ):
        """
        Args:
            device: The device to run the model on.
            num_datapoints: The number of datapoints in the dataset (rows)
            data_dim: The dimension of the data (columns)
            latent_dim: The dimension of the latent space
            stddv_datapoints: The standard deviation of the data points
            num_samples: The number of samples to use for the Monte Carlo approximation of the ELBO
            print_steps: The number of steps between logging to tensorboard
            summary_writer: The tensorboard summary writer
            annealing_factor: The annealing factor for the KL coefficient
        """
        super(PPCANatural, self).__init__()
        self.num_datapoints = num_datapoints
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.stddv_datapoints = stddv_datapoints
        self.num_samples = num_samples
        self.annealing_factor = annealing_factor
        self.print_steps = print_steps
        self.summary_writer = summary_writer
        self.device = device
        assert var_family == "normal", f"Unrecognized var_family {var_family}"
        self.var_family = var_family
        self.regularize_prior = regularize_prior
        self._setup_row_vars(
            init_loc=init_row_loc,
            scale=row_prior_scale,
            prior_family=prior_family,
            fixed_scale=var_fam_scale,
            init_scale=var_fam_init_scale,
            family=self.var_family,
        )
        self._setup_column_vars(
            init_loc=init_col_loc,
            scale=column_prior_scale,
            prior_family=prior_family,
            fixed_scale=var_fam_scale,
            init_scale=var_fam_init_scale,
            family=self.var_family,
        )

    def _setup_row_vars(
        self,
        init_loc,
        scale,
        prior_family=None,
        fixed_scale=None,
        init_scale=None,
        family=None,
    ):
        self.qz_distribution = VariationalFamilyNatural(
            device=self.device,
            family=family,
            shape=[self.latent_dim, self.num_datapoints],
            batch_aware=True,
            initial_loc=init_loc,
            prior_scale=scale,
        )

    def _setup_column_vars(
        self,
        init_loc,
        scale,
        prior_family=None,
        fixed_scale=None,
        init_scale=None,
        family=None,
    ):
        self.qw_distribution = VariationalFamilyNatural(
            device=self.device,
            family=family,
            shape=[self.data_dim, self.latent_dim],
            batch_aware=False,
            initial_loc=init_loc,
            prior_scale=scale,
        )

    @property
    def row_distribution(self):
        return self.qz_distribution

    @property
    def column_distribution(self):
        return self.qw_distribution

    @property
    def row_mean(self):
        # return self.row_distribution.location.cpu().detach().numpy()
        the_loc = self.row_distribution.location.cpu().detach().numpy()
        if self.var_family == 'normal':
            return the_loc
        else:
            raise ValueError(f'Unrecognized var_family {self.var_family}')

    @property
    def column_mean(self):
        # return self.column_distribution.location.cpu().detach().numpy()
        the_loc = self.column_distribution.location.cpu().detach().numpy()
        if self.var_family == 'normal':
            return the_loc
        else:
            raise ValueError(f'Unrecognized var_family {self.var_family}')

    def init_column_vars(self, w_init):
        # self.qw_distribution.location = torch.nn.Parameter(torch.DoubleTensor(w_init))
        self.qw_distribution.location.data = torch.DoubleTensor(w_init).to(self.device)

    def init_row_vars(self, z_init):
        # self.qz_distribution.location = torch.nn.Parameter(torch.DoubleTensor(z_init))
        self.qz_distribution.location.data = torch.DoubleTensor(z_init).to(self.device)

    def distribution(self, loc, scale):
        """The likelihoood distribution of the model.
        Args:
            loc: location param of the distribution
            scale: scale param of the distribution
        Returns:
            A torch.distribution object with location and scale set.
        """
        # print the device of each loc and scale
        distribution = torch.distributions.Normal(loc=loc, scale=scale)
        return distribution

    def get_samples(self, datapoints_indices):
        """
        Return samples from variational distributions.

        Args:
            datapoints_indices: The indices of the datapoints in the minibatch.

        Returns:
            A list of tensors of shape [num_samples, latent_dim, minibatch_size]
        """
        w_samples = self.qw_distribution.sample(self.num_samples)
        z_samples = self.qz_distribution.sample(self.num_samples)
        z_samples = z_samples[:, :, datapoints_indices]

        samples = [w_samples, z_samples]
        return samples

    def get_log_prior(self, samples):
        """Calculate log prior of variational samples.

        Args:
          samples: A list of samples. The length of the list is the number of variables being sampled.
                   w_sampels: [num_samples, data_dim, latent_dim]
                   z_samples: [num_samples, latent_dim, num_datapoints]

        Returns:
          log_prior: A Monte-Carlo approximation of the log prior, summed across
            latent dimensions and averaged over the number of samples.
        """
        (w_samples, z_samples) = samples
        w_log_prior = self.qw_distribution.get_log_prior(w_samples)

        z_log_prior = self.qz_distribution.get_log_prior(z_samples)

        # Adjust for minibatch
        batch_size = z_samples.shape[-1]
        z_log_prior = z_log_prior * (self.num_datapoints / batch_size)
        log_prior = w_log_prior + z_log_prior

        # Average over number of particles
        return torch.mean(log_prior)

    def get_entropy(self, samples, datapoints_indices):
        """Calculate entropy of variational samples.

        Args:
            samples: A list of samples. The length of the list is the number of variables being sampled.
                     w_sampels: [num_samples, data_dim, latent_dim]
                     z_samples: [num_samples, latent_dim, batch_size]

        Returns:
          log_prior: A Monte-Carlo approximation of the variational entropy,
            summed across latent dimensions and averaged over the number of
            samples.
        """
        (w_samples, z_samples) = samples
        w_entropy = self.qw_distribution.get_entropy(w_samples)
        z_entropy = self.qz_distribution.get_entropy(z_samples, datapoints_indices)

        # Adjust for minibatch
        batch_size = z_samples.shape[-1]
        z_entropy = z_entropy * (self.num_datapoints / batch_size)
        entropy = w_entropy + z_entropy
        return torch.mean(entropy)

    def generate_data(self, n_samples=1, x=None, datapoints_indices=None):
        """
        Generate observations from the model sequentially.
        Args:
            n_samples: The number of samples to generate.
            TODO:
            x: For amortized models, the data to use for encoding the latent variables.
            datapoints_indices: For amortized models, the indices of the datapoints in the minibatch.
        Returns:
            A list of tensors of shape [n_samples, num_datapoints, data_dim]
        """
        obs = []
        for _ in range(n_samples):
            w_samples = self.qw_distribution.sample(1)
            z_samples = self.qz_distribution.sample(1)
            mean_sample = torch.transpose(
                torch.squeeze(torch.matmul(w_samples, z_samples)), 0, 1
            )
            x_gen = self.distribution(
                loc=mean_sample, scale=self.stddv_datapoints
            ).sample()
            obs.append(x_gen)
        return obs

    def get_data_log_likelihood(self, samples, counts, holdout_mask):
        """Approximate log-likelihood term of ELBO using Monte Carlo samples.

        Args:
          samples: A list of samples. The length of the list is the number of variables being sampled.
                   w_sampels: [num_samples, data_dim, latent_dim]
                   z_samples: [num_samples, latent_dim, batch_size]
          counts: A float-tensor with shape [batch_size, num_words].
          holdout_mask: A binary tensor with shape [batch_size, data_dim]. 1=valid, 0=train

        Returns:
          data_log_likelihood: A Monte-Carlo approximation of the count
            log-likelihood, summed across latent dimensions and averaged over the
            number of samples, adjusted for minibatch.

        """
        # At this point, z is already subsampled
        (w_samples, z_samples) = samples
        loc_datapoints = torch.transpose(torch.matmul(w_samples, z_samples), 1, 2)
        data_distribution = self.distribution(
            loc=loc_datapoints, scale=self.stddv_datapoints
        )
        # data_log_likelihood is [num_samples, batch_size, data_dim]
        dense_counts = counts.to_dense()
        data_log_likelihood = data_distribution.log_prob(dense_counts)

        # Sum over the indiv data points and the data dimension (only keeping the train entries)
        # Two level masking
        # 1. Zero out the contribution of mask out entries
        # 2. Zero out the contribution of zero values
        m1 = torch.mul(data_log_likelihood, 1.0 - holdout_mask.to_dense())
        # form a binary matrix where every non zero element of counts is 1 and leave zeros as is
        counts_mask = torch.zeros_like(dense_counts)
        # set non zero indexes of dense_counts to 1 in counts_mask
        counts_mask[dense_counts != 0] = 1
        m2 = torch.mul(m1, counts_mask[None, :, :])
        data_log_likelihood = torch.sum(
            m2, dim=tuple(range(1, data_log_likelihood.dim()))
        )
        # data_log_likelihood = torch.sum(m1, dim=tuple(range(1, data_log_likelihood.dim())))

        # Adjust for minibatch
        batch_size = counts.shape[0]
        data_log_likelihood = data_log_likelihood * (self.num_datapoints / batch_size)
        return torch.mean(data_log_likelihood)

    def monitor_elbo(
        self, step, elbo, entropy, log_prior, data_log_likelihood, neg_kl_divergence
    ):
        """Write elbo and parts"""
        if step % self.print_steps == 0:
            self.summary_writer.add_scalar("elbo/entropy", entropy, step)
            self.summary_writer.add_scalar("elbo/log_prior", log_prior, step)
            self.summary_writer.add_scalar("elbo/kl_div", -neg_kl_divergence, step)
            self.summary_writer.add_scalar(
                "elbo/data_log_likelihood", data_log_likelihood, step
            )
            self.summary_writer.add_scalar("elbo/elbo", elbo, step)

    def forward(self, datapoints_indices, data, holdout_mask, step, max_steps=None):
        """Approximate variational Lognormal ELBO using reparameterization.

        Args:
            datapoints_indices: An int-vector with shape [batch_size].
            data: A matrix with shape `[batch_size, data_dim]`.
            step: The training step, used to log summaries to Tensorboard.

        Returns:
            elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value
              is averaged across samples and summed across batches.
        """
        samples = self.get_samples(datapoints_indices)
        log_prior = self.get_log_prior(samples)
        data_log_likelihood = self.get_data_log_likelihood(samples, data, holdout_mask)
        entropy = self.get_entropy(samples, datapoints_indices)
        neg_kl_divergence = log_prior + entropy
        # elbo = (
        #     data_log_likelihood
        #     + neg_kl_divergence * self.annealing_factor * (step / max_steps) ** 2
        # )
        elbo = data_log_likelihood + neg_kl_divergence * self.annealing_factor
        # elbo = data_log_likelihood + neg_kl_divergence
        # print(step/max_steps)
        self.monitor_elbo(
            step, elbo, entropy, log_prior, data_log_likelihood, neg_kl_divergence
        )
        return elbo

    def predictive_check(self, x_vad, holdout_mask, holdout_subjects, n_rep, n_eval):
        """Predictive model checking.

        Args:
            x_vad: The validation data matrix with shape `[num_datapoints, data_dim]`.
            holdout_mask: A binary tensor to mask the train/validation data with shape [num_datapoints, data_dim]. 1=valid, 0=train
            n_rep: The number of replicated datasets we generate.
            n_eval: The number of samples drawn from the variational posterior
        Returns:
            elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value
              is averaged across samples and summed across batches.
        """
        holdout_gen = np.zeros((n_rep, self.num_datapoints, self.data_dim))

        for i in tqdm(range(n_rep)):
            w_samples = self.qw_distribution.sample(1)
            z_samples = self.qz_distribution.sample(1)
            loc = torch.squeeze(
                torch.transpose(torch.matmul(w_samples, z_samples), 1, 2)
            )
            data_posterior = self.distribution(loc=loc, scale=self.stddv_datapoints)
            x_generated = data_posterior.sample(torch.Size([1]))
            x_generated = torch.squeeze(x_generated)
            # look only at the heldout entries
            holdout_gen[i] = np.multiply(x_generated, holdout_mask)

        obs_ll = []
        rep_ll = []
        for j in tqdm(range(n_eval)):
            w_samples = self.qw_distribution.sample(1).detach().numpy()
            z_samples = self.qz_distribution.sample(1).detach().numpy()

            holdoutmean_sample = np.transpose(np.squeeze(w_samples.dot(z_samples)))
            holdoutmean_sample = np.multiply(holdoutmean_sample, holdout_mask)
            obs_ll.append(
                np.mean(
                    stats.norm(holdoutmean_sample, self.stddv_datapoints).logpdf(x_vad),
                    axis=1,
                )
            )

            rep_ll.append(
                np.mean(
                    stats.norm(holdoutmean_sample, self.stddv_datapoints).logpdf(
                        holdout_gen
                    ),
                    axis=2,
                )
            )

        obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(
            np.array(rep_ll), axis=0
        )

        pvals = np.array(
            [
                np.mean(rep_ll_per_zi[:, i] < obs_ll_per_zi[i])
                for i in range(self.num_datapoints)
            ]
        )
        overall_pval = np.mean(pvals[holdout_subjects])
        print("Predictive check p-values", overall_pval)
        self.summary_writer.add_scalar("predictive check p-values", overall_pval)

    def generate_holdout_mean(self, holdout_mask):
        # if holdout_mask is not a torch sparse tensor, convert it to one
        if not isinstance(holdout_mask, torch.sparse.FloatTensor):
            holdout_mask = sparse_tensor_from_sparse_matrix(holdout_mask)

        u = holdout_mask.coalesce().indices()
        w = self.qw_distribution.sample(1).squeeze()
        z = self.qz_distribution.sample(1).squeeze().T
        z_rows = z[u[0]]  # shape: (P, L)
        w_rows = w[u[1]]  # shape: (P, L)
        prod = torch.sum(z_rows * w_rows, dim=1)  # shape: (P,)
        return prod, u

    def generate_heldout_data(self, holdout_mask):
        prod, u = self.generate_holdout_mean(holdout_mask)
        # sample from the data distribution
        data_posterior = self.distribution(loc=prod, scale=self.stddv_datapoints)
        x_generated = data_posterior.sample(torch.Size([1])).squeeze()
        # convert to sparse matrix
        return torch.sparse.FloatTensor(u, x_generated, holdout_mask.shape)

    # def compute_llhood_sparse_old(self, x_vad, holdout_mask, missing_indexes=None, w=None, z=None):
    #     """
    #     Compute llhood with sparse tensors.

    #     Args:
    #         x_vad: The validation data matrix with shape `[num_datapoints, data_dim]`.
    #         holdout_mask: A binary tensor to mask the train/validation data with shape [num_datapoints, data_dim]. 1=valid, 0=train

    #     Returns:
    #         log_ll: The log-likelihood of the validation data, of shape [num_datapoints]
    #     """
    #     # relationship of x_vad and holdout_mask: x_vad = x.multiply(self.holdout_mask.A)
    #     def inner_llhood(w, z, indx, vals):
    #         # If the index or vals are in cpu, cast u and v to cpu too
    #         if indx.device.type == "cpu":
    #             z = z.cpu()
    #             w = w.cpu()
    #             vals = vals.cpu()
    #         # Extract the required rows from z and w using advanced indexing
    #         z_rows = z[indx[0]] # shape: (P, L)
    #         w_rows = w[indx[1]] # shape: (P, L)
    #         prod = torch.sum(z_rows * w_rows, dim=1) # shape: (P,)
    #         return self.distribution(loc=prod, scale=self.stddv_datapoints).log_prob(vals)

    #     # set torch seed
    #     if w is None or z is None:
    #         w = self.qw_distribution.sample(1).squeeze()
    #         z = self.qz_distribution.sample(1).squeeze().T
    #     else:
    #         w = w.squeeze()
    #         z = z.squeeze().T

    #     # Compute for the observed indices
    #     indx = x_vad.coalesce().indices()
    #     vals = x_vad.coalesce().values()
    #     log_ll_obs = inner_llhood(w, z, indx, vals)

    #     # Compute for the missing indices
    #     indx1 = holdout_mask.coalesce().indices()
    #     if indx1.shape[1] > indx.shape[1]:
    #         m_indx = missing_indexes if missing_indexes is not None else find_diff(indx1, indx)
    #         assert m_indx.shape[1] == indx1.shape[1] - indx.shape[1]
    #         vals = torch.zeros(m_indx.shape[1]).to(self.device)
    #         log_ll_miss = inner_llhood(w, z, m_indx, vals)
    #         # Create a sparse tensor for the observed and missing indices
    #         log_ll = torch.sparse.FloatTensor(
    #             torch.cat([indx, m_indx], dim=1),
    #             torch.cat([log_ll_obs, log_ll_miss]),
    #             holdout_mask.shape
    #         )
    #     else:
    #         log_ll = torch.sparse.FloatTensor(indx, log_ll_obs, holdout_mask.shape)

    #     if self.__class__.is_log_space:
    #         log_ll = log_ll - x_vad
    #     # compute colsum
    #     log_ll = torch.sparse.sum(log_ll, dim=1).to_dense()
    #     return log_ll

    def compute_llhood_sparse(self, x_vad, missing_indexes=None, w=None, z=None):
        """
        Compute llhood with sparse tensors.

        Args:
            x_vad: The validation data matrix with shape `[num_datapoints, data_dim]`.
            holdout_mask: A binary tensor to mask the train/validation data with shape [num_datapoints, data_dim]. 1=valid, 0=train

        Returns:
            log_ll: The log-likelihood of the validation data, of shape [num_datapoints]
        """

        # relationship of x_vad and holdout_mask: x_vad = x.multiply(self.holdout_mask.A)
        def inner_llhood(w, z, indx, vals):
            # If the index or vals are in cpu, cast u and v to cpu too
            if indx.device.type == "cpu" and z.device.type != "cpu":
                z = z.cpu()
                w = w.cpu()
                vals = vals.cpu()
            # Extract the required rows from z and w using advanced indexing
            z_rows = z[indx[0]]  # shape: (P, L)
            w_rows = w[indx[1]]  # shape: (P, L)
            prod = torch.sum(z_rows * w_rows, dim=1)  # shape: (P,)
            return self.distribution(loc=prod, scale=self.stddv_datapoints).log_prob(
                vals
            )

        # set torch seed
        if w is None or z is None:
            w = self.qw_distribution.sample(1).squeeze()
            z = self.qz_distribution.sample(1).squeeze().T
        else:
            w = w.squeeze()
            z = z.squeeze().T

        # Compute for the observed indices
        indx = x_vad.coalesce().indices()
        vals = x_vad.coalesce().values()
        log_ll_obs = inner_llhood(w, z, indx, vals)

        # Compute for the missing indices
        if missing_indexes is not None and missing_indexes.shape[1] > 0:
            vals = torch.zeros(missing_indexes.shape[1]).to(self.device)
            log_ll_miss = inner_llhood(w, z, missing_indexes, vals)
            # Create a sparse tensor for the observed and missing indices
            log_ll = torch.cat((log_ll_obs, log_ll_miss), dim=0)
        else:
            log_ll = log_ll_obs

        if self.__class__.is_log_space:
            obs_len = indx.shape[1]
            log_ll[:obs_len] = log_ll[:obs_len] - x_vad.coalesce().values()

        return log_ll

    @torch.no_grad()
    def compute_mae_sparse(self, x_vad, w=None, z=None):
        """
        Compute MAE with sparse tensors.

        Args:
            x_vad: The validation data matrix with shape `[num_datapoints, data_dim]`.
            holdout_mask: A binary tensor to mask the train/validation data with shape [num_datapoints, data_dim]. 1=valid, 0=train

        Returns:
            log_ll: The log-likelihood of the validation data, of shape [num_datapoints]
        """

        # relationship of x_vad and holdout_mask: x_vad = x.multiply(self.holdout_mask.A)
        def inner_mae(w, z, indx, vals):
            # If the index or vals are in cpu, cast u and v to cpu too
            if indx.device.type == "cpu" and z.device.type != "cpu":
                z = z.cpu()
                w = w.cpu()
                vals = vals.cpu()
            # Extract the required rows from z and w using advanced indexing
            z_rows = z[indx[0]]  # shape: (P, L)
            w_rows = w[indx[1]]  # shape: (P, L)
            prod = torch.sum(z_rows * w_rows, dim=1)  # shape: (P,)
            # prod is the mean for the data, it is the E[X]
            mae = torch.mean(torch.abs(vals - prod))
            rmse = torch.sqrt(torch.mean((vals - prod) ** 2))

            xx = torch.abs(vals - prod)
            import pandas as pd

            tmp_diff = torch.abs(vals - prod)
            tmp_df = pd.DataFrame(
                {
                    "true": vals.cpu().numpy(),
                    "pred": prod.cpu().numpy(),
                    "diff": tmp_diff.cpu().numpy(),
                }
            )
            # sort by tmp_diff descending
            tmp_df = tmp_df.sort_values(by=["diff"], ascending=False)
            # print the top 10
            print(tmp_df.head(10))

            return mae, rmse

        # set torch seed
        if w is None or z is None:
            # w = self.qw_distribution.sample(1).squeeze()
            # z = self.qz_distribution.sample(1).squeeze().T
            # No need to sample, just return the mean
            w = self.qw_distribution.location.squeeze()
            z = self.qz_distribution.location.squeeze().T
        else:
            w = w.squeeze()
            z = z.squeeze().T

        # Compute for the observed indices
        indx = x_vad.coalesce().indices()
        vals = x_vad.coalesce().values()
        mae, rmse = inner_mae(w, z, indx, vals)

        def test_mae_dense(x_vad):
            # generate data from the model
            w_samples = self.qw_distribution.sample(1)
            z_samples = self.qz_distribution.sample(1)
            mean_sample = torch.transpose(
                torch.squeeze(torch.matmul(w_samples, z_samples)), 0, 1
            )
            x_gen = self.distribution(
                loc=mean_sample, scale=self.stddv_datapoints
            ).sample()

            # extract indices
            indx = x_vad.coalesce().indices()
            vals = x_vad.coalesce().values()
            x_gen_vals = x_gen[indx[0], indx[1]]
            mae = torch.mean(torch.abs(vals - x_gen_vals))
            rmse = torch.sqrt(torch.mean((vals - x_gen_vals) ** 2))
            # print these values for all zeros
            print("MAE trivial", torch.mean(torch.abs(vals)).cpu().numpy())
            print("RMSE trivial", torch.sqrt(torch.mean(vals**2)).cpu().numpy())

            return mae, rmse

        mae2, rmse2 = test_mae_dense(x_vad)
        print("MAE2", mae2.cpu().numpy(), rmse2.cpu().numpy())

        return mae, rmse

    def compute_llhood(self, x_vad, holdout_mask, w_samples=None, z_samples=None):
        """
        Compute loglikelihood for the given data, taking the mask into account.
        logP(X_vad | Z@W)

        Args:
            x_vad: A matrix with shape [num_datapoints, data_dim].
            holdout_mask: A binary tensor with shape [num_datapoints, data_dim]. 1=valid, 0=train

        Returns:
            log_ll: A tensor of shape [num_datapoints]
        """
        if w_samples is None or z_samples is None:
            w_samples = self.qw_distribution.sample(1)
            z_samples = self.qz_distribution.sample(1)
        holdoutmean_sample = torch.transpose(
            torch.squeeze(torch.matmul(w_samples, z_samples)), 0, 1
        )
        holdoutmean_sample = holdoutmean_sample * holdout_mask
        # log_ll is [num_datapoints, data_dim]
        log_ll = self.distribution(
            loc=holdoutmean_sample, scale=self.stddv_datapoints
        ).log_prob(x_vad)
        # Add the adjustment for using log data instead of count data
        if self.__class__.is_log_space:
            log_ll = log_ll - x_vad
        # Remove the heldin llhoods
        log_ll = log_ll * holdout_mask
        log_ll = torch.sum(log_ll, dim=1)
        return log_ll

    def quick_test(self, x_vad, holdout_mask):
        """Ensure the sparse and dense version are the same."""
        # log_ll = self.compute_llhood(x_vad, holdout_mask)
        w = self.qw_distribution.sample(1)
        z = self.qz_distribution.sample(1)
        log_ll = self.compute_llhood(x_vad.to_dense(), holdout_mask.to_dense(), w, z)
        log_ll2 = self.compute_llhood_sparse(x_vad, holdout_mask, w, z)
        assert (
            torch.sum(torch.abs(log_ll2 - log_ll)) < 1e-8
        ), "log_ll2 is not the same as log_ll"

    
    @torch.no_grad()
    def compute_heldout_loglikelihood_none_zero(self, x_vad, n_monte_carlo=100, use_mean=False, write_llhood=True):
        """Computes the heldout loglikelihood for the non-zero entries only."""
        n_obs = x_vad.coalesce().indices().shape[1]
        r, alpha = 0.0, 0.0
        if use_mean:
            norm_constant = 0.0
            #print('Warning: Using mean instead of sampling...')
            # Extract the mean of each distribution
            w = self.qw_distribution.location
            z = self.qz_distribution.location
            log_ll = self.compute_llhood_sparse(x_vad, missing_indexes=None, w=w, z=z)
            alpha = torch.min(log_ll) - 1.0
            r, alpha = streaming_logsumexp(log_ll, r, alpha)
        else:
            # Add the -log(M) term
            norm_constant = torch.log(torch.tensor(n_monte_carlo, dtype=torch.float))
            for j in tqdm(range(n_monte_carlo)):
                log_ll = self.compute_llhood_sparse(x_vad, missing_indexes=None)
                if j == 0:
                    alpha = torch.min(log_ll) - 1.0
                # Compute the running logsumexp per datapoint (over datasets)
                r, alpha = streaming_logsumexp(log_ll, r, alpha)

        heldout_llhood = torch.log(r) + alpha
        # Handle the norm_constant
        heldout_llhood = heldout_llhood - norm_constant

        llobs_ = torch.sum(heldout_llhood[:n_obs])
        heldout_llhood = llobs_

        # normalizing by the number of heldout values
        heldout_llhood = heldout_llhood / (n_obs)

        if write_llhood:
            self.summary_writer.add_scalar(
                f"llhood/heldout_llhood{x_vad.shape[0]}", heldout_llhood
            )
        return heldout_llhood

    @torch.no_grad()
    def compute_heldout_loglikelihood(
        self,
        x_vad,
        missing_indexes,
        n_monte_carlo=100,
        subsample_zeros=True,
        ignore_zeros=True,
        use_mean=False,
        write_llhood=True
    ):
        """
        Compute the heldout loglikelihood.
        NB: Ignores all zeros in the x_vad by default!
        Args:
        ----------
            x_vad: A matrix with shape [num_datapoints, data_dim].
            holdout_mask: A binary tensor with shape [num_datapoints, data_dim]. 1=valid, 0=train
            holdout_subjects: A list of subject indices.
            n_monte_carlo: The number of Monte-Carlo samples to use (i.e., how many datasets to generate from the posteior predictive)
            use_mean: If True, use the mean of the variational posterior instead of sampling from it.

        Returns:
            heldout_loglikelihood: A scalar representing the heldout loglikelihood.

        Approximate with M (n_monte_carlo) param monte carlo samples
        \frac{1}{M} \sum_{m}^{M}\sum_{i}^{N} \log(P(X_i \mid Z_m))

        Note: This is using streaming logsumexp to save memory.
        """

        def subsample_(arry, n_samples):
            indx = np.random.choice(arry.shape[1], n_samples, replace=False)
            return arry[:, indx]

        if n_monte_carlo == 1:
            # print("Warning: n_monte_carlo is 1, using mean instead of sampling...")
            use_mean = True

        if ignore_zeros:
            #print("Ignoring zeros in the computation of heldout llhood...")
            return self.compute_heldout_loglikelihood_none_zero(
                x_vad, n_monte_carlo=n_monte_carlo, use_mean=use_mean, write_llhood=write_llhood
            )

        n_obs = x_vad.coalesce().indices().shape[1]
        n_missing = missing_indexes.shape[1]

        n_subsample = np.minimum(n_obs * 1, n_missing)
        r, alpha = 0.0, 0.0

        if subsample_zeros:
            print("Subsampling missing indx for computationl expediency...")
            print(
                f"Keeping {n_subsample} from {missing_indexes.shape[1]} or {n_subsample/missing_indexes.shape[1]:.2f} %"
            )

        b_hat = (
            subsample_(missing_indexes, n_subsample)
            if subsample_zeros
            else missing_indexes
        )
        b_hat = torch.as_tensor(b_hat, dtype=torch.long).to(x_vad.device)
        if use_mean:
            #print('Warning: Using mean instead of sampling...')
            # Extract the mean of each distribution
            w = self.qw_distribution.location
            z = self.qz_distribution.location
            log_ll = self.compute_llhood_sparse(x_vad, missing_indexes=b_hat, w=w, z=z)
            alpha = torch.min(log_ll) - 1.0
            r, alpha = streaming_logsumexp(log_ll, r, alpha)
            
        else:
            for j in tqdm(range(n_monte_carlo)):
                # log_ll = self.compute_llhood(x_vad, holdout_mask)
                # log_ll = self.compute_llhood_sparse(x_vad, holdout_mask, missing_indexes=missing_indexes)
                log_ll = self.compute_llhood_sparse(x_vad, missing_indexes=b_hat)
                if j == 0:
                    alpha = torch.min(log_ll) - 1.0
                # Compute the running logsumexp per datapoint (over datasets)
                r, alpha = streaming_logsumexp(log_ll, r, alpha)

        heldout_llhood = torch.log(r) + alpha
        # Add the -log(M) term
        norm_constant = torch.log(torch.tensor(n_monte_carlo, dtype=torch.float))
        heldout_llhood = heldout_llhood - norm_constant

        llobs_ = torch.sum(heldout_llhood[:n_obs])
        llmiss_ = torch.sum(heldout_llhood[n_obs:])
        # print(f"l_obs = {llobs_:.2f}")
        # print(f"l_miss = {llmiss_:.2f}")
        # print(f"llmiss/llobs = {llmiss_ / llobs_:.2f}")
        if subsample_zeros:
            # Make the adjustment for the subsampling
            # Split the sum as sum(heldout_llhood[x_vad.nonzeros]) + (b_hat.shape[1]/missing_indexes.shape[1]) sum(heldout_llhood[missing_indexes])
            print(
                f"Using correction factor of {n_missing/n_subsample:.2f} for subsampling"
            )
            print(
                "adjusted llmiss = {:.2f}".format(llmiss_ * (n_missing / n_subsample))
            )
            heldout_llhood = llobs_ + (llmiss_ * (n_missing / n_subsample))
        else:
            heldout_llhood = llobs_ + llmiss_

        # normalizing by the number of heldout values
        heldout_llhood = heldout_llhood / (n_obs + n_missing)

        if write_llhood:
            self.summary_writer.add_scalar(
                f"llhood/heldout_llhood{x_vad.shape[0]}", heldout_llhood
            )
        return heldout_llhood

    def save_txt(self, param_save_dir, **kwargs):
        """Save the model params to file

        Args:
            param_save_dir: The validation data matrix with shape `[num_datapoints, data_dim]`.

        Returns:
            None
        """

        def quick_save(torch_dist, name):
            np.savetxt(
                os.path.join(param_save_dir, f"{name}_loc"),
                torch_dist.location.cpu().detach(),
            )
            np.savetxt(
                os.path.join(param_save_dir, f"{name}_scale"),
                torch_dist.scale().cpu().detach(),
            )

        quick_save(self.qw_distribution, "qw")
        quick_save(self.qz_distribution, "qz")
