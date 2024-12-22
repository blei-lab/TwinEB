""""
    The Probablistic Matrix Factorization class 

    Based on the codebase of the following paper:
    L. Zhang, Y. Wang, A. Ostropolets, J.J. Mulgrave, D.M. Blei, and G. Hripcsak. 
    The Medical Deconfounder: Assessing Treatment Effects with Electronic Health Records. 
    In Proceedings of the 4th Machine Learning for Healthcare Conference, volume 106 of Proceedings of Machine Learning Research, pages 490-512, Ann Arbor, Michigan, 2019
    https://github.com/zhangly811/Deconfounder/blob/master/inst/python/main.py
"""


import numpy as np
import torch
from tqdm import tqdm
import os
import yaml
from model_skeleton import ObservedDistribution
from model_variational_family import VariationalFamily
from torch.utils.tensorboard import SummaryWriter

from utils import (
    streaming_logsumexp,
    sparse_tensor_from_sparse_matrix,
    find_diff,
    get_missing_index,
)


class PMF(ObservedDistribution):
    """Object to hold model parameters and approximate ELBO."""

    is_log_space = False

    def __init__(
        self,
        device,
        num_datapoints,
        data_dim,
        latent_dim,
        num_samples,
        print_steps,
        summary_writer,
        init_row_loc=None,
        init_col_loc=None,
        annealing_factor=1.0,
        row_prior_concentration=0.1,
        row_prior_rate=0.3,
        column_prior_concentration=0.1,
        column_prior_rate=0.3,
        prior_family=None,
        var_fam_scale=None,
        var_fam_init_scale=None,
        var_family="lognormal",
    ):
        super(PMF, self).__init__()
        self.device = device
        self.num_datapoints = num_datapoints
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.print_steps = print_steps
        self.summary_writer = summary_writer
        self.annealing_factor = annealing_factor
        self.var_family = var_family

        self._setup_row_vars(
            init_row_loc,
            concentration=row_prior_concentration,
            rate=row_prior_rate,
            prior_family=prior_family,
            fixed_scale=var_fam_scale,
            init_scale=var_fam_init_scale,
            family=self.var_family,
        )
        self._setup_column_vars(
            init_col_loc,
            concentration=column_prior_concentration,
            rate=column_prior_rate,
            prior_family=prior_family,
            fixed_scale=var_fam_scale,
            init_scale=var_fam_init_scale,
            family=self.var_family,
        )

    def _setup_row_vars(
        self,
        init_loc,
        concentration,
        rate,
        prior_family=None,
        fixed_scale=None,
        init_scale=None,
        family=None
    ):
        # Set up row variables
        self.qu_distribution = VariationalFamily(
            device=self.device,
            family=family,
            shape=[self.latent_dim, self.num_datapoints],
            batch_aware=True,
            # kwargs
            # prior_scale=None,
            prior_concentration=concentration,
            prior_rate=rate,
            prior_family=prior_family,
            fixed_scale=fixed_scale,
            init_scale=init_scale,
        )

    def _setup_column_vars(
        self,
        init_loc,
        concentration,
        rate,
        prior_family=None,
        fixed_scale=None,
        init_scale=None,
        family=None
    ):
        # Set up column variables
        self.qv_distribution = VariationalFamily(
            device=self.device,
            family=family,
            shape=[self.data_dim, self.latent_dim],
            batch_aware=False,
            prior_concentration=concentration,
            prior_rate=rate,
            prior_family=prior_family,
            fixed_scale=fixed_scale,
            init_scale=init_scale,
        )

    @staticmethod
    def _match_lognormal_mean(mean, fixed_scales, nugget=1e-5):
        """
        Finds the `location` (mu) of the lognormal distribution  with the given mean, for fixed variances.
        Expected value of the lognormal distribution: exp(mu + sigma^2/2),
        Then its location parameter for a desired mean is:
            mu := log(mean + nugget) - .5*(fixed_scales**2)
        with a small nugget to avoid log(0).
        """
        return np.log(mean + nugget) - 0.5 * (fixed_scales**2)

    @staticmethod
    def _compute_lognormal_mean(location, scale):
        """
        Computes the mean of the lognormal distribution, given its location and scale (s.d).
        Mean of the lognormal: exp(mu + sigma^2/2)
        """
        # assert all elements of scale > 0
        assert np.all(scale > 0), "scale has negative elements"
        return np.exp(location + 0.5 * (scale**2))

    @staticmethod
    def _compute_gamma_mean(concentration, rate):
        """
        Computes the mean of the gamma distribution, given its concentration and rate.
        Mean of the gamma: concentration/rate
        """
        return concentration / rate

    @property
    def row_distribution(self):
        return self.qu_distribution

    @property
    def column_distribution(self):
        return self.qv_distribution


    # add a property that calculates the mean of hte column or row distribution based on the var_family
    @property
    def row_mean(self):
        the_loc = self.row_distribution.location.cpu().detach().numpy()
        the_scale = self.row_distribution.scale()
        the_scale = VariationalFamily.make_positive(the_scale, exponentiate=False).cpu().detach().numpy()
        if self.var_family == 'lognormal':
            return PMF._compute_lognormal_mean(the_loc, the_scale)
        elif self.var_family == 'gamma':
            return PMF._compute_gamma_mean(the_loc, the_scale)
        else:
            raise ValueError(f'Unrecognized var_family {self.var_family}')

    @property
    def column_mean(self):
        the_loc = self.column_distribution.location.cpu().detach().numpy()
        the_scale = self.column_distribution.scale()
        the_scale = VariationalFamily.make_positive(the_scale, exponentiate=False).cpu().detach().numpy()
        if self.var_family == 'lognormal':
            return PMF._compute_lognormal_mean(the_loc, the_scale)
        elif self.var_family == 'gamma':
            return PMF._compute_gamma_mean(the_loc, the_scale)
        else:
            raise ValueError(f'Unrecognized var_family {self.var_family}')

    # def init_column_vars(self, w_init):
    #     self.qv_distribution.location = torch.nn.Parameter(torch.DoubleTensor(w_init))

    # def init_row_vars(self, z_init):
    #     self.qu_distribution.location = torch.nn.Parameter(torch.DoubleTensor(z_init))

    def init_column_vars(self, w_init):
        # match the location to lognormal
        init_means = PMF._match_lognormal_mean(
            w_init, self.column_distribution.scale().cpu().detach().numpy()
        )
        self.qv_distribution.location.data = torch.DoubleTensor(init_means).to(
            self.device
        )
        # assert this is a torch.nn.parameter
        assert isinstance(self.qv_distribution.location, torch.nn.parameter.Parameter)

    def init_row_vars(self, z_init):
        # self.qu_distribution.location.data = torch.nn.Parameter(torch.DoubleTensor(z_init))
        init_means = PMF._match_lognormal_mean(
            z_init, self.row_distribution.scale().cpu().detach().numpy()
        )
        self.qu_distribution.location.data = torch.DoubleTensor(init_means).to(
            self.device
        )
        assert isinstance(self.qu_distribution.location, torch.nn.parameter.Parameter)

    def distribution(self, rate):
        """The likelihoood distribution of the model.
        Args:
            loc: location param of the distribution
        Returns:
            A torch.distribution object with location and scale set.
        """
        distribution = torch.distributions.Poisson(rate=rate)
        return distribution

    def get_samples(self, datapoints_indices):
        """Return samples from variational distributions."""
        v_samples = self.qv_distribution.sample(self.num_samples)
        u_samples = self.qu_distribution.sample(self.num_samples)
        # soft plus the samples
        assert u_samples.min() >= 0.0, "Negative values in u_samples."
        assert v_samples.min() >= 0.0, "Negative values in v_samples."

        # u_samples = torch.clamp_min(u_samples, 1e-6)
        u_samples = u_samples[:, :, datapoints_indices]

        return [v_samples, u_samples]

    def get_log_prior(self, samples):
        """Calculate log prior of variational samples.

        Args:
          samples: A list of samples. The length of the list is the number of variables being sampled.

        Returns:
          log_prior: A Monte-Carlo approximation of the log prior, summed across
            latent dimensions and averaged over the number of samples.
        """
        (v_samples, u_samples) = samples
        # print('qv, get_log_prior')
        v_log_prior = self.qv_distribution.get_log_prior(v_samples)
        u_log_prior = self.qu_distribution.get_log_prior(u_samples)
        # Adjust for minibatch
        batch_size = u_samples.shape[-1]
        u_log_prior = u_log_prior * (self.num_datapoints / batch_size)
        log_prior = v_log_prior + u_log_prior
        return torch.mean(log_prior)

    def get_entropy(self, samples, datapoints_indices):
        """Calculate entropy of variational samples.

        Args:
            samples: A list of samples. The length of the list is the number of variables being sampled.

        Returns:
          log_prior: A Monte-Carlo approximation of the variational entropy,
            summed across latent dimensions and averaged over the number of
            samples.
        """
        (v_samples, u_samples) = samples
        v_entropy = self.qv_distribution.get_entropy(v_samples)
        u_entropy = self.qu_distribution.get_entropy(u_samples, datapoints_indices)
        # Adjust for minibatch
        batch_size = u_samples.shape[-1]
        u_entropy = u_entropy * (self.num_datapoints / batch_size)
        entropy = v_entropy + u_entropy
        return torch.mean(entropy)

    def get_data_log_likelihood_sparse(self, samples, counts, holdout_mask):
        """
        P: number of non-zero elements in counts
        L: dimension of latent space
        b: batch size

        Ignores all zeros in the counts matrix
        # Sometimes counts and hodlout_mask do not match. This can happen if the original data had zeros, and holdout_mask accounts for them, but counts does not.
        # In such a scenario, compute the llhood twice, once for the indx in the counts, and once for the indx in the setdiff(indx_counts, indx_holdout_mask)
        """
        # Masking:
        # 1. (1) Remove the contribution of the heldout elements from the final sum

        # raise NotImplementedError("I'm not convinced this makes any senes")
        # True zeros: these are entries
        (v_samples, u_samples) = samples
        b = v_samples.shape[0]
        indx = counts.coalesce().indices()
        values = counts.coalesce().values()
        v = v_samples.transpose(1, 0).transpose(2, 1)  # columns
        u = u_samples.transpose(2, 0)  # rows
        u_rows = u[indx[0]]  # shape: (P, L, b)
        v_rows = v[indx[1]]  # shape: (P, L, b)

        prod = torch.sum(u_rows * v_rows, dim=1)  # shape: (P,b)
        # Compute value.xlogy(rate) - rate - (value + 1).lgamma()
        # 1. compute value.xlogy(rate) - (value + 1).lgamma()
        log_ll = values[:, None].xlogy(prod) - (values[:, None] + 1).lgamma()
        log_ll = torch.sum(log_ll, dim=0)
        # 2. compute -rate
        #neg_rate = torch.sum(u[None, :] * v[:, None], dim=tuple(range(0, b)))
        neg_rate = torch.sum(u[None, :] * v[:, None], dim=tuple(range(0, 3))) # shape: (,b)
        # Remove entries in neg_rate that ARE in the holdout_mask

        def compute_rate_for_missing(u, v, holdout_mask):
            """Just calculate the rate for the ones that are in the holdout_mask,"""
            indx = holdout_mask.coalesce().indices()
            u_rows = u[indx[0]]  # shape: (P, L, b)
            v_rows = v[indx[1]]  # shape: (P, L, b)
            # return torch.sum(u_rows * v_rows, dim=1) # shape: (,b)
            return torch.sum(u_rows * v_rows, dim=tuple(range(0, 2)))  # shape: (,b)

        correction = compute_rate_for_missing(u, v, holdout_mask)
        neg_rate = neg_rate - correction
        log_ll = log_ll - neg_rate
        # Adjust for minibatch
        batch_size = counts.shape[0]
        log_ll = log_ll * (self.num_datapoints / batch_size)
        return torch.mean(log_ll)

    def get_data_log_likelihood(self, samples, counts, holdout_mask):
        """Approximate log-likelihood term of ELBO using Monte Carlo samples.

        Args:
          samples: A list of samples. The length of the list is the number of variables being sampled.
          counts: A float-tensor with shape [batch_size, num_words].
          holdout_mask: A binary tensor with shape [batch_size, data_dim]. 1=valid, 0=train


        Returns:
          data_log_likelihood: A Monte-Carlo approximation of the count
            log-likelihood, summed across latent dimensions and averaged over the
            number of samples.
        """
        # TODO: check this with n_particles > 1
        (v_samples, u_samples) = samples
        # return 1
        rate = torch.transpose(torch.matmul(v_samples, u_samples), 1, 2)
        data_distribution = self.distribution(rate=rate)
        data_log_likelihood = data_distribution.log_prob(counts.to_dense())
        data_log_likelihood = torch.sum(
            torch.mul(data_log_likelihood, 1 - holdout_mask.to_dense()),
            dim=tuple(range(1, data_log_likelihood.dim())),
        )
        # Adjust for minibatch
        # TODO: do we need this, after averaging over?
        batch_size = counts.shape[0]
        data_log_likelihood = data_log_likelihood * (self.num_datapoints / batch_size)
        return torch.mean(data_log_likelihood)
        # Just zero-out the maksed entries
        # data_log_likelihood = data_log_likelihood - np.mul(data_log_likelihood, holdout_mask.to_dense()
        # data_log_likelihood = torch.sum(data_log_likelihood, dim=tuple(range(1, data_log_likelihood.dim())))

    def write_elbo(
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

    def forward(self, datapoints_indices, data, holdout_mask, step, use_sparse=True):
        """Approximate variational Lognormal ELBO using reparameterization.

        Args:
            datapoints_indices: An int-vector with shape [batch_size].
            data: A matrix with shape `[batch_size, num_words]`.
            step: The training step, used to log summaries to Tensorboard.

        Returns:
            elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value
              is averaged across samples and summed across batches.
        """
        samples = self.get_samples(datapoints_indices)
        log_prior = self.get_log_prior(samples)
        if use_sparse:
            data_log_likelihood = self.get_data_log_likelihood_sparse(
                samples, data, holdout_mask
            )
        else:
            data_log_likelihood = self.get_data_log_likelihood(
                samples, data, holdout_mask
            )
        entropy = self.get_entropy(samples, datapoints_indices)
        neg_kl_divergence = log_prior + entropy
        elbo = data_log_likelihood + neg_kl_divergence * self.annealing_factor
        self.write_elbo(
            step, elbo, entropy, log_prior, data_log_likelihood, neg_kl_divergence
        )
        return elbo

    def generate_data(self, n_samples=1, x=None, datapoints_indices=None):
        """
        Generate observations from the model.
        """
        obs = []
        for j in range(n_samples):
            v_samples = self.qv_distribution.sample(1)
            u_samples = self.qu_distribution.sample(1)
            mean_sample = torch.transpose(
                torch.squeeze(torch.matmul(v_samples, u_samples)), 0, 1
            )
            x_gen = self.distribution(mean_sample).sample()
            obs.append(x_gen)

        return obs

    @torch.no_grad()
    def save_txt(self, param_save_dir, **kwargs):
        """Save the model params to file

        Args:
            param_save_dir: The validation data matrix with shape `[num_datapoints, data_dim]`.

        Returns:
            None
        """
        np.savetxt(
            os.path.join(param_save_dir, "qv_loc"),
            self.qv_distribution.location.cpu().detach(),
        )
        np.savetxt(
            os.path.join(param_save_dir, "qv_scale"),
            self.qv_distribution.scale().cpu().detach(),
        )
        np.savetxt(
            os.path.join(param_save_dir, "qu_loc"),
            self.qu_distribution.location.cpu().detach(),
        )
        np.savetxt(
            os.path.join(param_save_dir, "qu_scale"),
            self.qu_distribution.scale().cpu().detach(),
        )

    def load_model(PATH):
        """Load the model from the PATH"""
        config = yaml.load(os.path.join(PATH, "config.yaml"))
        summary_writer = SummaryWriter(config["param_save_dir"])
        model = (
            PMF(
                config["device"],
                config["num_datapoints"],
                config["data_dim"],
                config["latent_dim"],
                config["num_samples"],
                config["print_steps"],
                summary_writer,
            )
            .to(config["device"])
            .double()
        )
        model.load_state_dict(torch.load(os.path.join(PATH, "model_trained.pt")))
        model.eval()
        return model

    def generate_holdout_mean(self, holdout_mask):
        # if holdout_mask is not a torch sparse tensor, convert it to one
        if not isinstance(holdout_mask, torch.sparse.FloatTensor):
            holdout_mask = sparse_tensor_from_sparse_matrix(holdout_mask)

        indx = holdout_mask.coalesce().indices()
        v = self.qv_distribution.sample(1).squeeze()  # columns
        u = self.qu_distribution.sample(1).squeeze().T  # rows
        u_rows = u[indx[0]]  # shape: (P, L)
        v_rows = v[indx[1]]  # shape: (P, L)
        prod = torch.sum(u_rows * v_rows, dim=1)  # shape: (P,)
        return prod, indx

    def generate_heldout_data(self, holdout_mask):
        prod, u = self.generate_holdout_mean(holdout_mask)
        # sample from the data distribution
        data_posterior = self.distribution(prod)
        x_generated = data_posterior.sample(torch.Size([1])).squeeze()
        # convert to sparse matrix
        return torch.sparse.FloatTensor(u, x_generated, holdout_mask.shape)

    def compute_llhood_sparse(self, x_vad, missing_indexes=None, v=None, u=None):
        """
        1. Compute llhood for x_vad indices
        2. Compute llhood for missing indices
            missing indices := mask_indices \ x_vad_indices
        3. Put together the two llhoods

        Computes the log-likelihood of every entry
        (i,j) -> k, theta_j = (u_j, v_j)
        for each x_k, compute the llhood, i.e., p(x_k | theta_j)

        Returns:
            llhood: the vector of llhoods for each datapoint (x_vad + missing)

        """

        assert missing_indexes is not None, "missing_indexes is None. Comptue it first."

        def inner_llhood(v, u, indx, vals):
            """Computes the likelihood for the given indices and values, and the local and global parameters."""
            # If the index or vals are in cpu, cast u and v to cpu too
            if indx.device.type == "cpu" and u.device.type != "cpu":
                print("putting to cpu")
                u = u.cpu()
                v = v.cpu()
                vals = vals.cpu()
            u_rows = u[indx[0]]  # shape: (P, L)
            v_rows = v[indx[1]]  # shape: (P, L)
            prod = torch.sum(u_rows * v_rows, dim=1)  # shape: (P,)
            # If vals are all zeros, the Poisson log-likelihood is just the -rate
            if vals.shape[0] == 1 and vals.sum() == 0:
                return -prod
            return self.distribution(prod).log_prob(vals)
        
        if v is None or u is None:
            v = self.qv_distribution.sample(1).squeeze()  # columns
            u = self.qu_distribution.sample(1).squeeze().T  # rows
        else:
            v = v.squeeze()
            u = u.squeeze().T

        # Compute for the observed indices
        indx = x_vad.coalesce().indices()
        vals = x_vad.coalesce().values()
        log_ll_obs = inner_llhood(v, u, indx, vals)

        # Compute for the missing indices
        if missing_indexes.shape[1] > 0:
            # vals = torch.zeros(missing_indexes.shape[1]).to(self.device)
            # Create a tensor of zeros of size 1
            vals = torch.zeros(1).to(self.device)
            log_ll_miss = inner_llhood(v, u, missing_indexes, vals)
            log_ll = torch.cat((log_ll_obs, log_ll_miss), dim=0)
        else:
            log_ll = log_ll_obs

        return log_ll

    def compute_llhood(self, x_vad, holdout_mask, v_samples=None, u_samples=None):
        """
        Compute loglikelihood for the given data, taking the mask into account.
        logP(X_vad | Z@W)

        Args:
            x_vad: A matrix with shape [num_datapoints, data_dim].
            holdout_mask: A binary tensor with shape [num_datapoints, data_dim]. 1=valid, 0=train

        Returns:
            log_ll: A tensor of shape [num_datapoints]
        """
        if v_samples is None or u_samples is None:
            v_samples = self.qv_distribution.sample(1)  # columns
            u_samples = self.qu_distribution.sample(1)  # rows

        holdoutmean_sample = torch.transpose(
            torch.squeeze(torch.matmul(v_samples, u_samples)), 0, 1
        )
        # print the device for each tensor
        # print("holdoutmean_sample.device", holdoutmean_sample.device)
        # print("holdout_mask.device", holdout_mask.device)
        holdoutmean_sample = holdoutmean_sample * holdout_mask
        # log_ll is [num_datapoints, data_dim]
        log_ll = self.distribution(holdoutmean_sample).log_prob(x_vad)
        # Remove the heldin llhoods
        log_ll = log_ll * holdout_mask
        log_ll = torch.sum(log_ll, dim=1)
        return log_ll

    def quick_test(self, x_vad, holdout_mask):
        v_samples = self.qv_distribution.sample(1)  # columns
        u_samples = self.qu_distribution.sample(1)  # rows
        ll1 = self.compute_llhood(
            x_vad.to_dense(), holdout_mask.to_dense(), v_samples, u_samples
        )
        ll2 = self.compute_llhood_sparse(x_vad, holdout_mask, v_samples, u_samples)
        assert torch.allclose(ll1, ll2)

    # @torch.no_grad()
    # def compute_heldout_loglikelihood_old(
    #     self, x_vad, holdout_mask, holdout_subjects, n_monte_carlo=100
    # ):
    #     """
    #     Compute the heldout loglikelihood.
    #     Approximate with L param monte carlo samples
    #     \frac{1}{L} \sum_{l}^{L}\sum_{i}^{N} \log(P(X_i \mid Z_m))

    #     Also cashes the llhood computation for the missing zeros [i.e., zero elements in the counts that have 1 in the mask]
    #     """
    #     r = 0.0
    #     # self.quick_test(x_vad, holdout_mask)
    #     print('Find missing indexes...')
    #     missing_indexes = get_missing_index(x_vad, holdout_mask)
    #     subsample_zeros = False
    #     print('Subsampling missing indx for computationl expediency...')
    #     for j in tqdm(range(n_monte_carlo), disable=n_monte_carlo == 1):
    #         if subsample_zeros:
    #             indx = torch.randint(0, missing_indexes.shape[1], (x_vad.coalesce().indices().shape[1],))
    #             b_hat = missing_indexes[:, indx]
    #         else:
    #             b_hat = missing_indexes
    #         log_ll = self.compute_llhood_sparse(x_vad, missing_indexes=b_hat)
    #         if j == 0:
    #             alpha = torch.min(log_ll) - 1.0
    #         # Compute the running logsumexp per datapoint (over datasets)
    #         r, alpha = streaming_logsumexp(log_ll, r, alpha)
    #     heldout_llhood = torch.log(r) + alpha
    #     # Add the -log(M) term
    #     norm_constant = torch.log(torch.tensor(n_monte_carlo, dtype=torch.float))
    #     heldout_llhood = torch.sum(heldout_llhood - norm_constant)
    #     # normalizing by the number of heldout values
    #     heldout_llhood /= log_ll.shape[0]
    #     if subsample_zeros:
    #         # Also divide by the number of missing values
    #         heldout_llhood =  heldout_llhood * indx.shape[0]/missing_indexes.shape[1]
    #     self.summary_writer.add_scalar(f"llhood/heldout_llhood{x_vad.shape[0]}", heldout_llhood)
    #     print(f"Heldout llhood: {heldout_llhood}")
    #     return heldout_llhood

    def t_test(self, x_vad, missing_indexes, n_monte_carlo=100):
        # Sample u and v, then, using the same u, v, compute the llhood with and without subsample_zeros
        # Create a list for v and u
        v_list = []
        u_list = []

        for i in range(n_monte_carlo):
            v = self.qv_distribution.sample(1).squeeze()
            u = self.qu_distribution.sample(1).squeeze()
            v_list.append(v)
            u_list.append(u)

        print("Using subsample_zeros = True")
        ll2 = self.compute_heldout_loglikelihood(
            x_vad,
            missing_indexes,
            n_monte_carlo,
            subsample_zeros=True,
            u_list=u_list,
            v_list=v_list,
        )
        print("Using subsample_zeros = False")
        ll1 = self.compute_heldout_loglikelihood(
            x_vad,
            missing_indexes,
            n_monte_carlo,
            subsample_zeros=False,
            u_list=u_list,
            v_list=v_list,
        )
        print(f"ll1: {ll1}, ll2: {ll2}")

    @torch.no_grad()
    def compute_heldout_loglikelihood(
        # self, x_vad, missing_indexes, n_monte_carlo=100, subsample_zeros=True,
        self,
        x_vad,
        missing_indexes,
        n_monte_carlo=100,
        subsample_zeros=False,
        write_llhood=True,
    ):
        """
        Compute the heldout loglikelihood.

        Args:
            missing_indexes [2, N]

        Approximate with L param monte carlo samples
        \frac{1}{L} \sum_{l}^{L}\sum_{i}^{N} \log(P(X_i \mid Z_m))

        Also cashes the llhood computation for the missing zeros [i.e., zero elements in the counts that have 1 in the mask]
        """

        def subsample_(arry, n_samples):
            indx = np.random.choice(arry.shape[1], n_samples, replace=False)
            return arry[:, indx]

        n_obs = x_vad.coalesce().indices().shape[1]

        # n_missing = 0 if len(missing_indexes.shape) > 0 else missing_indexes.shape[1]
        n_missing = missing_indexes.shape[1]

        n_subsample = np.minimum(n_obs * 1, n_missing)
        # n_subsample = int(n_missing*.1)
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
        for j in tqdm(range(n_monte_carlo), disable=n_monte_carlo == 1):
            # torch.cuda.empty_cache()
            # log_ll = self.compute_llhood_sparse(x_vad, missing_indexes=b_hat, u=u_list[j], v=v_list[j])
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
            # print(
            #     "adjusted llmiss = {:.2f}".format(llmiss_ * (n_missing / n_subsample))
            # )
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
