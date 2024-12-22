"""
    Implements the Population Mixture of Priors.

    @Author: Sohrab Salehi (sohrab.salehi@columbia.edu)
"""


import os
import torch
import numpy as np
from model_skeleton import StandAloneDistribution, VariableBank


class PopulationPrior(StandAloneDistribution):
    """
    The agregate prior
    P(z; \lambda) = 1/K * \sum_{k=1}^K q(z | X; u_k)
    where u_k's are the pseudo-observations [K x L]
    and q is the variational posterior distribution
    and \lambda : \{u_1, \dots u_k, log_scale\}
    with a fixed scale
    """

    def __init__(
        self,
        device,
        family,
        shape,
        log_scale,
        pseudo_obs,
        the_scale,
        tag="",
        use_sparse=False,
    ):
        super().__init__()
        """
        Args:
            Variable_bank: an object of type VariableBank that supprot the following methods:
                - get_variable: returns a set of variables of size (K, L)
        """
        self.tag = tag
        self.device = device
        self.family = family
        self.shape = shape
        self.K = shape[0]
        self.pseudo_obs = pseudo_obs
        self.the_scale = the_scale
        self.use_sparse = use_sparse

        # TODO: do we need to disable gradient for pseudo_obs?

    def distribution(self, location, the_scale):
        """Create variational distribution."""
        if self.family == "normal":
            distribution = torch.distributions.Normal(loc=location, scale=the_scale)
        elif self.family == "lognormal":
            distribution = torch.distributions.LogNormal(
                loc=location,
                scale=torch.clamp(the_scale, min=1e-8, max=1e1),
            )
        elif self.family == "gamma":
            # clamp the scale and the location and also put the location through softplus
            the_location = torch.clamp(torch.nn.functional.softplus(location), min=1e-5, max=1e2)
            # the_location = torch.nan_to_num(the_location, nan=1e-5)
            distribution = torch.distributions.Gamma(
                concentration=the_location, 
                rate=torch.clamp(the_scale, min=1e-8, max=1e1),
            )
        else:
            raise ValueError(f"Unrecognized prior distribution {self.family}.")
        return distribution

    def scale(self, the_log_scale):
        """Constrain scale to be positive using softplus."""
        return torch.nn.functional.softplus(the_log_scale)

    def sample(self, out_shape):
        """Pick a k then sample from q(z | u_k)
        This does not need rsample since there is no need for grad computation.

        Args:
            out_shape: A list of size 3 with the following num_samples, batch_size, event_size

        Returns:
            A tensor of shape [num_samples, batch_size, event_size]
        """
        num_samples, batch_size, event_size = out_shape
        assert (
            num_samples == 1
        ), "This is a slow implementation. Use a batch version instead."

        # TODO: Should we use torch.distributions.mixture_same_family.MixtureSameFamily?
        # TODO: should we disable gradient for this operation? Probably not...
        # pseudo_obs, the_scale = self.variable_bank.get_variable(self.K)
        # get sample indexes of the size of self.pseudo_obs.shape[0]
        indx = torch.randint(high=self.K, size=(self.pseudo_obs.shape[0],))

        k = torch.randint(high=self.K, size=(batch_size,))
        return self.distribution(pseudo_obs[k, :], the_scale[k, :]).rsample(
            [num_samples]
        )

    def log_prob(self, x):
        """
        Log probability for each sample.
        P(z; \lambda) = 1/K * \sum_{k=1}^K q(z | X; u_k)

        This prior does not depend on individual datapoints.
        Args:
            x: A tensor of shape [n_sample, L1, L2], i.e., [n_particles, event_shape, batch_shape] (for row_param, this is [n_paricles, latent_dim, batch_shape])

        Returns:
            A tensor of shape [n_paricles/n_sample, batch_shape]

        Descriptoin:
            pseudo_obs: [K, L], i.e., [n_mixtures, event_shape]
            sum_logs: A tensor of shape [n_mixtures, n_particles, event_shape, batch_shape], i.e., [K, n_sample, L1, L2]
            a. sum over the event_shape
            b. compute sumlogexp over n_mixture
        """
        # see here for hints on reshaping...  https://github.com/pytorch/pytorch/issues/76709
        # Gather the components of the mixture
        if self.tag == "_local":
            n_tmp = self.pseudo_obs.shape[1]
            indx = np.random.randint(low=0, high=n_tmp, size=(self.K,))
            pseudo_obs = self.pseudo_obs[:, indx].T
            the_scale = self.scale(self.the_scale[:, indx]).T
        else:
            n_tmp = self.pseudo_obs.shape[0]
            indx = np.random.randint(low=0, high=n_tmp, size=(self.K,))
            pseudo_obs = self.pseudo_obs[indx, :]
            the_scale = self.scale(self.the_scale[indx, :])
        # pseudo_obs, the_scale  = self.variable_bank.get_variable(self.K)

        # #use_sparse = False
        # if self.use_sparse:
        #     # mean of lognormal: exp(mu + sigma^2/2)
        #     # std of lognormal: sqrt((exp(sigma^2) - 1) * exp(2 * mu + sigma^2))
        #     # Add a 20% artificial pseudo_observations with mu \in  [-1, 0] and sigma = 1
        #     new_size = int(self.K * 1)
        #     # sample uniformly from [-1, 0], of size new_size  (new_size, L)
        #     # get an array of all -4
        #     new_means = torch.ones(new_size, pseudo_obs.shape[1]) * -4
        #     new_scale = torch.ones(new_size, pseudo_obs.shape[1])
        #     # assign them to the right device
        #     new_means = new_means.to(self.device)
        #     new_scale = new_scale.to(self.device)
        #     # concatenate with the original pseudo_obs
        #     pseudo_obs = torch.cat((pseudo_obs, new_means), dim=0)
        #     the_scale = torch.cat((the_scale, new_scale), dim=0)
        #     # pseudo_obs = new_means
        #     # the_scale = new_scale

        the_K = self.K
        if self.tag == "_local":
            expansion_pattern = (slice(None), None, slice(None), None)
            # sum_logs.shape: [nPseudoObsRows, nParticles, nLatent, batchSize]
            sum_logs = (
                self.distribution(
                    pseudo_obs[expansion_pattern], the_scale[expansion_pattern]
                )
                .log_prob(x[None, :])
                .to(self.device)
            )

            if self.use_sparse:
                n_tmp = np.int(self.K * 0.2)
                the_K = self.K + n_tmp
                # add mixture components as very narrow gaussians with mean 0 and std 1e-8
                the_means = torch.zeros(pseudo_obs.shape[1], n_tmp).T
                the_scales = torch.ones_like(the_means) * 1e-8
                sparse_logs = (
                    torch.distributions.Normal(
                        loc=the_means[expansion_pattern].to(self.device),
                        scale=the_scales[expansion_pattern].to(self.device),
                    )
                    .log_prob(x[None, :])
                    .to(self.device)
                )
                # concatenate to sum_logs by the nPseudoObsRow's dimension
                sum_logs = torch.cat((sum_logs, sparse_logs), dim=0)

            # log [(1/K) \sum logq(z_i | X; u_k))]
            # Sum over the latent dimension (all dimensions are IID)
            sum_logs = torch.sum(sum_logs, dim=2)
        elif self.tag == "_global":
            expansion_pattern = (slice(None), None, None, slice(None))
            # sum_logs.shape: [nPseudoObsCol, nParticles, nFeatures, latnet_dim],
            # nPseudoObsCol is the number_of_mixtures
            # log_prob would be: [nParticles, nFeatures]
            sum_logs = (
                self.distribution(
                    pseudo_obs[expansion_pattern], the_scale[expansion_pattern]
                )
                .log_prob(x[None, :])
                .to(self.device)
            )

            if self.use_sparse:
                n_tmp = np.int(self.K * 0.2)
                the_K = self.K + n_tmp
                # TODO: if the distributions are identical, just weight them...
                # add mixture components as very narrow gaussians with mean 0 and std 1e-8
                the_means = torch.zeros(n_tmp, pseudo_obs.shape[1])
                the_scales = torch.ones_like(the_means) * 1e-8
                sparse_logs = (
                    torch.distributions.Normal(
                        loc=the_means[expansion_pattern].to(self.device),
                        scale=the_scales[expansion_pattern].to(self.device),
                    )
                    .log_prob(x[None, :])
                    .to(self.device)
                )
                
                # concatenate to sum_logs by the nPseudoObsCol's dimension
                sum_logs = torch.cat((sum_logs, sparse_logs), dim=0)

            # log [(1/K) \sum logq(z_i | X; u_k))]
            # Sum over the latent dimension (all dimensions are IID)
            sum_logs = torch.sum(sum_logs, dim=3)
        else:
            raise ValueError(f"Unrecognized tag: {self.tag}.")

        # Average over the mixtures
        log_prob = -torch.log(
            # torch.tensor(self.K, dtype=torch.float)
            torch.tensor(the_K, dtype=torch.float)
        ) + torch.logsumexp(sum_logs, dim=0)

        return log_prob

    @torch.no_grad()
    def save_txt(self, param_save_dir, **kwargs):
        """Save the model params to file using numpy.savetxt

        Args:
            param_save_dir: The directory to save the model params to.

        Returns:
            None
        """
        pass
