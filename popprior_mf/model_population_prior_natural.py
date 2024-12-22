"""
    Implements the Population Mixture of Priors.

    @Author: Sohrab Salehi (sohrab.salehi@columbia.edu)
"""


import os
import torch
import numpy as np
from model_skeleton import StandAloneDistribution
from model_variational_family_natural import VariationalFamilyNatural


class PopulationPriorNatural(StandAloneDistribution):
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
        eta1,
        eta2,
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
        self.eta1 = eta1
        self.eta2 = eta2
        self.use_sparse = use_sparse

        assert (
            self.use_sparse == False
        ), "Sparse is not implemented for natural parameters."
        # TODO: do we need to disable gradient for pseudo_obs?

    def distribution(self, the_eta1, the_eta2):
        """Create variational distribution."""
        if self.family == "lognormal":
            the_location, the_scale = VariationalFamilyNatural.natural_to_lognormal(
                the_eta1, the_eta2
            )
            the_scale = VariationalFamilyNatural.make_positive(the_scale)
            distribution = torch.distributions.LogNormal(
                loc=the_location, scale=the_scale
            )
        else:
            raise ValueError(f"Unrecognized prior distribution {self.family}.")
        return distribution


    @property
    def scale(self):
        """Return scale parameter, applying the transformation from natural to lognormal."""
        the_scale = VariationalFamilyNatural.natural_to_lognormal(self.eta1, self.eta2)[1]
        return VariationalFamilyNatural.make_positive(the_scale)
    
    @property
    def mean(self):
        """Return location parameter, applying the transformation from natural to lognormal."""
        return VariationalFamilyNatural.natural_to_lognormal(self.eta1, self.eta2)[0]


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
        # TODO: detach eta1_sub and eta2_sub from the graph
        if self.tag == "_local":
            n_tmp = self.eta1.shape[1]
            indx = np.random.randint(low=0, high=n_tmp, size=(self.K,))
            eta1_sub = self.eta1[:, indx].detach().T
            eta2_sub = self.eta2[:, indx].detach().T
        else:
            n_tmp = self.eta1.shape[0]
            indx = np.random.randint(low=0, high=n_tmp, size=(self.K,))
            eta1_sub = self.eta1[indx, :].detach()
            eta2_sub = self.eta2[indx, :].detach()

        the_K = self.K
        if self.tag == "_local":
            expansion_pattern = (slice(None), None, slice(None), None)
            # sum_logs.shape: [nPseudoObsRows, nParticles, nLatent, batchSize]
            sum_logs = (
                self.distribution(
                    eta1_sub[expansion_pattern], eta2_sub[expansion_pattern]
                )
                .log_prob(x[None, :])
                .to(self.device)
            )

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
                    eta1_sub[expansion_pattern], eta2_sub[expansion_pattern]
                )
                .log_prob(x[None, :])
                .to(self.device)
            )

            # log [(1/K) \sum logq(z_i | X; u_k))]
            # Sum over the latent dimension (all dimensions are IID)
            sum_logs = torch.sum(sum_logs, dim=3)
        else:
            raise ValueError(f"Unrecognized tag: {self.tag}.")

        # Average over the mixtures
        log_prob = -torch.log(torch.tensor(the_K, dtype=torch.float)) + torch.logsumexp(
            sum_logs, dim=0
        )

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
