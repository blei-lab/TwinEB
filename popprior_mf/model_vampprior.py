"""
    Implements the Variational Mixture of Posteriors Prior (VampPrior) model.

    @Author: Sohrab Salehi (sohrab.salehi@columbia.edu)
"""


import os
import torch
import numpy as np
from model_skeleton import StandAloneDistribution


class VampPrior(StandAloneDistribution):
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
        tag="",
        initial_loc=None,
        use_orthogonal=False,
    ):
        super().__init__()

        self.tag = tag
        self.device = device
        self.family = family
        self.shape = shape
        self.K = shape[0]
        # which random initialization to use?
        init_func = (
            torch.nn.init.orthogonal_
            if use_orthogonal
            else torch.nn.init.xavier_uniform_
        )
        if initial_loc is None:
            self.pseudo_obs = init_func(torch.nn.Parameter(torch.ones(shape)))

        else:
            self.pseudo_obs = torch.nn.Parameter(
                torch.FloatTensor(torch.log(initial_loc))
            )

        self.log_scale = init_func(torch.nn.Parameter(torch.zeros(shape)))

        # print('WARNING - fixing pseudo-observations to 0.0')
        # self.pseudo_obs = torch.zeros(shape)
        # self.pseudo_obs.requires_grad = False
        # print("The scale is fixed to 1.")
        # self.log_scale = torch.ones(shape)
        # self.log_scale = torch.ones(shape).to(self.device)*log_scale
        # self.log_scale = torch.ones(shape).to(self.device)
        # Make it a parameter to learn
        # self.log_scale = torch.nn.Parameter(torch.ones(shape).to(self.device))
        

        print(f"The scale is set to {log_scale}")
        print(f"Pseudo-observations are initialized to size {self.pseudo_obs.shape}")

    def init_pseudo_obs(self, pseudo_obs_list):
        """
        Initialize the pseudo_obs

        Args:
            pseudo_obs: A list, first item is the pseudo_obs, second item is the scale        
        """
        # for each dimension of the pseudo_obs check the shape is correct
        # for i in range(len(pseudo_obs.shape)):
        #     assert (
        #         pseudo_obs.shape[i] == self.shape[i]
        #     ), "The shape of pseudo_obs is not correct. Should be {}, but is {}".format(
        #         self.shape, pseudo_obs.shape
        #     )
        pseudo_obs, log_scale = pseudo_obs_list
        print("init pseudo_obs in vamp_prior")
        self.pseudo_obs = torch.nn.Parameter(torch.DoubleTensor(pseudo_obs).to(self.device))
        self.log_scale = torch.nn.Parameter(torch.DoubleTensor(log_scale).to(self.device))

    def distribution(self, location, the_scale):
        """Create variational distribution."""
        # the_scale = self.scale()
        if self.family == "normal":
            distribution = torch.distributions.Normal(loc=location, scale=the_scale)
        elif self.family == "lognormal":
            distribution = torch.distributions.LogNormal(
                loc=location, scale=torch.clamp(the_scale)
            )
        # add gamma
        elif self.family == "gamma":
            # put the location true softplus and clamp
            the_location = torch.clamp(
                torch.nn.functional.softplus(location)
            )
            distribution = torch.distributions.Gamma(
                concentration=the_location, rate=the_scale
            )
        else:
            raise ValueError(f"Unrecognized prior distribution {self.family}.")
        return distribution

    def scale(self):
        """Constrain scale to be positive using softplus."""
        return torch.nn.functional.softplus(self.log_scale)
        # return self.log_scale

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
        k = torch.randint(high=self.K, size=(batch_size,))
        return self.distribution(self.pseudo_obs[k, :], self.scale()[k, :]).rsample(
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
        the_scale = self.scale()
        if self.tag == "_local":
            sum_logs = (
                self.distribution(
                    self.pseudo_obs[:, None, :, None], the_scale[:, None, :, None]
                )
                .log_prob(x[None, :])
                .to(self.device)
            )
            # log [(1/K) \sum logq(z_i | X; u_k))]
            # Sum over the latent dimension (all dimensions are IID)
            sum_logs = torch.sum(sum_logs, dim=2)
        elif self.tag == "_global":
            sum_logs = (
                self.distribution(
                    self.pseudo_obs[:, None, None, :], the_scale[:, None, None, :]
                )
                .log_prob(x[None, :])
                .to(self.device)
            )
            # log [(1/K) \sum logq(z_i | X; u_k))]
            # Sum over the latent dimension (all dimensions are IID)
            sum_logs = torch.sum(sum_logs, dim=3)
        else:
            raise ValueError(f"Unrecognized tag: {self.tag}.")

        log_prob = -torch.log(
            torch.tensor(self.K, dtype=torch.float)
        ) + torch.logsumexp(sum_logs, dim=0)

        # # Assert that log_prob is equal to log_prob under Normal(0, 1)
        # tmp_dist = torch.distributions.Normal(loc=0.0, scale=1.)
        # tmp_indiv_log_prob = tmp_dist.log_prob(x)
        # return tmp_indiv_log_prob
        # # tmp_indiv_log_prob = torch.sum(tmp_indiv_log_prob, dim=1)
        # # assert that tmp_indiv_log_prob is equal to log_prob
        # assert torch.allclose(tmp_indiv_log_prob, log_prob), "The log_prob is not equal to the log_prob under Normal(0, 1)"
        return log_prob

    @torch.no_grad()
    def save_txt(self, param_save_dir, **kwargs):
        """Save the model params to file using numpy.savetxt

        Args:
            param_save_dir: The directory to save the model params to.

        Returns:
            None
        """
        np.savetxt(
            os.path.join(param_save_dir, f"pseudo_obs_loc{self.tag}"),
            self.pseudo_obs.cpu().detach(),
        )
        # np.savetxt(
        #     os.path.join(param_save_dir, f"pseudo_obs_loc_init{self.tag}"),
        #     self._init_pseudo_obs,
        # )
