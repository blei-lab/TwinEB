"""
    Variation family concrete class. 
    Supports normal and log-normal mean-field inference. 


    See here: https://pyro.ai/examples/sparse_gamma.html

    @Author: Sohrab Salehi sohrab.salehi@columbia.edu
"""


import torch
import numpy as np
from model_skeleton import VariationalDistribution
from exponential_family_utils import _compute_lognormal_mean, _compute_gamma_mean

ln_half = torch.log(torch.tensor(0.5))

class VariationalFamilyNatural(VariationalDistribution):
    """
    Implementation of a standard variational family.

    The natural parameterization logic is chiefly implemented in the self.distribution method.

    """

    def __init__(
        self,
        device,
        family,
        shape,
        initial_loc=None,
        batch_aware=True,
        init_params=True,
        nugget=0.0,
        setup_prior=True,
        use_orthogonal=False,
        regularize_prior=False,
        **kwargs,
    ):
        """Initialize variational family.

        Args:
            device: Device where operations take place (CPU or GPU).
            family: A string representing the variational family, either "normal" or  "lognormal".
            shape: A list representing the shape of the variational family.
            batch_aware: If true enforce minibatch, that is require datapoint indices when computing llhood.
        """
        super().__init__()
        self.device = device
        init_func = (
            torch.nn.init.orthogonal_
            if use_orthogonal
            else torch.nn.init.xavier_uniform_
        )
        if init_params:
            self.eta1 = init_func(torch.nn.Parameter(torch.zeros(shape)))
            self.ln_minus_eta2 = init_func(torch.nn.Parameter(torch.zeros(shape)))
        self.family = family
        self.batch_aware = batch_aware
        self.shape = shape
        self.nugget = nugget
        self.regularize_prior = regularize_prior

        if setup_prior:
            self.prior = self._setup_prior(**kwargs)


    def mean_parameters(self):
        return self.eta1
    
    def variance_parameters(self):
        return self.ln_minus_eta2

    def _compute_mean(self, convert_numpy=True):
        """
        Compute the mean of the variational family.
        """
        the_loc = self.location
        the_scale = self.scale
        if convert_numpy:
            the_loc = the_loc.cpu().detach().numpy()
            the_scale = the_scale.cpu().detach().numpy()

        if self.family == 'lognormal':
            return _compute_lognormal_mean(the_loc, the_scale)
        elif self.family == 'gamma':
            return _compute_gamma_mean(the_loc, the_scale)
        else:
            raise ValueError(f'Unrecognized var_family {self.family}')


    def _setup_prior(
        self,
        prior_scale=1.0,
        prior_concentration=0.1,
        prior_rate=0.3,
        prior_family=None,
        fixed_scale=None,
    ):
        """
        Set prior distribution.
        If prior_family is None, then pick normal for normal, and gamma for lognormal.
        Otherwise, it is up to the user to ensure compatible prior_family and family.
        """
        assert self.family in [
            "normal",
            "lognormal",
            "gamma",
        ], f"Unrecognized family {self.family}."
        # print("Prior setting in VariationalFamily...")
        if prior_family is None:
            if self.family == "normal":
                prior_family = "normal"
            elif self.family == "lognormal":
                prior_family = "gamma"
            elif self.family == "gamma":
                prior_family = "gamma"
            else:
                raise ValueError(
                    f"Unrecognized prior distribution for family {self.family}"
                )

        print(f"Setting prior to {prior_family}")
        if prior_family == "normal":
            return torch.distributions.Normal(loc=0.0, scale=prior_scale)
        elif prior_family == "gamma":
            return torch.distributions.Gamma(
                concentration=prior_concentration, rate=prior_rate
            )
        elif prior_family == "half_normal":
            return torch.distributions.HalfNormal(scale=prior_scale)
        elif prior_family == "exponential":
            return torch.distributions.Exponential(rate=prior_rate)
        elif prior_family == "beta":
            return torch.distributions.Beta(
                concentration1=prior_concentration, concentration0=prior_rate
            )
        elif prior_family == "lognormal":
            print(
                f"Warning! Setting prior to lognormal with loc={prior_concentration} and scale={prior_rate}"
            )
            return torch.distributions.LogNormal(
                loc=prior_concentration, scale=prior_rate
            )
        else:
            raise ValueError(f"Unrecognized prior distribution {prior_family}")
        return None

    @staticmethod
    def natural_to_lognormal(eta_1, ln_minus_eta2, exponentiate=False, nugget=1e-8):
        """
        
        (eta_1, lng_minus_eta_2) -> (location (mu), scale (sigma))

        Return the location and scale (standard deviation) parameters of a lognormal distribution, given its natural parameters.
        \psi = ln(-eta_2)
        mu = -0.5 * eta_1 / eta_2   ---> mu = -0.5 * eta_1 * exp(-\psi) = -eta_1 * exp(ln(.5) - \psi)
        sigma^2 = -0.5 / eta_2 ---> sigma^2 = exp(ln(.5) - \psi))) ---> sigma = sqrt(sigma^2)
        """
        # assert torch.all(eta_2 <= 0.0), f"eta_2 must be negative, but is {eta_2}."
        pos_func = torch.exp if exponentiate else torch.nn.functional.softplus
        sigma2 = pos_func(ln_half - ln_minus_eta2)
        assert torch.all(sigma2 >= 0.0), f"sigma2 must be positive, but is {sigma2}."
        # should this be clamped here?
        #sigma2 = VariationalFamilyNatural.clamp(sigma2)
        mu = -eta_1 * sigma2

        # Compute the norm of eta_2
        # the_norm = torch.norm(eta_2, keepdim=False)
        # print(the_norm.detach().cpu().numpy())
        #return mu, torch.sqrt(sigma2)
        return mu, sigma2


    @staticmethod
    def lognormal_to_natural(location, ln_sigma, exponentiate=False):
        """
        Given the mean parameterization, i.e., location and standard deviation, return the natural parameters of the lognormal distribution.
        """
        pos_func = torch.exp if exponentiate else torch.nn.functional.softplus
        sigma = pos_func(ln_sigma)
        eta1 = location / (sigma ** 2)
        eta2 = -0.5 / (sigma ** 2)
        return eta1, eta2

    # @staticmethod
    # def natural_to_lognormal(eta_1, log_eta_2, exponentiate=False, nugget=1e-8):
    #     """
    #     mu = -0.5 * eta_1 / eta_2
    #     sigma^2 = -0.5 / eta_2
    #     """
    #     # ensure eta_2 is negative
    #     eta_2 = (
    #         #-torch.exp(eta_2) if exponentiate else torch.nn.functional.softplus(-eta_2) # this softplus (with minus inside) is somehow more stable for mixture priors
    #         -torch.exp(eta_2) if exponentiate else -torch.nn.functional.softplus(eta_2)
    #     )
    #     assert torch.all(eta_2 <= 0.0), f"eta_2 must be negative, but is {eta_2}."

    #     # Compute the norm of eta_2
    #     # the_norm = torch.norm(eta_2, keepdim=False)
    #     # print(the_norm.detach().cpu().numpy())
    #     return -0.5 * eta_1 / (eta_2), -0.5 / (eta_2)


    @staticmethod
    def make_positive(x, exponentiate=False, min=1e-8, max=1e3):
        """Make sure x is positive."""
        pos_func = torch.exp if exponentiate else torch.nn.functional.softplus
        return VariationalFamilyNatural.clamp(pos_func(x), min=min, max=max)

    @staticmethod
    def clamp(x, min=1e-8, max=1e3):
        return torch.clamp(x, min=min, max=max)

    @property
    def location(self):
        """Return location parameter, applying the transformation from natural to lognormal."""
        return self.natural_to_lognormal(self.eta1, self.ln_minus_eta2)[0]

    @property
    def scale(self):
        """Return scale parameter, applying the transformation from natural to lognormal."""
        the_scale = self.natural_to_lognormal(self.eta1, self.ln_minus_eta2)[1]
        #return self.make_positive(the_scale)
        return VariationalFamilyNatural.clamp(the_scale)
        #return the_scale

    def _distribution(self, the_loc, the_scale):
        """Return a distribution object with the provided parameters."""
        #the_scale = self.make_positive(the_scale)
        the_scale = VariationalFamilyNatural.clamp(the_scale)

        # compute the norm of the the_scale
        # the_norm = torch.norm(the_scale, keepdim=False)
        # # just show 2 decimal places .2f
        # print(f"|scale|: {the_norm.detach().cpu().numpy():.2f}")

        # # compute the norm for the location
        # the_norm = torch.norm(the_loc, keepdim=False)
        # # just show 2 decimal places .2f
        # print(f"|loc|: {the_norm.detach().cpu().numpy():.2f}")

        if self.family == "normal":
            distribution = torch.distributions.Normal(loc=the_loc, scale=the_scale)
        elif self.family == "lognormal":
            distribution = torch.distributions.LogNormal(loc=the_loc, scale=the_scale)
        elif self.family == "gamma":
            distribution = torch.distributions.Gamma(
                concentration=self.make_positive(the_loc, None, None), rate=the_scale
            )
        else:
            raise ValueError(f"Unrecognized prior distribution {self.family}.")
        return distribution

    def distribution(self, datapoints_indices):
        """Create variational distribution."""
        if self.batch_aware:
            assert datapoints_indices is not None

        if self.batch_aware:
            the_loc = self.eta1[:, datapoints_indices]
            the_scale = self.ln_minus_eta2[:, datapoints_indices]
        else:
            the_loc = self.eta1
            the_scale = self.ln_minus_eta2

        # now convert to mean parameterization
        the_loc, the_scale = self.natural_to_lognormal(the_loc, the_scale)

        return self._distribution(the_loc, the_scale)

    def get_log_prior(self, samples, reduction="sum"):
        """
        Compute log prior of samples.

        Args:
            samples:
            If W:
                [number_of_particles, data_dim, latent_dim]
            If Z:
                [number_of_particles, latent_dim, batch_size]
        Returns:
            A tensor of shape [n_sample]
        """
        # Sums over the last dim (indiv_log_prob is [n_sample, data_dim, latent_dim])
        # log_prob will be [n_samples]
        indiv_log_prob = self.prior.log_prob(samples)

        if reduction == "sum":
            log_prior = torch.sum(
                indiv_log_prob.to(self.device),
                dim=tuple(range(1, len(indiv_log_prob.shape))),
            )
        elif reduction == "mean":
            log_prior = torch.mean(
                indiv_log_prob.to(self.device),
                dim=tuple(range(1, len(indiv_log_prob.shape))),
            )
        else:
            raise ValueError(f"Unrecognized reduction {reduction}.")

        # add regularization (log_prob_reg is a scalar) to each particle
        # check if the prior has a regularization term
        if self.regularize_prior:
            if hasattr(self.prior, "get_regularization"):
                log_prior = log_prior + self.prior.get_regularization()

        return log_prior

    def get_entropy(self, samples, datapoints_indices=None, reduction="sum"):
        """Compute entropy of samples from variational distribution."""
        if self.batch_aware:
            assert datapoints_indices is not None
        # Sum all but first axis
        individual_entropy = self.distribution(datapoints_indices).log_prob(
            samples + self.nugget
        )
        if reduction == "sum":
            entropy = -torch.sum(
                individual_entropy.to(self.device),
                dim=tuple(range(1, len(samples.shape))),
            )
        else:
            entropy = -torch.mean(
                individual_entropy.to(self.device),
                dim=tuple(range(1, len(samples.shape))),
            )
        return entropy

    def sample(self, num_samples):
        """Sample from variational family using reparameterization."""
        datapoints_indices = (
            None if not self.batch_aware else np.arange(self.location.shape[1])
        )

        return self.distribution(datapoints_indices).rsample([num_samples])

    def save_txt(self):
        raise NotImplementedError
