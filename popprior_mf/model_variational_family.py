"""
    Variation family concrete class. 
    Supports normal and log-normal mean-field inference. 


    See here: https://pyro.ai/examples/sparse_gamma.html

    @Author: Sohrab Salehi sohrab.salehi@columbia.edu
"""


import torch
import numpy as np
from model_skeleton import VariationalDistribution


class VariationalFamily(VariationalDistribution):
    """Implementation of a standard variational family."""

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
        fixed_scale=None,
        init_scale=-1, # -1 for xavier_uniform, > 0 for the value of the scale
        use_orthogonal=False,
        **kwargs,
    ):
        """Initialize variational family.

        Args:
            device: Device where operations take place (CPU or GPU).
            family: A string representing the variational family, either "normal" or  "lognormal".
            shape: A list representing the shape of the variational family.
            initial_loc: An optional tensor with shape `shape`, denoting the initial
            location of the variational family.
            batch_aware: If true enforce minibatch, that is require datapoint indices when computing llhood.
        """
        super().__init__()
        self.device = device
        self.fixed_scale = fixed_scale
        init_func = (
            torch.nn.init.orthogonal_
            if use_orthogonal
            else torch.nn.init.xavier_uniform_
        )
        if init_params:
            print(f'Initializing params at shape {shape}')
            if initial_loc is None:
                self.location = init_func(torch.nn.Parameter(torch.ones(shape)))
            else:
                # self.location = torch.nn.Parameter(torch.FloatTensor(np.log(initial_loc)))
                self.location = torch.nn.Parameter(torch.DoubleTensor(initial_loc))

            if self.fixed_scale is None:
                if init_scale == -1:
                    print('Initializing scale with xavier_uniform...')
                    self.log_scale = init_func(
                        torch.nn.Parameter(torch.zeros(shape))
                    )
                elif init_scale > 0:
                    self.log_scale = torch.nn.Parameter(torch.ones(shape)*init_scale)
                elif init_scale == -2:
                    # Have only one parameter for scale, i.e., all scales are the same
                    self.log_scale = (torch.ones(shape) * torch.nn.parameter(torch.ones(1))).to(self.device)
                else:
                    raise ValueError(f'Invalid init_scale value: {init_scale}')
            else:
                print(f'Fixing the SCALE TO {fixed_scale} at shape {shape}')
                self.log_scale = (torch.ones(shape)*fixed_scale).to(self.device)
        assert family == 'normal', f'Only normal family is supported for now. Got {self.family}'
        self.family = family
        self.batch_aware = batch_aware
        self.shape = shape
        self.nugget = nugget

        if setup_prior:
            self.prior = self._setup_prior(**kwargs)

    def _setup_prior(self, prior_scale=1.0, prior_concentration=0.1, prior_rate=0.3, prior_family=None):
        """
            Set prior distribution.
            If prior_family is None, then pick normal for normal, and gamma for lognormal. 
            Otherwise, it is up to the user to ensure compatible prior_family and family.
        """
        assert self.family == 'normal', f'Only normal family is supported for now. Got {self.family}'
        assert self.family in ["normal", "lognormal", 'gamma'], f"Unrecognized family {self.family}."
        # print('Prior setting in VariationalFamily...')
        if prior_family is None:
            if self.family == "normal":
                prior_family = "normal"
            elif self.family == "lognormal":
                prior_family = "gamma"
            elif self.family == 'gamma':
                prior_family = 'gamma'
            else:
                raise ValueError(f"Unrecognized prior distribution for family {self.family}")
       
        # print(f'Setting prior to {prior_family}')

        if prior_family == "normal":
            print(f'prior_scale is {prior_scale}')
            return torch.distributions.Normal(loc=0.0, scale=prior_scale)
        elif prior_family == "gamma":
            return torch.distributions.Gamma(concentration=prior_concentration, rate=prior_rate)
        elif prior_family == "half_normal":
            return torch.distributions.HalfNormal(scale=prior_scale)
        elif prior_family == "exponential":
            return torch.distributions.Exponential(rate=prior_rate)
        elif prior_family == "beta":
            return torch.distributions.Beta(concentration1=prior_concentration, concentration0=prior_rate)
        elif prior_family == "lognormal":
            print(f"Warning! Setting prior to lognormal with loc={prior_concentration} and scale={prior_rate}")
            return torch.distributions.LogNormal(loc=prior_concentration, scale=prior_rate)
        else:
            raise ValueError(f"Unrecognized prior distribution {prior_family}")
        return None


    def mean_parameters(self):
        return self.location
    
    def variance_parameters(self):
        return self.log_scale

    @staticmethod
    def make_positive(x, exponentiate=False, min=1e-8, max=1e3):
        """Make sure x is positive."""
        pos_func = torch.exp if exponentiate else torch.nn.functional.softplus
        return VariationalFamily.clamp(pos_func(x), min=min, max=max)
        
    @staticmethod
    def clamp(x, min=1e-8, max=1e3):
        return torch.clamp(x, min=min, max=max)

    def scale(self):
        """Constrain scale to be positive using softplus."""
        if self.fixed_scale is None:
            return torch.nn.functional.softplus(self.log_scale)
        else:
            return self.log_scale
        
    def _distribution(self, the_loc, the_scale):
        """Return a distribution object with the provided parameters."""
        if self.family == "normal":
            distribution = torch.distributions.Normal(loc=the_loc, scale=the_scale)
        elif self.family == "lognormal":
            distribution = torch.distributions.LogNormal(
                loc=the_loc,
                scale=torch.clamp(the_scale, min=1e-8, max=1e3),
            )
        elif self.family == "gamma":
            # clamp the scale and the location and also put the location through softplus
            #the_location = torch.clamp(torch.nn.functional.softplus(the_loc), min=1e-5, max=1e2)
            the_location = self.make_positive(the_loc, None, None)
            # replace nans with 1e-5 in the_location
            # the_location = torch.nan_to_num(the_location, nan=1e-5)
            distribution = torch.distributions.Gamma(
                concentration=the_location, 
                rate=torch.clamp(the_scale, min=1e-8, max=1e3),
            )
        else:
            raise ValueError(f"Unrecognized prior distribution {self.family}.")
        return distribution

    def distribution(self, datapoints_indices):
        """Create variational distribution."""
        if self.batch_aware:
            assert datapoints_indices is not None

        if self.batch_aware:
            the_loc = self.location[:, datapoints_indices]
            the_scale = self.scale()[:, datapoints_indices]
        else:
            the_loc = self.location
            the_scale = self.scale()

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
        # Sums over the last dim (indiv_log_prob is [n_sample, L1])
        indiv_log_prob = self.prior.log_prob(samples)
        # TODO: check if summing over W makes sense
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
        # print the device of num_samples and self.location and datapoint_indices
        #print(f'self.location device: {self.location.device}')
        
        return self.distribution(datapoints_indices).rsample([num_samples])
    
    def save_txt(self):
        raise NotImplementedError
