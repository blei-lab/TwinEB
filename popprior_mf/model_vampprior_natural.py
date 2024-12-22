"""
    Implements the Variational Mixture of Posteriors Prior (VampPrior) model.

    Author: De-identified Author
"""


import os
import torch
import numpy as np
from model_skeleton import StandAloneDistribution
from model_variational_family_natural import VariationalFamilyNatural
from torch.distributions import MixtureSameFamily




class MixtureOfMixtures(StandAloneDistribution):
    """
    Define a class that is a mixture of two Mixtures, and has a log_prob method

    It's a wrapper around the torch.distributions.MixtureSameFamily. 
    Accepts two torch.distributions.MixtureSameFamily, for any given input, will chunk it, give hte first to the first mixture, and the second to the second mixture, then return the average
    P(X1:K) = 1/2 * P1(X1:K) + 1/2 * P2(X1:K)

    # Args
    mix1: torch.distributions.MixtureSameFamily
    mix2: torch.distributions.MixtureSameFamily
    """

    def __init__(self, mix1, mix2):
        super().__init__()
        self.mix1 = mix1
        self.mix2 = mix2
    
    def log_prob(self, x):
        return 0.5 * self.mix1.log_prob(x) + 0.5 * self.mix2.log_prob(x)





class VampPriorNatural(StandAloneDistribution):
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
        tag="",
        initial_loc=None,
        use_orthogonal=False,
        use_sparse=False,
        use_mixture_weights=True,
    ):
        """
        use_sparse: if True, will set eta1 to zero.
        """
        super().__init__()
        self.tag = tag
        self.device = device
        self.family = family
        self.shape = shape
        self.K = shape[0]
        self.use_sparse = use_sparse
        # which random initialization to use?
        init_func = (
            torch.nn.init.orthogonal_
            if use_orthogonal
            else torch.nn.init.xavier_uniform_
        )
        if self.use_sparse:
            print('WARNING - fixing pseudo-observations to 0.0')
            self.eta1 = torch.zeros(shape).to(self.device)
            self.eta1.requires_grad = False
        else:
            if initial_loc is None:
                self.eta1 = init_func(torch.nn.Parameter(torch.ones(shape)))
                #print('Warning! Setting all pseudo_obs to 0')
                #self.eta1 = torch.zeros(shape)
            else:
                raise NotImplementedError("Need to map from mean to natural parameters")
                self.eta1 = torch.nn.Parameter(torch.FloatTensor(initial_loc))
        self.eta2 = init_func(torch.nn.Parameter(torch.ones(shape)))
        self.use_mixture_weights = use_mixture_weights
        if self.use_mixture_weights:
            self.mix_vals = torch.nn.Parameter(torch.ones(self.K))
            self.log_e0 = torch.nn.Parameter(torch.tensor(1/self.K))
        else:
            self.mix_vals = torch.zeros(self.K)
            self.mix_vals.requires_grad = False
            self.log_e0 = torch.zeros(1)
            self.log_e0.requires_grad = False
        
        

    def init_pseudo_obs(self, pseudo_obs_list):
        """Initialize the pseudo_obs"""
        # TODO: now map from mean to natural parameters
        #raise NotImplementedError("Need to map from mean to natural parameters")
        # for each dimension of the pseudo_obs check the shape is correct
        # for i in range(len(pseudo_obs.shape)):
        #     assert (
        #         pseudo_obs.shape[i] == self.shape[i]
        #     ), "The shape of pseudo_obs is not correct. Should be {}, but is {}".format(
        #         self.shape, pseudo_obs.shape
        #     )

        print("init pseudo_obs in vamp_prior")
        #eta1, eta2 = pseudo_obs_list
        eta1, eta2, mix_vals, log_e0 = pseudo_obs_list
        if self.use_sparse:
            print('WARNING - fixing pseudo-observations to 0.0')
            self.eta1 = torch.zeros(eta1.shape).to(self.device)
            self.eta1.requires_grad = False
        else:
            self.eta1 = torch.nn.Parameter(torch.DoubleTensor(eta1).to(self.device))
        self.eta2 = torch.nn.Parameter(torch.DoubleTensor(eta2).to(self.device))
        if self.use_mixture_weights:
            self.mix_vals = torch.nn.Parameter(torch.DoubleTensor(mix_vals).to(self.device))
            self.log_e0 = torch.nn.Parameter(torch.DoubleTensor(log_e0).to(self.device))
        else:
            self.mix_vals = torch.zeros(self.K)
            self.mix_vals.requires_grad = False            
            self.log_e0 = torch.zeros(1)
            self.log_e0.requires_grad = False
        
        

    def distribution_mix(self, the_eta1, the_eta2):
        """
        Create a mixture distribution
            the_con = VariationalFamilyNatural.make_positive(the_eta1)
            the_rate = VariationalFamilyNatural.make_positive(the_eta2)
        """
        if self.use_mixture_weights:
            #mix = torch.distributions.Categorical(logits=self.mix_vals)
            mix = torch.distributions.Categorical(probs=torch.nn.functional.softmax(self.mix_vals))
        else:
            mix = torch.distributions.Categorical(torch.ones(self.K,))
        
        if self.family == 'FIXED_VAR_GAMMA':
            the_con = VariationalFamilyNatural.make_positive(the_eta1)
            # use a grid of fixed rates
            n_rates = self.K
            rates = torch.linspace(0.01, 1e4, n_rates).to(self.device)
            # fill the_rate to match shape of the_con
            the_rate = rates[None, :].repeat(the_con.shape[0], 1)
            comp = torch.distributions.Independent(torch.distributions.Gamma(the_con, the_rate), 1)
            raise NotImplementedError("Still working on this!")
        elif self.family == 'MIX':
            # use half gamam and half lognormal
            eta1_m1, eta1_m2 = the_eta1.chunk(2, dim=0)
            eta2_m1, eta2_m2 = the_eta2.chunk(2, dim=0)
            # also handle the mixtures weights
            mix_vals_m1, mix_vals_m2 = self.mix_vals.chunk(2, dim=0)
            # define the first mixture, i.e., Gamma
            the_con_m1 = VariationalFamilyNatural.make_positive(eta1_m1)
            the_rate_m1 = VariationalFamilyNatural.make_positive(eta2_m1)
            mix_m1 = torch.distributions.Categorical(probs=torch.nn.functional.softmax(mix_vals_m1))
            comp_m1 = torch.distributions.Independent(torch.distributions.Gamma(the_con_m1, the_rate_m1), 1)
            m1 = MixtureSameFamily(mix_m1, comp_m1)

            # define the second mixture, i.e., LogNormal
            the_rate_m2 = VariationalFamilyNatural.make_positive(eta2_m2)
            mix_m2 = torch.distributions.Categorical(probs=torch.nn.functional.softmax(mix_vals_m2))
            comp_m2 = torch.distributions.Independent(torch.distributions.LogNormal(eta1_m2, the_rate_m2), 1)
            m2 = MixtureSameFamily(mix_m2, comp_m2)
            return MixtureOfMixtures(m1, m2)
        elif self.family == 'TEST_LOGNORMAL':
            the_rate = VariationalFamilyNatural.make_positive(the_eta2)
            comp = torch.distributions.Independent(torch.distributions.LogNormal(the_eta1, the_rate), 1)
        elif self.family == 'HIGH_GAMMA':
            the_con = VariationalFamilyNatural.make_positive(the_eta1)
            the_rate = VariationalFamilyNatural.make_positive(the_eta2, max=100000)
            comp = torch.distributions.Independent(torch.distributions.Gamma(the_con, the_rate), 1)
        elif self.family == 'lognormal': # actually implemented as Gamma
            # the_rate = VariationalFamilyNatural.make_positive(the_eta2)
            # comp = torch.distributions.Independent(torch.distributions.LogNormal(the_eta1, the_rate), 1)
            the_con = VariationalFamilyNatural.make_positive(the_eta1)
            the_rate = VariationalFamilyNatural.make_positive(the_eta2)
            comp = torch.distributions.Independent(torch.distributions.Gamma(the_con, the_rate), 1)
        elif self.family == 'gamma':
            the_con = VariationalFamilyNatural.make_positive(the_eta1)
            the_rate = VariationalFamilyNatural.make_positive(the_eta2)
            comp = torch.distributions.Independent(torch.distributions.Gamma(the_con, the_rate), 1)
        elif self.family == 'normal':
            the_std = VariationalFamilyNatural.make_positive(the_eta2)
            comp = torch.distributions.Independent(torch.distributions.Normal(the_eta1, the_std), 1)
        return MixtureSameFamily(mix, comp)



    def get_regularization(self):
        """
        Returns the log_prob for the regularization terms
        TODO: perhaps we should regularize the log_eta instead of eta for numerical stability
        """
        # regularize_pseudo = True
        # regularize_mixture_weights = True
        #regularize_pseudo, regularize_mixture_weights = False, False
        regularize_pseudo, regularize_mixture_weights = True, True

        log_prob_pseudo, log_prob_mixture = 0, 0

        if regularize_pseudo:
            # put a Gamma(1, 1) on the pseudo-observations
            pseudo_regularizer = torch.distributions.Gamma(concentration=1, rate=1)
            the_con = VariationalFamilyNatural.make_positive(self.eta1)
            the_rate = VariationalFamilyNatural.make_positive(self.eta2)
            log_prob_pseudo = pseudo_regularizer.log_prob(the_con).to(self.device).sum() + pseudo_regularizer.log_prob(the_rate).to(self.device).sum()
            
            

        if regularize_mixture_weights:
            # Add a regularizer on the mixture weights, a Gamma with mean 1/K and rate 10
            # e0 ~ Gamma(mean=1/k, var= 1/a(K^2)), with a = 10 ---> Gamma(alpha = 10, beta = 10K)
            # \omega ~ Dir(e0, ..., e0)
            # (1) fix e0 to be very small 1/100
            # logprob +=  Dir(make_pos(self.mix_vals) | e0, ..., e0).logprob() 
            #e0 = 1/self.mix_vals.shape[0]
            e0 = VariationalFamilyNatural.make_positive(self.log_e0)
            #e0_regularizer = torch.distributions.Gamma(concentration=1/self.K, rate=10)
            e0_regularizer = torch.distributions.Gamma(concentration=10, rate=10*self.K)
            mix_regularizer = torch.distributions.Dirichlet(concentration=torch.ones_like(self.mix_vals) * e0)
            # use torch softmax on the mix_vals
            simplex = torch.nn.functional.softmax(self.mix_vals)
            log_prob_mixture = mix_regularizer.log_prob(simplex).to(self.device) + e0_regularizer.log_prob(e0).to(self.device)
    
        return log_prob_pseudo + log_prob_mixture


    def log_prob(self, x):
        if self.tag == "_local":
            # x is [n_particles, L, N], log_prob is [n_particles, N]
            # swap the last two dimensions
            log_prob = self.distribution_mix(self.eta1, self.eta2).log_prob(x.permute(0, 2, 1)).to(self.device)
        elif self.tag == "_global":
            # x is [n_particles, D, L], log_prob is [n_particles, D]
            log_prob = self.distribution_mix(self.eta1, self.eta2).log_prob(x).to(self.device)
        else:
            raise ValueError(f"Unrecognized tag: {self.tag}.")
        return log_prob
    

    def distribution(self, the_eta1, the_eta2):
        """Create variational distribution."""
        #the_scale = self.scale()
        # use_natural = False
        # if self.family == "lognormal":
        #     if use_natural == True:
        #         the_location, the_scale = VariationalFamilyNatural.natural_to_lognormal(
        #             the_eta1, the_eta2
        #         )
        #         the_scale = VariationalFamilyNatural.clamp(the_scale)
        #         distribution = torch.distributions.LogNormal(loc=the_location, scale=the_scale)
        #     else:
        #         the_location, the_scale = the_eta1, the_eta2
        #         the_scale = VariationalFamilyNatural.make_positive(the_scale)
        #         distribution = torch.distributions.LogNormal(loc=the_location, scale=the_scale)
        # elif self.family == "gamma":
        if self.family == 'lognormal':
            # ensure both are positive
            the_con = VariationalFamilyNatural.make_positive(the_eta1)
            the_rate = VariationalFamilyNatural.make_positive(the_eta2)
            distribution = torch.distributions.Gamma(concentration=the_con, rate=the_rate)
        elif self.family == 'normal':
            the_mean = the_eta1
            the_std = VariationalFamilyNatural.make_positive(the_eta2)
            distribution = torch.distributions.Normal(loc=the_mean, scale=the_std)
        else:
            raise ValueError(f"Unrecognized prior distribution {self.family}")
        return distribution

    # add properties for location and scale
    @property
    def location(self):
        raise NotImplementedError("Need to map from mean to natural parameters")
        # convert from natural to mean parameterization
        return VariationalFamilyNatural.natural_to_lognormal(self.eta1, self.eta2)[0]

    @property
    def scale(self):
        # convert from natural to mean parameterization and make positive
        raise NotImplementedError("Need to map from mean to natural parameters")
        the_scale = VariationalFamilyNatural.natural_to_lognormal(self.eta1, self.eta2)[1]
        return VariationalFamilyNatural.make_positive(the_scale)


    def sample(self, out_shape):
        """Pick a k then sample from q(z | u_k)
        This does not need rsample since there is no need for grad computation.

        Args:
            out_shape: A list of size 3 with the following num_samples, batch_size, event_size

        Returns:
            A tensor of shape [num_samples, batch_size, event_size]
        """
        raise NotImplementedError("This is a slow implementation. Use a batch version instead.")
        num_samples, batch_size, event_size = out_shape
        assert (
            num_samples == 1
        ), "This is a slow implementation. Use a batch version instead."

        # TODO: Should we use torch.distributions.mixture_same_family.MixtureSameFamily?
        k = torch.randint(high=self.K, size=(batch_size,))
        return self.distribution(self.pseudo_obs[k, :], self.scale()[k, :]).rsample(
            [num_samples]
        )


    def log_prob_old(self, x):
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
        if self.tag == "_local":
            sum_logs = (
                self.distribution(
                    self.eta1[:, None, :, None], self.eta2[:, None, :, None]
                )
                .log_prob(x[None, :])
                .to(self.device)
            )
            # log [(1/K) \sum logq(z_i | X; u_k))]
            # Sum over the latent dimension (all dimensions are IID)
            sum_logs = torch.sum(sum_logs, dim=2)
        elif self.tag == "_global":
            sum_logs = ( # [K, n_particles, D, L]
            self.distribution(
                self.eta1[:, None, None, :], self.eta2[:, None, None, :]
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

        # TODO: add a reguralizer here - i.e., a prior on the self.eta1 and self.eta2, or their transformations...
        regularize = False
        if regularize:
            # one way is to have Exp() for shape (alpha) and inverse gamma for rate (beta)
            reg_dist_1 = torch.distributions.Normal(loc=0, scale=1)
            reg_dist_2 = torch.distributions.Exponential(.1)
            the_loc, the_scale = VariationalFamilyNatural.natural_to_lognormal(self.eta1, self.eta2)
            the_scale = VariationalFamilyNatural.make_positive(the_scale)
            log_prob_reg = reg_dist_1.log_prob(the_loc).sum() + reg_dist_2.log_prob(the_scale).sum()
            log_prob += log_prob_reg/log_prob.shape[1]
        # n_samples (particles) by (n_pseudo)
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
        # np.savetxt(
        #     os.path.join(param_save_dir, f"pseudo_obs_loc{self.tag}"),
        #     self.location.cpu().detach(),
        # )
        # np.savetxt(
        #     os.path.join(param_save_dir, f"pseudo_obs_loc_init{self.tag}"),
        #     self._init_pseudo_obs,
        # )
