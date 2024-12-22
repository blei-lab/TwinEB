"""
    Implements the Variational Mixture of Posteriors Prior (VampPrior) model.

    @Author: Sohrab Salehi (sohrab.salehi@columbia.edu)
"""


import os
import torch
import numpy as np
from model_skeleton import StandAloneDistribution
from model_variational_family_natural import VariationalFamilyNatural
from torch.distributions import MixtureSameFamily

class VampPriorNaturalSingle(StandAloneDistribution):
    """
    A mixture of singleton, same-family
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
        self.K = shape[0]
        self.use_sparse = use_sparse
        
        if self.use_sparse:
            print('WARNING - fixing pseudo-observations to 0.0')
            self.eta1 = torch.zeros(shape).to(self.device)
            self.eta1.requires_grad = False
        else:
            if initial_loc is None:
                self.eta1 = torch.nn.Parameter(torch.ones(self.K))
            else:
                raise NotImplementedError("Need to map from mean to natural parameters")
                self.eta1 = torch.nn.Parameter(torch.FloatTensor(initial_loc))
        self.eta2 = torch.nn.Parameter(torch.ones(self.K))
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
        
        

    def distribution(self, the_eta1, the_eta2):
        """
        Create a mixture distribution
            the_con = VariationalFamilyNatural.make_positive(the_eta1)
            the_rate = VariationalFamilyNatural.make_positive(the_eta2)
        """
        if self.use_mixture_weights:
            mix = torch.distributions.Categorical(logits=self.mix_vals)
        else:
            mix = torch.distributions.Categorical(logits=torch.ones(self.K,))
        
        pos_eta2 = VariationalFamilyNatural.make_positive(the_eta2)
        if self.family == 'TEST_LOGNORMAL':
            comp = torch.distributions.Independent(torch.distributions.LogNormal(the_eta1, pos_eta2), 0)
        elif self.family == 'lognormal':
            # comp = torch.distributions.Independent(torch.distributions.LogNormal(the_eta1, pos_eta2), 1)
            the_con = VariationalFamilyNatural.make_positive(the_eta1)
            comp = torch.distributions.Independent(torch.distributions.Gamma(the_con, pos_eta2), 0)
        elif self.family == 'gamma':
            the_con = VariationalFamilyNatural.make_positive(the_eta1)
            comp = torch.distributions.Independent(torch.distributions.Gamma(the_con, pos_eta2), 0)
        elif self.family == 'normal':
            comp = torch.distributions.Independent(torch.distributions.Normal(the_eta1, pos_eta2), 0)
        return MixtureSameFamily(mix, comp)


    def get_regularization(self):
        """
        Returns the log_prob for the regularization terms
        """
        log_prob_pseudo, log_prob_mixture = 0, 0

        #regularize_pseudo, regularize_mixture_weights = False, False
        regularize_pseudo, regularize_mixture_weights = True, True
        
        if regularize_pseudo:
            # put a Gamma(1, 1) on the pseudo-observations
            pseudo_regularizer = torch.distributions.Gamma(concentration=1, rate=1)
            the_con = VariationalFamilyNatural.make_positive(self.eta1)
            the_rate = VariationalFamilyNatural.make_positive(self.eta2)
            log_prob_pseudo = pseudo_regularizer.log_prob(the_con).to(self.device).sum() + pseudo_regularizer.log_prob(the_rate).to(self.device).sum()

        if regularize_mixture_weights:
            e0 = VariationalFamilyNatural.make_positive(self.log_e0)
            e0_regularizer = torch.distributions.Gamma(concentration=10, rate=10*self.K)
            #e0_regularizer = torch.distributions.Gamma(concentration=1/self.K, rate=10)
            mix_regularizer = torch.distributions.Dirichlet(concentration=torch.ones_like(self.mix_vals) * e0)
            # use torch softmax on the mix_vals
            simplex = torch.nn.functional.softmax(self.mix_vals)
            log_prob_mixture = mix_regularizer.log_prob(simplex).to(self.device) + e0_regularizer.log_prob(e0).to(self.device)

        return log_prob_pseudo + log_prob_mixture

    def log_prob(self, x):
        """
        Output:
            the same size as x
        """
        log_prob = self.distribution(self.eta1, self.eta2).log_prob(x).to(self.device)

        return log_prob


    # add properties for location and scale
    @property
    def location(self):
        raise NotImplementedError("Need to map from mean to natural parameters")

    @property
    def scale(self):
        raise NotImplementedError("Need to map from mean to natural parameters")
        

    def sample(self, out_shape):
        """Pick a k then sample from q(z | u_k)
        This does not need rsample since there is no need for grad computation.

        Args:
            out_shape: A list of size 3 with the following num_samples, batch_size, event_size

        Returns:
            A tensor of shape [num_samples, batch_size, event_size]
        """
        raise NotImplementedError("This is a slow implementation. Use a batch version instead.")
        


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
