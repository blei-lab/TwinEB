
from model_variational_family_natural import VariationalFamilyNatural
from model_vampprior_natural import VampPriorNatural


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import MultivariateNormal
import numpy as np

# Basic flow layer (planar flow as an example)
class PlanarFlow_old(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        breakpoint()
        dot = torch.matmul(self.w, z.t()) + self.b
        f_z = z + self.u * torch.tanh(dot).t()
        psi = (1 - torch.tanh(dot)**2) * self.w
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u.t()))
        return f_z, torch.log(det_jacobian + 1e-8)

class PlanarFlow_mid(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        # z is shape (n_obs, latent_dim)
        dot = torch.matmul(z, self.w) + self.b  # shape (n_obs,)
        f_z = z + self.u * torch.tanh(dot).unsqueeze(1)  # shape (n_obs, latent_dim)

        # Compute psi = (1 - tanh(dot)^2) * w
        psi = (1 - torch.tanh(dot) ** 2).unsqueeze(1) * self.w  # shape (n_obs, latent_dim)

        # Compute the determinant of the Jacobian for each sample
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u))  # shape (n_obs,)

        return f_z, torch.log(det_jacobian + 1e-8)  # log_det_jacobian is shape (n_obs,)



# Normalizing Flow using multiple planar flows
class NormalizingFlow_old(nn.Module):
    def __init__(self, dim, flow_length=3):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(flow_length)])
        self.base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    def _innter_compute(self, z):
        log_det_jacobian = 0
        for flow in self.flows:
            z, ldj = flow(z)
            log_det_jacobian += ldj
        return z, log_det_jacobian

    def forward(self, z):
        breakpoint()
        # dim: a, b, c
        # a: the number of particles
        # b: the number of latent variables
        # c: the dimension of latent variables
        # for each of the particles, compute this function and return the result
        particle_results = torch.stack([self._innter_compute(z[i]) for i in range(z.shape[0])])
        return particle_results[:, 0, :], particle_results[:, 1, :]

    def log_prob(self, z):
        transformed_z, log_det_jacobian = self.forward(z)
        base_log_prob = self.base_dist.log_prob(transformed_z)
        return base_log_prob + log_det_jacobian


class NormalizingFlow_mid(nn.Module):
    def __init__(self, dim, flow_length=3):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(flow_length)])
        self.base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    def forward(self, z):
        breakpoint()
        log_det_jacobian = torch.zeros(z.size(0), )  # Initialize log-det Jacobian (n_obs,)
        for flow in self.flows:
            z, ldj = flow(z)
            log_det_jacobian += ldj  # Accumulate log-determinant (n_obs,)
        return z, log_det_jacobian

    def log_prob(self, z):
        transformed_z, log_det_jacobian = self.forward(z)
        base_log_prob = self.base_dist.log_prob(transformed_z)  # shape (n_obs,)
        return base_log_prob + log_det_jacobian  # shape (n_obs,)



class PlanarFlow(nn.Module):
    def __init__(self, dim, device=None):
        super(PlanarFlow, self).__init__()
        self.device = device
        self.u = nn.Parameter(torch.randn(dim).to(device))
        self.w = nn.Parameter(torch.randn(dim).to(device))
        self.b = nn.Parameter(torch.randn(1).to(device))

    def forward(self, z):
        # z is of shape (n_particles, n_obs, latent_dim)
        dot = torch.matmul(z, self.w) + self.b  # shape (n_particles, n_obs)
        f_z = z + self.u * torch.tanh(dot).unsqueeze(2)  # shape (n_particles, n_obs, latent_dim)

        # Compute psi = (1 - tanh(dot)^2) * w
        psi = (1 - torch.tanh(dot)**2).unsqueeze(2) * self.w  # shape (n_particles, n_obs, latent_dim)

        # Compute the determinant of the Jacobian for each sample
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u))  # shape (n_particles, n_obs)

        return f_z, torch.log(det_jacobian + 1e-8)  # log_det_jacobian is shape (n_particles, n_obs)


class NormalizingFlow(nn.Module):
    def __init__(self, dim, flow_length=3, device=None, do_swap=False):
        super(NormalizingFlow, self).__init__()
        self.device = device
        self.do_swap = do_swap
        self.flows = nn.ModuleList([PlanarFlow(dim, device=self.device) for _ in range(flow_length)])
        self.base_dist = MultivariateNormal(torch.zeros(dim, device=self.device), torch.eye(dim, device=self.device))

    # Define a function to get the u, w, and b params per flow
    def get_params(self):
        # Just extract the current values (and not the parameters themselves)
        u = [flows.u.clone().detach().cpu() for flows in self.flows]
        w = [flows.w.clone().detach().cpu() for flows in self.flows]
        b = [flows.b.clone().detach().cpu() for flows in self.flows]
        return u, w, b
    
    # Define a function to set the u, w, and b params per flow (these should be the initial values of the params)
    def set_params(self, u, w, b):
        for i, flow in enumerate(self.flows):
            # These are params, so redeclare them as nn.Parameters, with the initial values 
            flow.u = nn.Parameter(u[i])
            flow.w = nn.Parameter(w[i])
            flow.b = nn.Parameter(b[i])


    def init_pseudo_obs(self, pseudo_obs):
        u, w, b = pseudo_obs
        self.set_params(u, w, b)


    def forward(self, z):
        # put it on cuda
        log_det_jacobian = torch.zeros(z.size(0), z.size(1)).to(self.device)  # Initialize log-det Jacobian (n_particles, n_obs)
        for flow in self.flows:
            z, ldj = flow(z)
            log_det_jacobian += ldj  # Accumulate log-determinant (n_particles, n_obs)
        return z, log_det_jacobian

    def log_prob(self, z):
        if self.do_swap:
            # print(f'Swapping! Original shape: {z.shape}')
            # swap the last two dimensions
            # z shape is [n_particles, latent_dim, n_obs (batch_size)]
            z = z.permute(0, 2, 1)
        transformed_z, log_det_jacobian = self.forward(z)
        # Reshape transformed_z to batch over particles and obs (n_particles * n_obs, latent_dim)
        if self.do_swap:
            transformed_z_flat = transformed_z.reshape(-1, transformed_z.size(2))
        else:
            transformed_z_flat = transformed_z.view(-1, transformed_z.size(2))
        # Base log probability (repeating for each particle)
        base_log_prob = self.base_dist.log_prob(transformed_z_flat).view(z.size(0), z.size(1))
        #print(f"Base log prob shape: {base_log_prob.shape}")
        return base_log_prob + log_det_jacobian  # shape (n_particles, n_obs)






class VariationalFamilyNaturalEBNormalizingFlow(VariationalFamilyNatural):
    """
    Implementation of the Emprical Bayes prior for Variational Family
    Puts a Vamprior on a per-datapoint latent variable
    """

    def __init__(
        self,
        device,
        family,
        shape,
        num_pseudo_obs,
        initial_loc=None,
        initial_pseudo_obs=None,
        batch_aware=True,
        pseudo_var_family=None,
        sparse=False,
        use_mixture_weights=True,
        use_normalizing_flow_prior=False,
        **kwargs,
    ):
        """Initialize variational family.

        Args:
            device: Device where operations take place.
            family: A string representing the variational family, either "normal" or
            "lognormal".
            shape: [latent_dim, num_datapoints]
            initial_loc: An optional tensor with shape `shape`, denoting the initial
            location of the variational family.
            initial_pseudo_obs: An optional tensor with shape `[K, L]`, denoting the starting value for pseudo observations.
            batch_aware: A boolean indicating whether the variational family is batch-aware (important for debuggig)
        """
        self.sparse = sparse
        self.pseudo_var_family = pseudo_var_family
        self.tag = '_local' if batch_aware else '_global'
        self.num_pseudo_obs = num_pseudo_obs
        self.use_mixture_weights = use_mixture_weights,
        self.use_normalizing_flow_prior = use_normalizing_flow_prior
        super().__init__(device=device, family=family, shape=shape, initial_loc=initial_loc, batch_aware=batch_aware, initial_pseudo_obs=initial_pseudo_obs, **kwargs)

        #self.prior = self._set_prior(initial_pseudo_obs)

    def _setup_prior(self, initial_pseudo_obs=None, prior_scale=1.0, fixed_scale=None, init_scale=None):
        """Set prior distribution."""
        #raise NotImplementedError("The LOG SCALE SHOULD BE IDENTICAL TO VARIATIONAL FAMILTIES LOG SCALE")
        # The log.scale cannot be shared unless it is a scalar and shared among the variational posterior and this prior or we're using an encoder. Otherwise, each obs has its own scale with num_datapiont >> num_pseudo_obs
        if self.use_normalizing_flow_prior:
            print("Using Normalizing Flow!!!")
            flow_length = 10
            latent_dim = self.shape[0] if self.batch_aware else self.shape[1]
            the_flow = NormalizingFlow(latent_dim, flow_length, self.device, do_swap=self.batch_aware)
            the_flow.to(self.device)
            return the_flow
        else:
            vp = VampPriorNatural(
                device=self.device,
                family=self.pseudo_var_family,
                shape=[self.num_pseudo_obs, self.shape[0] if self.batch_aware else self.shape[1]],
                #log_scale=self.log_scale,
                initial_loc=initial_pseudo_obs,
                tag=self.tag,
                use_sparse=self.sparse,
                use_mixture_weights=self.use_mixture_weights,
            )
            return vp
        