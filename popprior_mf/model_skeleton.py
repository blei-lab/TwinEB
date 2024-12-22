"""
    The basic classes for distributions.

    Distribution:
        - StandAloneDistribution: for special purpose distributions, e.g., priors
        - ObservedDistribution: for the generative model that will be equiped with VI
        - VariationalFamilyDistribution: for the variational family distribution

    NB: To define a Generative model and its variational family in this framework:
    1. Define the prior P(Z) as the prior in the [VariationalFamilyDistribution] instance.
    2. Define the likelihood P(X|Z) in the `distribution()` method of the [ObservedDistribution] instance.
    3. Define the variational family q(Z|X) in the `__init__()` method of the [ObservedDistribution] instance by  \
        equiping the each latent with the appropriate [VariationalFamilyDistribution] instance.

    @Author: Sohrab Salehi sohrab.salehi@columbia.edu
"""


import torch
from abc import ABC, abstractmethod


class Distribution(torch.nn.Module, ABC):
    """
    Abstract class for a distribution.
    """

    @abstractmethod
    def save_txt(self):
        """Save the model params to file."""
        pass
    


class StandAloneDistribution(Distribution):
    """
    Abstract class that defines the interface for a distribution that can be sampled from.
    """
    @abstractmethod
    def sample(self):
        """Returns a sample from the distribution."""
        pass

    @abstractmethod
    def log_prob(self):
        """Returns the log probability of the sample."""
        pass

    @abstractmethod
    def distribution(self):
        """Returns a disribution object that can be sampled from."""
        pass

    @abstractmethod
    def scale(self):
        """Special treatment for the scale paramter."""
        pass



class ObservedDistribution(Distribution):
    """
    Absrtact class for a distribution with observed data, to be equiped with one or many variation families and understand ELBO.
    """

    # Methods related to ELBO
    @abstractmethod
    def get_data_log_likelihood(self):
        """Returns the log likelihood of the data."""
        pass

    @abstractmethod
    def get_samples(self):
        """Returns a sample from the variational distribution."""
        pass

    @abstractmethod
    def get_log_prior(self):
        """Returns the log prior of the latent sampels."""
        pass

    # Methods related to evaluation
    @abstractmethod
    def generate_data(self):
        """Simluates the observables from the model."""
        pass

    @abstractmethod
    def compute_heldout_loglikelihood(self):
        """Computes the heldout log likelihood."""
        pass

    @abstractmethod
    def distribution(self):
        """Returns a distribution object to compute the loglikelihood of the data."""
        pass



class VariationalDistribution(Distribution):
    """
    Absract class for a variational distribution for a latent variable.
    """

    @abstractmethod
    def get_entropy(self):
        """Computes the entropy of the variational distribution."""
        pass

    @abstractmethod
    def get_log_prior(self):
        """
        Compute the log prior of the variational distribution.
        """
        pass

    @abstractmethod
    def sample(self):
        """Returns a sample from the variational family."""
        pass

    @abstractmethod
    def distribution(self):
        """Returns a disribution object that can be sampled from."""
        pass

    @abstractmethod
    def scale(self):
        """Special treatment for the scale paramter."""
        pass
    

    @abstractmethod
    def mean_parameters(self):
        """Returns parameters designated as location parameters for location-scale distributions."""
        pass

    @abstractmethod
    def variance_parameters(self):
        """Returns parameters designated as scale parameters for location-scale distributions."""
        pass


class VariableBank(ABC):
    """
    Absract class for a variational distributions that need a dependent prior (e.g., a population prior) that needs access to their variational parameters. 
    """

    @abstractmethod
    # Define n_samples as int
    def get_variable(self, n_samples: int):
        """Returns a random subset of the variational parameters of size n_samples."""
        pass

    