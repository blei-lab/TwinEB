"""
    The Probabilistic Principal Component Analysis (PPCA) with an Emprical Bayes prior using variational inference
    With EB prior only on the columns
    
    @Author: Sohrab Salehi (sohrab.salehi@columbia.edu)
"""

import torch
from model_pmf import PMF
from model_variational_family_eb_plus import VariationalFamilyEBPlus


class PMFEBColumnPlus(PMF):
    """Object to hold model parameters and approximate ELBO."""

    def __init__(
        self,
        device,
        num_datapoints,
        data_dim,
        latent_dim,
        num_samples,
        print_steps,
        summary_writer,
        num_pseudo_obs_global,
        annealing_factor=1.0,
        row_prior_concentration=.1,
        row_prior_rate=.3,
    ):
        self.num_pseudo_obs_global = num_pseudo_obs_global
        super().__init__(
            device=device,
            num_datapoints=num_datapoints,
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_samples=num_samples,
            print_steps=print_steps,
            summary_writer=summary_writer,
            annealing_factor=annealing_factor,
            row_prior_concentration=row_prior_concentration,
            row_prior_rate=row_prior_rate,
        )

        assert (
            num_pseudo_obs_global > 0
        ), f"num_pseudo_obs_global must be greater than 0, but got {num_pseudo_obs_global}."
        

    def _setup_column_vars(self, init_loc=None, concentration=.1, rate=.3):
        # print('Setting up col vars in PMFEBColumn')
        self.qv_distribution = VariationalFamilyEBPlus(
            device=self.device,
            family="lognormal",
            shape=[self.data_dim, self.latent_dim],
            num_pseudo_obs=self.num_pseudo_obs_global,
            batch_aware=False,
        )

    def init_pseudo_obs_global(self, init_location):
        """Initialize pseudo observations."""
        self.qv_distribution.prior.init_pseudo_obs(init_location)

    @torch.no_grad()
    def save_txt(self, param_save_dir, **kwargs):
        """Save the model params to file

        Args:
            param_save_dir: The directory to save the model params to.

        Returns:
            None
        """
        super().save_txt(param_save_dir, **kwargs)
        self.qv_distribution.prior.save_txt(param_save_dir)