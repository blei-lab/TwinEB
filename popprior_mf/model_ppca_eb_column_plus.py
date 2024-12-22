"""
    The Probabilistic Principal Component Analysis (PPCA) with an Emprical Bayes prior using variational inference
    with just an EB prior on the global paramters.
    
    Author: De-identified Author
"""

import torch
from model_ppca import PPCA
from model_variational_family_eb_plus import VariationalFamilyEBPlus

class PPCAEBColumnPlus(PPCA):
    """Object to hold model parameters and approximate ELBO."""

    def __init__(
        self,
        device,
        num_datapoints,
        data_dim,
        latent_dim,
        stddv_datapoints,
        num_samples,
        print_steps,
        summary_writer,
        num_pseudo_obs_global,
        annealing_factor=1.0,
        init_row_loc=None,
        init_col_loc=None,
        row_prior_scale=1.0,
    ):
        print("num_pseudo_obs_global: ", num_pseudo_obs_global)
        self.num_pseudo_obs_global = num_pseudo_obs_global
        super().__init__(
            device=device,
            num_datapoints=num_datapoints,
            data_dim=data_dim,
            latent_dim=latent_dim,
            stddv_datapoints=stddv_datapoints,
            num_samples=num_samples,
            print_steps=print_steps,
            summary_writer=summary_writer,
            annealing_factor=annealing_factor,
            init_row_loc=init_row_loc,
            init_col_loc=init_col_loc,
            row_prior_scale=row_prior_scale,
        )

        assert (
            num_pseudo_obs_global > 0
        ), f"num_pseudo_obs_global must be greater than 0, but got {num_pseudo_obs_global}."

    def _setup_column_vars(self, init_loc=None, scale=1.0):
        print('Setting up column vars...')
        self.qw_distribution = VariationalFamilyEBPlus(
            device=self.device,
            family="normal",
            shape=[self.data_dim, self.latent_dim],
            num_pseudo_obs=self.num_pseudo_obs_global,
            batch_aware=False,
            initial_loc=init_loc,
            prior_scale=scale,
        )
        print('Setting up column vars...done')

    def init_pseudo_obs_global(self, init_location):
        """Initialize pseudo observations."""
        self.qw_distribution.prior.init_pseudo_obs(init_location)

    def init_pseudo_obs(self, init_location):
        """Initialize pseudo observations."""
        self.qz_distribution.prior.init_pseudo_obs(init_location)


    @torch.no_grad()
    def save_txt(self, param_save_dir, **kwargs):
        """Save the model params to file

        Args:
            param_save_dir: The directory to save the model params to.

        Returns:
            None
        """
        super().save_txt(param_save_dir, **kwargs)
        self.qw_distribution.prior.save_txt(param_save_dir)
