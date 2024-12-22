"""
    The Probabilistic Principal Component Analysis (PPCA) with an Emprical Bayes prior using variational inference
    
    Author: De-identified Author
"""

import torch
from model_ppca_natural import PPCANatural
from model_variational_family_natural_eb_single import VariationalFamilyNaturalEBSingle

# TODO: fix the scale for row vars to a scalar to share with the VampPrior

class PPCANaturalEBSingle(PPCANatural):
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
        num_pseudo_obs,
        annealing_factor=1.0,
        init_row_loc=None,
        init_col_loc=None,
        row_prior_scale=1.0,
        column_prior_scale=1.0,
        var_family="normal", 
        pseudo_var_family="normal",
        sparse=False,
        use_mixture_weights=True,
        regularize_prior=False,
    ):
        assert pseudo_var_family == 'normal', f"pseudo_var_family must be normal, but got {pseudo_var_family}"
        self.pseudo_var_family = pseudo_var_family
        self.num_pseudo_obs = num_pseudo_obs
        self.sparse = sparse
        self.use_mixture_weights = use_mixture_weights
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
            var_family=var_family,
            column_prior_scale=column_prior_scale,
            regularize_prior=regularize_prior,
        )
        assert (
            num_pseudo_obs > 0
        ), f"num_pseudo_obs must be greater than 0, but got {num_pseudo_obs}."

    def _setup_row_vars(self, init_loc=None, scale=1.0, prior_family=None, fixed_scale=None, init_scale=None, family=None):
        self.qz_distribution = VariationalFamilyNaturalEBSingle(
            device=self.device,
            family="normal",
            shape=[self.latent_dim, self.num_datapoints],
            num_pseudo_obs=self.num_pseudo_obs,
            batch_aware=True,
            initial_loc=init_loc,
            prior_scale=scale,
            pseudo_var_family=self.pseudo_var_family,
            sparse=self.sparse,
            use_mixture_weights=self.use_mixture_weights,
        )

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
        #self.qz_distribution.prior.save_txt(param_save_dir)
