"""
    The Probabilistic Principal Component Analysis (PPCA) with an Emprical Bayes prior using variational inference
    
    Is a more true populatin prior, that is, instead of a VampPrior, it uses a mixture of the q's 
    
    Author: De-identified Author
"""

import torch
from model_pmf import PMF
from model_variational_family_eb_plus import VariationalFamilyEBPlus


class PMFEBPlus(PMF):
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
        num_pseudo_obs,
        annealing_factor=1.0,
        row_prior_concentration=0.1,
        row_prior_rate=0.3,
        column_prior_concentration=0.1,
        column_prior_rate=0.3,
        prior_family=None,
        var_fam_scale=None,
        var_fam_init_scale=None,
        sparse=False,
        var_family="lognormal",
    ):
        self.num_pseudo_obs = num_pseudo_obs
        self.sparse = sparse
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
            column_prior_concentration=column_prior_concentration,
            column_prior_rate=column_prior_rate,
            prior_family=prior_family,
            var_fam_scale=var_fam_scale,
            var_fam_init_scale=var_fam_init_scale,
            var_family=var_family
        )

        assert (
            num_pseudo_obs > 0
        ), f"num_pseudo_obs must be greater than 0, but got {num_pseudo_obs}."

    def _setup_row_vars(
        self,
        init_loc=None,
        concentration=0.1,
        rate=0.3,
        prior_family=None,
        fixed_scale=None,
        init_scale=None,
        family=None
    ):
        self.qu_distribution = VariationalFamilyEBPlus(
            device=self.device,
            family=family,
            shape=[self.latent_dim, self.num_datapoints],
            num_pseudo_obs=self.num_pseudo_obs,
            batch_aware=True,
            fixed_scale=fixed_scale,
            init_scale=init_scale,
            use_sparse=self.sparse,
        )

    def init_pseudo_obs(self, init_location):
        """Initialize pseudo observations."""
        self.qu_distribution.prior.init_pseudo_obs(init_location)

    @torch.no_grad()
    def save_txt(self, param_save_dir, **kwargs):
        """Save the model params to file

        Args:
            param_save_dir: The directory to save the model params to.

        Returns:
            None
        """
        super().save_txt(param_save_dir, **kwargs)
        self.qu_distribution.prior.save_txt(param_save_dir)
