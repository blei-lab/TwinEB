from model_population_prior import PopulationPrior
from model_variational_family import VariationalFamily


class VariationalFamilyEBPlus(VariationalFamily):
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
        use_sparse=False,
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
        self.tag = '_local' if batch_aware else '_global'
        self.num_pseudo_obs = num_pseudo_obs
        self.use_sparse = use_sparse
        super().__init__(device=device, family=family, shape=shape, initial_loc=initial_loc, batch_aware=batch_aware, initial_pseudo_obs=initial_pseudo_obs, **kwargs)

    def _setup_prior(self, initial_pseudo_obs=None, prior_scale=1.0):
        """Set prior distribution."""
        pp = PopulationPrior(
            device=self.device,
            family=self.family,
            shape=[self.num_pseudo_obs, self.shape[0] if self.batch_aware else self.shape[1]],
            log_scale=prior_scale,
            pseudo_obs=self.location,
            the_scale=self.log_scale,
            tag=self.tag,
            use_sparse=self.use_sparse,
        )
        return pp

    # def get_variable(self, n_samples):
    #     """Returns a random subset of the variational parameters of size n_samples."""
    #     indices = torch.randperm(self.num_pseudo_obs)[:n_samples]
    #     return self.location[indices], self.log_scale[indices]