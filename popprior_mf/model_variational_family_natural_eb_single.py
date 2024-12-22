from model_vampprior_natural_single import VampPriorNaturalSingle
from model_variational_family_natural import VariationalFamilyNatural


class VariationalFamilyNaturalEBSingle(VariationalFamilyNatural):
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
        super().__init__(device=device, family=family, shape=shape, initial_loc=initial_loc, batch_aware=batch_aware, initial_pseudo_obs=initial_pseudo_obs, **kwargs)

    def _setup_prior(self, initial_pseudo_obs=None, prior_scale=1.0, fixed_scale=None, init_scale=None):
        """Set prior distribution."""
        vp = VampPriorNaturalSingle(
            device=self.device,
            family=self.pseudo_var_family,
            shape=[self.num_pseudo_obs, 0],
            initial_loc=initial_pseudo_obs,
            tag=self.tag,
            use_sparse=self.sparse,
            use_mixture_weights=self.use_mixture_weights,
        )
        return vp