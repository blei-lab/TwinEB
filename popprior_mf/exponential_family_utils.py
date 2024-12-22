# """
#     Exponential Family parameterization utils
# """

import numpy as np
import torch

def _compute_lognormal_mean(location, scale):
    """
    Computes the mean of the lognormal distribution, given its location and scale (s.d).
    Mean of the lognormal: exp(mu + sigma^2/2)
    """
    if isinstance(location, torch.Tensor):
        assert torch.all(scale > 0), "scale has negative elements"
        return torch.exp(location + 0.5 * (scale))
    else:
        assert np.all(scale > 0), "scale has negative elements"
        #return np.exp(location + 0.5 * (scale**2))
        # print('Warning - asssuming scale^2 instead of scale is retained')
        # TODO: also chanage the initialization
        return np.exp(location + 0.5 * (scale))


def _compute_gamma_mean(concentration, rate):
    """
    Computes the mean of the gamma distribution, given its concentration and rate.
    Mean of the gamma: concentration/rate
    """
    if isinstance(concentration, torch.Tensor):
        assert torch.all(concentration > 0), "concentration has negative elements"
        assert torch.all(rate > 0), "rate has negative elements"
        return concentration / rate
    else:
        assert np.all(concentration > 0), "concentration has negative elements"
        assert np.all(rate > 0), "rate has negative elements"
        return concentration / rate


def _compute_lognormal_mean_natural(eta1, ln_minus_eta2, exponential=False):
    """
    Computes the mean of the lognormal distribution, given its natural parameters.

    E[X] = exp(exp(-\phi) * (0.5 * \eta_1 + 0.25)), where \phi = ln_minus_eta2
    """
    raise NotImplementedError("This is not numerically stable")
    pos_func = torch.exp if exponential else torch.nn.functional.softplus
    sigma2 = pos_func(torch.log(torch.tensor(.5)) - ln_minus_eta2)
    sigma2 = torch.clamp(sigma2, min=1e-8, max=1e3)
    the_mean = torch.exp(sigma2 * (0.5 * eta1 + 0.5))
    return the_mean.detach().cpu().numpy()

