import numpy as np
import scipy
from ..utils import get_standard_normal_pdf_cdf
from .base import Acquisition


class PI(Acquisition):
    """A class for Probability of Improvement.

    Parameters
    ----------
    model, inputs : see parent class (Acquisition)
    zeta : float
        Jitter parameter balancing exploration and exploitation.

    Attributes
    ----------
    model, inputs, zeta : see Parameters

    """

    def __init__(self, model, inputs, zeta=0.01):
        super(PI, self).__init__(model, inputs)
        self.zeta = zeta

    def evaluate(self, x):
        x = np.atleast_2d(x)
        y_min = np.min(self.model.Y, axis=0)
        mu, var = self.model.predict_noiseless(x)
        if self.model.normalizer:
            y_min = self.model.normalizer.normalize(y_min)
            mu = self.model.normalizer.normalize(mu)
            var /= self.model.normalizer.std**2
        std = np.sqrt(var)
        mu += self.zeta
        cdf = scipy.stats.norm.cdf(y_min, mu, std)
        return -cdf

    def jacobian(self, x):
        x = np.atleast_2d(x)
        y_min = np.min(self.model.Y, axis=0)
        mu, var = self.model.predict_noiseless(x)
        if self.model.normalizer:
            y_min = self.model.normalizer.normalize(y_min)
            mu = self.model.normalizer.normalize(mu)
            var /= self.model.normalizer.std**2
        std = np.sqrt(var)
        mu += self.zeta
        mu_jac, var_jac = self.model.predictive_gradients(x)
        mu_jac = mu_jac[:,:,0]
        std_jac = var_jac / (2*std)
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_min, mu, std)
        cdf_jac = - pdf * (mu_jac + std_jac * u) / std
        return -cdf_jac


