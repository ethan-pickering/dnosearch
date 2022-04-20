import numpy as np
from ..utils import get_standard_normal_pdf_cdf
from .base import Acquisition


class EI(Acquisition):
    """A class for Expected Improvement.

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
        super(EI, self).__init__(model, inputs)
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
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_min, mu, std)
        ei = std * (u * cdf + pdf)
        return -ei

    def jacobian(self, x):
        x = np.atleast_2d(x)
        y_min = np.min(self.model.Y, axis=0)
        mu, var = self.model.predict_noiseless(x)
        if self.model.normalizer:
            y_min = self.model.normalizer.normalize(y_min)
            mu = self.model.normalizer.normalize(mu)
            var /= self.model.normalizer.std**2
        std = np.sqrt(var)
        mu_jac, var_jac = self.model.predictive_gradients(x)
        mu_jac = mu_jac[:,:,0]
        std_jac = var_jac / (2*std)
        mu += self.zeta
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_min, mu, std)
        ei_jac = std_jac * pdf - cdf * mu_jac
        return -ei_jac



