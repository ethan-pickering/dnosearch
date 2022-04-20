import numpy as np
from .base import Acquisition, AcquisitionWeighted


class LCB(Acquisition):
    """A class for Lower Confidence Bound.

    Parameters
    ----------
    model, inputs : see parent class (Acquisition)
    kappa : float
        Parameter balancing exploration and exploitation.

    Attributes
    ----------
    model, inputs, kappa : see Parameters

    """

    def __init__(self, model, inputs, kappa=1):
        super(LCB, self).__init__(model, inputs)
        self.kappa = kappa

    def evaluate(self, x):
        x = np.atleast_2d(x)
        mu, var = self.model.predict_noiseless(x)
        if self.model.normalizer:
            mu = self.model.normalizer.normalize(mu)
            var /= self.model.normalizer.std**2
        std = np.sqrt(var)
        w = self.get_weights(x)
        lcb = mu - self.kappa * std * w
        return lcb

    def jacobian(self, x):
        x = np.atleast_2d(x)
        mu, var = self.model.predict_noiseless(x)
        if self.model.normalizer:
            mu = self.model.normalizer.normalize(mu)
            var /= self.model.normalizer.std**2
        std = np.sqrt(var)
        mu_jac, var_jac = self.model.predictive_gradients(x)
        mu_jac = mu_jac[:,:,0]
        std_jac = var_jac / (2*std)
        w = self.get_weights(x)
        w_jac = self.get_weights_jac(x)
        lcb_jac = mu_jac - self.kappa * (std_jac*w + std*w_jac)
        return lcb_jac


class LCB_LW(AcquisitionWeighted, LCB):
    """A class for Likelihood-Weighted Lower Confidence Bound.

    Parameters
    ----------
    model, inputs : see parent class (AcquisitionWeighted)
    kappa : see parent class (LCB)

    Attributes
    ----------
    model, inputs, kappa : see Parameters

    """

    def __init__(self, model, inputs, likelihood=None, kappa=1):
        super().__init__(model, inputs, likelihood=likelihood)
        self.kappa = kappa


