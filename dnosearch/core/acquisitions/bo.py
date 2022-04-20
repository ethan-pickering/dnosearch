import numpy as np
from .base import Acquisition
from .us import US, US_LW
from .ivr import IVR, IVR_LW


class EDToBO(Acquisition):
    """A class that repurposes a purely explorative acquisition function
    into one suitable for Bayesian optimization.

    Parameters
    ----------
    model, inputs : see parent class (Acquisition)
    kappa : float
        Parameter balancing exploration and exploitation.

    Attributes
    ----------
    model, inputs, kappa : see Parameters

    Notes
    -----
    This subclass overrides the `evaluate` and `jacobian` methods of 
    the parent class. Repurposing is done as follows:
        a_BO(x) = mu(x) + kappa*a_ED(x)

    """

    def evaluate(self, x):
        x = np.atleast_2d(x)
        mu, _ = self.model.predict_noiseless(x)
        if self.model.normalizer:
            mu = self.model.normalizer.normalize(mu)
        return mu + self.kappa * self._evaluate_explore(x)

    def jacobian(self, x):
        x = np.atleast_2d(x)
        mu_jac, _ = self.model.predictive_gradients(x)
        mu_jac = mu_jac[:,:,0]
        return mu_jac + self.kappa * self._jacobian_explore(x)

    def _evaluate_explore(self, x):
        return super().evaluate(x)

    def _jacobian_explore(self, x):
        return super().jacobian(x)


class US_BO(EDToBO, US):
    """A class that repurposes US into US-BO."""

    def __init__(self, model, inputs, kappa=1):
        super().__init__(model, inputs)
        self.kappa = kappa


class US_LWBO(EDToBO, US_LW):
    """A class that repurposes US-LW into US-LWBO."""

    def __init__(self, model, inputs, likelihood=None, kappa=1):
        super().__init__(model, inputs, likelihood=likelihood)
        self.kappa = kappa


class IVR_BO(EDToBO, IVR):
    """A class that repurposes IVR into US-BO."""

    def __init__(self, model, inputs, kappa=1):
        super().__init__(model, inputs)
        self.kappa = kappa


class IVR_LWBO(EDToBO, IVR_LW):
    """A class that repurposes IVR-LW into IVR-LWBO."""

    def __init__(self, model, inputs, likelihood=None, kappa=1):
        super().__init__(model, inputs, likelihood=likelihood)
        self.kappa = kappa

