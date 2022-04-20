import numpy as np
from .base import Acquisition, AcquisitionWeighted


class US(Acquisition):
    """A class for Uncertainty Sampling.

    Parameters
    ----------
    model, inputs : see parent class (Acquisition)

    Attributes
    ----------
    model, inputs : see Parameters

    """

    def evaluate(self, x, vector=True):
        if vector:
            dim = np.shape(self.inputs.domain)[0]
            x = np.atleast_2d(x)
            unique_pts = int(np.size(x)/dim) 
            x = x.reshape(unique_pts,dim)
            _, var = self.model.predict_noiseless(x)
            if self.model.normalizer:
                var /= self.model.normalizer.std**2
            w = self.get_weights(x, vector=True)
            value = np.sum(-np.log(var * w))

        else:
            x = np.atleast_2d(x)
            _, var = self.model.predict_noiseless(x)
            if self.model.normalizer:
                var /= self.model.normalizer.std**2
            w = self.get_weights(x, vector)
            #return - var * w
            value = -np.log(var * w)
        return value
    
    def jacobian(self, x, vector=True):
        if vector:
            dim = np.shape(self.inputs.domain)[0]
            x = np.atleast_2d(x)
            unique_pts = int(np.size(x)/dim) 
            x = x.reshape(unique_pts,dim)
            _, var = self.model.predict_noiseless(x)
            if self.model.normalizer:
                var /= self.model.normalizer.std**2
            _, var_jac = self.model.predictive_gradients(x)
            w, w_jac = self.get_weights(x, False), self.get_weights_jac(x, False)
            #return - (var_jac * w + var * w_jac)
            value = - (var_jac / var + w_jac / w)
            value = value.reshape(1,unique_pts*dim)
            
        else:
            x = np.atleast_2d(x)
            _, var = self.model.predict_noiseless(x)
            if self.model.normalizer:
                var /= self.model.normalizer.std**2
            _, var_jac = self.model.predictive_gradients(x)
            w, w_jac = self.get_weights(x, False), self.get_weights_jac(x, False)
            #return - (var_jac * w + var * w_jac)
            value = - (var_jac / var + w_jac / w)
        return value

class US_LW(AcquisitionWeighted, US):
    """A class for Likelihood-Weighted Uncertainty Sampling.

    Parameters
    ----------
    model, inputs, likelihood : see parent class (AcquisitionWeighted)

    Attributes
    ----------
    model, inputs, likelihood : see Parameters

    """

    def __init__(self, model, inputs, surrogate_model, likelihood=None):
        super().__init__(model, inputs, surrogate_model, likelihood=likelihood)

