import numpy as np
from .base import Acquisition, AcquisitionWeighted
from ..utils import grid_nint, add_xnew, jacobian_fdiff


class IVRInt(Acquisition):
    """A class for IVR computed by numerical integration.

    Parameters
    ----------
    model, inputs : see parent class (Acquisition)
    ngrid : int
        Number of grid points in each direction

    Attributes
    ----------
    model, inputs, ngrid : see Parameters
    pts : array  
        Grid points as a vector of size ngrid^dim 
    
    Notes
    -----
    This class computes IVR by numerical integration on a grid. This 
    is intractable/inaccurate in dimensions greater than 4, so it is 
    intended for debugging purposes only.

    """

    def __init__(self, model, inputs, ngrid=200):
        super().__init__(model, inputs)
        self.ngrid = ngrid
        grd = np.mgrid[ [slice(-5*np.max(np.abs(bd)), 
                                5*np.max(np.abs(bd)), ngrid*1j) \
                         for bd in inputs.domain] ]
        self.pts = grd.T.reshape(-1, inputs.input_dim)

    def evaluate(self, x):
        x = np.atleast_2d(x)
        _, var = self.model.predict(x)
        if self.model.normalizer:
            var /= self.model.normalizer.std**2
        cov = self.model.posterior_covariance_between_points(x, self.pts)
        wghts = self.get_weights(self.pts)
        int_cov = grid_nint(self.pts, wghts.flatten() * cov.flatten()**2, 
                            ngrid=self.ngrid)
        return -int_cov/var

    def jacobian(self, x):
        return jacobian_fdiff(self, x)


class IVR_LWInt(AcquisitionWeighted, IVRInt):
    """A class for IVR-LW computed by numerical integration.

    Parameters
    ----------
    model, inputs, likelihood : see parent class (AcquisitionWeighted)
    ngrid : int
        Number of grid points in each direction

    Attributes
    ----------
    model, inputs, likelihood, ngrid : see Parameters
    pts : array  
        Grid points as a vector of size ngrid^dim 
    
    Notes
    -----
    This class computes IVR-LW by numerical integration on a grid. This 
    is intractable/inaccurate in dimensions greater than 4, so it is 
    intended for debugging purposes only.

    """

    def __init__(self, model, inputs, likelihood=None, ngrid=200):
        super().__init__(model, inputs, likelihood=likelihood)
        self.ngrid = ngrid
        grd = np.mgrid[ [slice(-5*np.max(np.abs(bd)), 
                                5*np.max(np.abs(bd)), ngrid*1j) \
                         for bd in inputs.domain] ]
        self.pts = grd.T.reshape(-1, inputs.input_dim)


class QInt(AcquisitionWeighted):
    """A class for the Q criterion computed by numerical integration.

    Parameters
    ----------
    model, inputs, likelihood : see parent class (AcquisitionWeighted)
    ngrid : int
        Number of grid points in each direction

    Attributes
    ----------
    model, inputs, likelihood, ngrid : see Parameters
    pts : array  
        Grid points as a vector of size ngrid^dim 

    Notes
    -----
    This class computes Q by numerical integration on a grid. This 
    is intractable/inaccurate in dimensions greater than 4, so it is 
    intended for debugging purposes only.

    """

    def __init__(self, model, inputs, likelihood=None, ngrid=200):
        super().__init__(model, inputs, likelihood=likelihood)
        self.ngrid = ngrid
        grd = np.mgrid[ [slice(-5*np.max(np.abs(bd)),
                                5*np.max(np.abs(bd)), ngrid*1j) \
                         for bd in inputs.domain] ]
        self.pts = grd.T.reshape(-1, inputs.input_dim)

    def evaluate(self, x):
        x = np.atleast_2d(x)
        gpn = add_xnew(x, self.model)
        _, var_new = gpn.predict(self.pts)
        wghts = np.exp(self.likelihood.gmm.score_samples(self.pts))
        qdx = wghts * var_new.flatten()
        # Normalize by current variance
        if gpn.normalizer:
            qdx /= gpn.normalizer.std**2
        Q = grid_nint(self.pts, qdx, ngrid=self.ngrid)
        return Q

    def jacobian(self, x):
        return jacobian_fdiff(self, x)


