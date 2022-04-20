import numpy as np
import GPy
from GPy.util.linalg import symmetrify, trace_dot, pdinv
from .base import AcquisitionWeighted
from ..utils import fix_dim_gmm, add_xnew


class Q(AcquisitionWeighted):
    """A class for the Q criterion.

    Parameters
    ----------
    model, inputs, likelihood : see parent class (AcquisitionWeighted)

    Attributes
    ----------
    model, inputs, likelihood : see Parameters
    
    Notes
    -----
    This class implements the original formulation of the Q criterion 
    (see https://arxiv.org/abs/1907.07552). However, evaluation is 
    cumbersome, and IVR-LW should be preferred, as the two are 
    strictly equivalent. 

    """

    def evaluate(self, x):
        return self.comp_Qcrit(x, self.model, self.likelihood.gmm)

    def jacobian(self, x):
        return self.comp_dQcrit_dXnew(x, self.model, self.likelihood.gmm)

    def comp_Qcrit(self, x_new, model, gmm):
        x_new = np.atleast_2d(x_new)
        gpn = add_xnew(x_new, model)
        Q = 0
        covs = fix_dim_gmm(gmm, matrix_type="covariance")
        for ii in range(gmm.n_components):
            mean_i = gmm.means_[ii]
            cov_i = covs[ii]
            wei = gmm.weights_[ii]
            Q += wei*self.comp_Qi(gpn, mean_i, cov_i)
        return Q

    def comp_dQcrit_dXnew(self, x_new, model, gmm):
        x_new = np.atleast_2d(x_new)
        gpn = add_xnew(x_new, model)
        jacQ = 0
        covs = fix_dim_gmm(gmm, matrix_type="covariance")
        for ii in range(gmm.n_components):
            mean_i = gmm.means_[ii]
            cov_i = covs[ii]
            wei = gmm.weights_[ii]
            jacQ += wei*self.comp_dQi_dXnew(gpn, mean_i, cov_i)
        return jacQ

    def comp_Qi(self, gpr, mean_i, cov_i):
        """Compute contribution of each Gaussian mixture."""
        X = gpr.X
        kernel = gpr.kern
        mukk = kernel.variance + gpr.Gaussian_noise.variance
        Ckk, _ = self.comp_Ckk(X, kernel, mean_i, cov_i)
        Skk_inv = gpr.posterior.woodbury_inv
        Qcrit = mukk - trace_dot(Skk_inv, Ckk)
        return Qcrit

    def comp_dQi_dXnew(self, gpr, mean_i, cov_i):
        """Compute gradients of contribution for each Gaussian mixture."""
        X = gpr.X
        kernel = gpr.kern
        Ckk, cov_inv = self.comp_Ckk(X, kernel, mean_i, cov_i)
        Skk_inv = gpr.posterior.woodbury_inv
        ell2 = kernel.lengthscale**2
        Xnew = np.atleast_2d(X[-1,:])
        B = np.dot(Skk_inv, np.dot(Ckk, Skk_inv))
        grad1 = kernel.K(X, Xnew) * (X-Xnew) / ell2
        tr1 = 2.*np.dot(B[:,-1], grad1)
        Xavg = (X+Xnew)/2.
        aa = mean_i[None,:] + np.dot( 2.0 * Xavg / ell2[None,:], cov_i)
        bb = 0.5*np.dot(aa, cov_inv) - Xnew/ell2[None,:]   
        grad2 = bb*Ckk[-1,:][:,None]
        tr2 = 2*np.dot(grad2.T, Skk_inv[:,-1])
        return tr1 - tr2

    @staticmethod
    def comp_Ckk(X, kernel, mean_i, cov_i):
        """Compute C_kk = \int k(X,x') k(x',X) Normal_i dx'."""
        ndim = kernel.input_dim
        var = kernel.variance
        Xsh = (X-mean_i)/2.
        ell2 = np.ones((ndim,))*kernel.lengthscale**2
        sqrt_det = np.power(np.prod(ell2), 1/2.)
        cov = cov_i + 0.5*np.diag(ell2)
        cov_inv, _, _, ld, = pdinv(cov)
        X1s = np.sum(Xsh * np.dot(Xsh, cov_inv), 1)
        arg = 2.*np.dot(Xsh, np.dot(Xsh, cov_inv).T) \
              + X1s[:,None] + X1s[None,:]
        con = -0.5 * ( ndim * np.log(2 * np.pi) + ld )
        zc = np.exp(con-0.5*arg)
        norm_const = var * np.power(np.pi, ndim/2.0) * sqrt_det \
                     * kernel.K(X/np.sqrt(2))
        return norm_const*zc, cov_inv

    @staticmethod
    def comp_dStd_dXnew(x_new, model, gpn):
        """Compute gradients of standard deviation of output of 
        augmented dataset with respect to ghost point."""
        N = gpn.Y.shape[0]
        kernel = model.kern
        ell2 = kernel.lengthscale**2
        X = model.X
        Y = model.Y_normalized
        Skk_inv = model.posterior.woodbury_inv
        grad1 = kernel.K(X, x_new) * (X-x_new) / ell2
        grad = np.dot(grad1.T, np.dot(Skk_inv, Y)) * model.normalizer.std
        dStd = 2 * ( gpn.Y[-1,:] - gpn.Y.mean(0) ) * grad / N
        return dStd.T



