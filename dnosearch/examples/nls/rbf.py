import GPy
import numpy as np
from GPy.util.linalg import pdinv


class RBF(GPy.kern.RBF):
    """A GPy RBF kernel augmented with integration routines.

    Notes
    -----
    Parameters and attributes are inherited from GPy.kern.RBF. 

    """
    def IntKK(self, x1, x2=None):
        """Compute \int k(x1,x') k(x',x2) dx'.

        Parameters
        ----------
        x1 : array, size (n1, n_dim)
        x2 : array, size (n2, n_dim)
            If None, we take x2 = x1.

        Returns
        -------
        res : array, size (n1, n2)

        """
        if x2 is None:
            x2 = x1
        ndim = self.input_dim
        var = self.variance
        ell2 = np.ones((ndim,)) * self.lengthscale**2
        sqrt_det = np.sqrt(np.prod(ell2))
        const = var * np.power(np.pi, ndim/2.0) * sqrt_det
        k1 = self.K(x1/np.sqrt(2), x2/np.sqrt(2))
        return const * k1

    def dIntKK_dX(self, x1, x2=None):
        """Compute d/dx1 \int k(x1,x') k(x',x2) dx'.

        Parameters
        ----------
        x1 : array, size (n1, n_dim)
        x2 : array, size (n2, n_dim)
            If None, we take x2 = x1.

        Returns
        -------
        jac : array, size (n1, n2, n_dim)
            The gradients of IntKK w.r.t. x1.

        """
        if x2 is None:
            jacobian = x1[:,None] - x1[None,:]
        else:
            ndim = self.input_dim
            ell2 = np.ones((ndim,)) * self.lengthscale**2
            integral = self.IntKK(x1, x2)
            const = (x1[:,None] - x2[None,:]) / ell2 / 2.0
            jacobian = - const * integral[:,:,None] 
        return jacobian

    def IntKKNorm(self, x1, x2, mu, sigma):
        """Compute \int k(x1,x') k(x',x2) Normal_x'(\mu, \Sigma) dx'.

        Parameters
        ----------
        x1 : array, size (n1, n_dim)
        x2 : array, size (n2, n_dim)
        mu : array, size (n_dim)
            The mean of the Gaussian distribution.
        cov : array, size (n_dim, n_dim)
            The covariance of the Gaussian distribution.

        Returns
        -------
        res : array, size (n1, n2)

        """
        ndim = self.input_dim
        var = self.variance
        ell2 = np.ones((ndim,)) * self.lengthscale**2
        sqrt_det = np.sqrt(np.prod(ell2))
        cov = sigma + 0.5*np.diag(ell2)
        cov_inv, _, _, ld, = pdinv(cov)
        x_shift = 0.5 * ( x1[:,None] + x2[None,:] ) - mu
        arg = np.sum( x_shift * np.matmul(x_shift, cov_inv), axis=2)
        k1 = var * np.exp(-0.5*arg)
        k2 = self.K(x1/np.sqrt(2), x2/np.sqrt(2))
        const = np.exp(-0.5*ld) * sqrt_det / np.power(2, ndim/2.0)
        return const * k1 * k2

    def dIntKKNorm_dX(self, x1, x2, mu, sigma):
        """Compute 
            d/dx1 \int k(x1,x') k(x',x2) Normal_x'(\mu, \Sigma) dx'.

        Parameters
        ----------
        x1 : array, size (n1, n_dim)
        x2 : array, size (n2, n_dim)
        mu : array, size (n_dim)
            The mean of the Gaussian distribution.
        cov : array, size (n_dim, n_dim)
            The covariance of the Gaussian distribution.

        Returns
        -------
        jac : array, size (n1, n2, n_dim)
            The gradients of IntKKNorm w.r.t. x1.

        """
        ndim = self.input_dim
        ell2 = np.ones((ndim,)) * self.lengthscale**2
        cov = sigma + 0.5*np.diag(ell2)
        cov_inv, _, _, ld, = pdinv(cov)
        x_avg = 0.5 * ( x1[:,None] + x2[None,:] )
        aa = np.dot(2*x_avg/ell2, sigma) + mu
        const = -x1[:,None]/ell2 + 0.5*np.dot(aa, cov_inv)
        integral = self.IntKKNorm(x1, x2, mu, sigma)
        jacobian = const * integral[:,:,None]
        return jacobian


