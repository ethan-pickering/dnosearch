import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from .utils import set_worker_env
import time


class BlackBox:
    """A class for definition of the black-box objective function.

    Parameters
    ----------
    fun : callable
        Black-box objective function to be minimized.
    args : dict, optional
        A dictionary of arguments for the objective function.
    kwargs : dict, optional
        A dictionary of keyword arguments for the objective function.
    noise_var : float, optional
        Variance for additive Gaussian noise. Default is 0, equivalent 
        to noiseless observations.

    Attributes
    ----------
    fun, args, kwargs, noise_var : see Parameters

    """

    def __init__(self, fun, args={}, kwargs={}, noise_var=0.0):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.noise_var = noise_var

    def evaluate(self, x, dim, parallel=False, include_noise=True):
        """Evaluates the black box at point x.

        Parameters
        ----------
        x : array
            Query points. Should be of size (n_pts, n_dim)
        parallel : boolean, optional
            Whether or not to evaluate the black box in parallel.
        include_noise : boolean, optional
            Whether or not to add observation noise.

        Returns
        -------
        y : array
            (Possibly noisy) observations at query points. 

        """
        x = np.atleast_2d(x)

        if parallel:
            y = self._evaluate_parallel(x)
        else:
            y = self._evaluate_serial(x, dim)

        if self.noise_var == 0.0:
            noise = 0.0
        else:
            std = np.sqrt(self.noise_var)
            noise = np.random.normal(0, std, y.shape)

        if include_noise:
            y += noise

        return y

    __call__ = evaluate

    def _evaluate_serial(self, x, dim):
        """Evaluates the black box at x in serial."""
        f_evals = np.empty(shape=[0, dim])
        for i in range(x.shape[0]):
            rlt = self.fun(x[i], *self.args, **self.kwargs)
            f_evals = np.vstack([f_evals, rlt])
        return f_evals


    def _evaluate_parallel(self, x, n_jobs=10, callback=True):
        """Evaluates the black box at x in parallel."""
        set_worker_env()

        if callback:
            print("Sampling in parallel...")
            t = time.time()

        y = Parallel(n_jobs=n_jobs, backend="loky")(
                     delayed(self.fun)(x[i], *self.args, **self.kwargs)
                     for i in range(x.shape[0]) )

        samples = np.atleast_2d(np.array(y).ravel()).T

        if callback:
            m, s = divmod(time.time() - t, 60)
            print("Completed in {:02d}:{:02d}".format(int(m), int(s)))

        return samples


