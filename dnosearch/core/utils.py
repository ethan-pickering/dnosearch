import os
import numpy as np
import scipy
import warnings
from KDEpy import FFTKDE


def set_worker_env(n_threads=1):
    """Prevents over-subscription in joblib."""
    MAX_NUM_THREADS_VARS = [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for var in MAX_NUM_THREADS_VARS:
        os.environ[var] = str(n_threads)


def comp_pca(usnap, n_trunc, detrend=True):
    """Perform Principal Component Analysis on data. 

    Parameters
    ----------
    usnap : array
        Data array, size (n_pts, n_dim).
    n_trunc : integer
        The number of PCA dimensions to be retained.
    detrend : boolean, optional
        Whether or not to deduct the mean from the data.

    Returns
    -------
    lam : array
        The first n_trunc PCA eigenvalues.
    phi : array
        The first n_trunc PCA eigenfunctions.
    usnap_mean : array
        The mean of the data.

    """
    m = usnap.shape[0]
    usnap_mean = np.mean(usnap, axis=0)
    if detrend:
        usnap = usnap - usnap_mean[np.newaxis,:]
    R = np.einsum('ij,kj->ik', usnap, usnap) / m
    lam, phi = np.linalg.eigh(R)
    idx = lam.argsort()[::-1]
    lam = lam[idx]
    phi = phi[:,idx] / np.sqrt(m)
    lam_inv = np.diag(1.0/np.sqrt(lam))
    psi = np.dot(usnap.T, np.dot(phi,lam_inv))
    return lam[0:n_trunc], psi[:,0:n_trunc], usnap_mean


def grid_nint(pts, fmu, ngrid):
    ndim = pts.shape[-1]
    grd = pts.reshape( (ngrid,)*ndim + (ndim,) ).T
    fmug = fmu.reshape( (ngrid,)*ndim ).T
    axes = [ grd[ (i,) + (0,)*i + (slice(None),) + (0,)*(ndim-i-1) ] \
             for i in range(ndim) ]
    I = fmug
    for ii in range(ndim):
        I = np.trapz(I, x=axes[ii], axis=0)
    return I


def fix_dim_gmm(gmm, matrix_type="covariance"):
 
    if matrix_type == "covariance":
        matrix = gmm.covariances_
    elif matrix_type == "precisions":
        matrix = gmm.precisions_
    elif matrix_type == "precisions_cholesky":
        matrix = gmm.precisions_cholesky_

    n_components, n_features = gmm.means_.shape
    m = np.empty((n_components, n_features, n_features))

    for n in range(gmm.n_components):
        if gmm.covariance_type == "full":
            m[n] = matrix[n]
        elif gmm.covariance_type == "tied":
            m[n] = matrix
        elif gmm.covariance_type == "diag":
            m[n] = np.diag(matrix[n])
        elif gmm.covariance_type == "spherical":
            m[n] = np.eye(gmm.means_.shape[1]) * matrix[n]

    return m


def process_parameters(dim, mean, cov):
    """
    Infer dimensionality from mean or covariance matrix, ensure that
    mean and covariance are full vector resp. matrix.
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if dim == 1:
        mean.shape = (1,)
        cov.shape = (1, 1)

    if mean.ndim != 1 or mean.shape[0] != dim:
        raise ValueError("Array 'mean' must be vector of length %d." % dim)
    if cov.ndim == 0:
        cov = cov * np.eye(dim)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    else:
        if cov.shape != (dim, dim):
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                                 " but cov.ndim = %d" % cov.ndim)

    return mean, cov


def add_xnew(x, gpr):
    gpn = gpr.copy()
    x = np.atleast_2d(x)
    y, _ = gpn.predict(x)
    X = np.vstack((gpn.X,x))
    Y = np.vstack((gpn.Y,y))
    gpn.set_XY(X,Y)
    return gpn


def true_pdf(inputs, filename=None):
    if filename is None:
        filename = "map_samples.txt"
    smpl = np.genfromtxt(filename)
    weights = inputs.pdf(smpl[:,0:-1])
    pdf = custom_KDE(smpl[:,-1], weights=weights)
    return pdf


def model_pdf(model, inputs, alpha=1.96, pts=None):
    if pts is None:
        pts = inputs.draw_samples(int(1e6), "uni")
    mu, var = model.predict(pts)
    mu, var = np.squeeze(mu), np.squeeze(var) 
    ww = inputs.pdf(pts)
    pb = custom_KDE(mu, weights=ww)
    pp = custom_KDE(mu+alpha*np.sqrt(var), weights=ww)
    pm = custom_KDE(mu-alpha*np.sqrt(var), weights=ww)
    return pb, pp, pm


def compute_mean(x, model):
    x = np.atleast_2d(x)
    mu, _ = model.predict(x)
    return mu.flatten()


def compute_mean_jac(x, model):
    x = np.atleast_2d(x)
    mu_jac, _ = model.predictive_gradients(x)
    return mu_jac[:,:,0]


def get_standard_normal_pdf_cdf(x, mean, standard_deviation):
    u = (x - mean) / standard_deviation
    pdf = scipy.stats.norm.pdf(u)
    cdf = scipy.stats.norm.cdf(u)
    return u, pdf, cdf


def jacobian_fdiff(function, x):
    x = np.atleast_2d(x)
    Q = function.evaluate(x)
    eps = 1e-8
    jacQ = np.zeros(x.shape[1])
    for ii in range(x.shape[1]):
        x_eps = x + eps*np.eye(1, x.shape[1], ii)
        Qeps = function.evaluate(x_eps)
        jacQ[ii] = (Qeps-Q)/eps
    return jacQ


def custom_KDE(data, weights=None, bw=None):
    data = data.flatten()
    if bw is None:
        try:
            sc = scipy.stats.gaussian_kde(data, weights=weights)
            bw = np.sqrt(sc.covariance).flatten()
            # Ensure that bw is a scalar value
            if np.size(bw) == 1:
                bw = np.asscalar(bw)
            else:
                raise ValueError("The bw must be a number.")
        except:
            bw = 1.0
        if bw < 1e-8:
            bw = 1.0           
    return FFTKDE(bw=bw).fit(data, weights)


def trim_to_65535(x):
    """Incorrect output when batch size > 65535.
    https://github.com/tensorflow/tensorflow/issues/9870
    https://github.com/tensorflow/tensorflow/issues/13869
    """
    N = 65535
    if isinstance(x, (list, tuple)):
        return (x[0][:N], x[1][:N]), (x[0][N:], x[1][N:])
    return x[:N], x[N:]


def mean_squared_error_outlier(y_true, y_pred):
    error = np.ravel((y_true - y_pred) ** 2)
    error = np.sort(error)[: -len(error) // 1000]
    return np.mean(error)


def safe_test(model, data, X_test, y_test, fname=None):
    def is_nonempty(X):
        return len(X[0]) > 0 if isinstance(X, (list, tuple)) else len(X) > 0

    y_pred = []
    X = X_test
    while is_nonempty(X):
        X_add, X = trim_to_65535(X)
        y_pred.append(model.predict(data.transform_inputs(X_add)))
    y_pred = np.vstack(y_pred)
    error = np.mean((y_test - y_pred) ** 2)
    print("Test MSE: {}".format(error))
    error = mean_squared_error_outlier(y_test, y_pred)
    print("Test MSE w/o outliers: {}\n".format(error))

    if fname is not None:
        np.savetxt(fname, np.hstack((X_test[1], y_test, y_pred)))


