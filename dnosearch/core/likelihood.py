import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.mixture import GaussianMixture as GMM
from .utils import fix_dim_gmm, custom_KDE
import scipy.io as sio


class Likelihood(object):
    """A class for computation of the likelihood ratio.

    Parameters
    ----------
    model : instance of GPRegression
        A GPy model 
    inputs : instance of Inputs
        The input space.
    weight_type : str, optional
        Type of likelihood weight. Must be one of
            - "nominal" : uses w(x) = p(x)
            - "importance" : uses w(x) = p(x)/p_y(mu(x))
    fit_gmm : boolean, optional
        Whether or not to use a GMM approximation for the likelihood
        ratio.  
    kwargs_gmm : dict, optional
        A dictionary of keyword arguments for scikit's GMM routine.
        Use this to specify the number of Gaussian mixtures and the
        type of covariance matrix.

    Attributes
    ----------
    model, inputs, weight_type, fit_gmm, kwargs_gmm : see Parameters
    fy_interp : scipy 1-D interpolant
        An interpolant for the output pdf p_y(mu)
    gmm : scikit Gaussian Mixture Model
        A GMM object approximating the likelihood ratio.

    """

    def __init__(self, model, inputs, surrogate_model, weight_type="importance", 
                 fit_gmm=True, kwargs_gmm=None):

        self.model = model
        self.inputs = inputs
        self.dim = np.shape(inputs.domain)[0]
        self.weight_type = self.check_weight_type(weight_type)
        #self.fit_gmm = fit_gmm
        self.fit_gmm = False
        #self.vector = True
        self.surrogate_model = surrogate_model       
        print(surrogate_model)
        if kwargs_gmm is None:
            kwargs_gmm = dict(n_components=100, covariance_type="full")
        self.kwargs_gmm = kwargs_gmm

        self._prepare_likelihood()

    def update_model(self, model):
        self.model = model
        self._prepare_likelihood()
        return self

    def evaluate(self, x, vector=True):
        """Evaluates the likelihood ratio at x.

        Parameters
        ----------
        x : array
            Query points. Should be of size (n_pts, n_dim)

        Returns
        -------
        w : array
            The likelihood ratio at x.

        """
        if self.fit_gmm:
            w = self._evaluate_gmm(x)
        else:
            w = self._evaluate_raw(x, vector)
        return w

    def jacobian(self, x, vector=True):
        """Evaluates the gradients of the likelihood ratio at x.

        Parameters
        ----------
        x : array
            Query points. Should be of size (n_pts, n_dim)

        Returns
        -------
        w_jac : array
            Gradients of the likelihood ratio at x.

        """
        if self.fit_gmm:
            w_jac = self._jacobian_gmm(x)
        else:
            w_jac = self._jacobian_raw(x, vector)
        return w_jac

    def _evaluate_gmm(self, x):
        x = np.atleast_2d(x)
        w = np.exp(self.gmm.score_samples(x))
        return w[:,None]

    def _jacobian_gmm(self, x):
        x = np.atleast_2d(x)
        w_jac = np.zeros(x.shape)
        p = np.exp(self.gmm._estimate_weighted_log_prob(x))
        precisions = fix_dim_gmm(self.gmm, matrix_type="precisions")
        for ii in range(self.gmm.n_components):
            w_jac += p[:,ii,None] * np.dot(self.gmm.means_[ii]-x, \
                                           precisions[ii])
        return w_jac

    def _evaluate_raw(self, x, vector):
        vector = False
        if vector:
            dim = self.dim
            x = np.atleast_2d(x)
            unique_pts = int(np.shape(x)[0]/dim) 
            #fxs = np.zeros((unique_pts,))
            #fys = np.zeros((unique_pts,))
            #mus = np.zeros((unique_pts,))
            x = x.reshape(unique_pts,dim)
            fx = self.inputs.pdf(x)
            if self.weight_type == "nominal":
                w = fx
            elif self.weight_type == "importance":
                if self.surrogate_model == 'DON':
                    mu = self.model.predict_spec(x,batches=1,batch_size=np.shape(x)[0],i=0,j=0)[0].flatten()
                    #mu = self.model.predict(x)[0].flatten()

                elif self.surrogate_model == 'GP':
                    mu = self.model.predict(x)[0].flatten()
                if self.model.normalizer:
                    mu = self.model.normalizer.normalize(mu)
                fy = self.fy_interp(mu)
                w = np.sum(fx/fy)
        else:
            x = np.atleast_2d(x)
            fx = self.inputs.pdf(x)
            if self.weight_type == "nominal": 
                w = fx
            elif self.weight_type == "importance":
                if self.surrogate_model == 'DON':
                    mu = self.model.predict_spec(x,batches=1,batch_size=np.shape(x)[0],i=0,j=0)[0].flatten()
                    #mu = self.model.predict(x)[0].flatten()
                elif self.surrogate_model == 'GP':
                    mu = self.model.predict(x)[0].flatten()
                if self.model.normalizer:
                    mu = self.model.normalizer.normalize(mu)
                fy = self.fy_interp(mu)
                w = fx/fy
            
        return w[:,None]

    def _jacobian_raw(self, x, vector):
        vector = False
        if vector:
            dim = self.dim
            x = np.atleast_2d(x)
            unique_pts = int(np.shape(x)[0]/dim)
            #fxs = np.zeros((unique_pts,))
            #fys = np.zeros((unique_pts,))
            #mus = np.zeros((unique_pts,))
            x = x.reshape(unique_pts,dim)
            fx = self.inputs.pdf_jac(x)
    
            x = np.atleast_2d(x)
            fx_jac = self.inputs.pdf_jac(x)
    
            if self.weight_type == "nominal":
                w_jac = fx_jac
    
            elif self.weight_type == "importance":
                if self.surrogate_model == 'DON':
                    mu = self.model.predict_spec(x,batches=1,batch_size=np.shape(x)[0],i=0,j=0)[0].flatten()
                    #mu = self.model.predict(x)[0].flatten()
                elif self.surrogate_model == 'GP':
                    mu = self.model.predict(x)[0].flatten()
                if self.model.normalizer:
                    mu = self.model.normalizer.normalize(mu)
    
                if self.surrogate_model == 'DON':
                    mu_jac, _ = self.model.predictive_gradients_spec(x,batches=1,batch_size=np.shape(x)[0],i=0,j=0)
                    #mu_jac, _ = self.model.predictive_gradients(x)
                elif self.surrogate_model == 'GP':
                    mu_jac, _ = self.model.predictive_gradients(x)
                    mu_jac = mu_jac[:,:,0]
                #mu_jac, _ = self.model.predictive_gradients_spec(x,batches=1,batch_size=np.shape(x)[0],i=0,j=0)
                #mu_jac = mu_jac[:,:,0] I am not sure why this is here?!?!?!
                fx = self.inputs.pdf(x)
                fy = self.fy_interp(mu)
                fy_jac = self.fy_interp.derivative()(mu)
                tmp = fx * fy_jac / fy**2
                w_jac = fx_jac / fy[:,None] - tmp[:,None] * mu_jac
        else:
            x = np.atleast_2d(x)
            fx_jac = self.inputs.pdf_jac(x)
    
            if self.weight_type == "nominal":
                w_jac = fx_jac
    
            elif self.weight_type == "importance":
                if self.surrogate_model == 'DON':
                    mu = self.model.predict_spec(x,batches=1,batch_size=np.shape(x)[0],i=0,j=0)[0].flatten()
                    #mu = self.model.predict(x)[0].flatten()
                elif self.surrogate_model == 'GP':
                    mu = self.model.predict(x)[0].flatten()
                if self.model.normalizer:
                    mu = self.model.normalizer.normalize(mu)
    
                if self.surrogate_model == 'DON':
                    mu_jac, _ = self.model.predictive_gradients_spec(x,batches=1,batch_size=np.shape(x)[0],i=0,j=0)
                    #mu_jac, _ = self.model.predictive_gradients(x)
                elif self.surrogate_model == 'GP':
                    mu_jac, _ = self.model.predictive_gradients(x)
                    mu_jac = mu_jac[:,:,0]
                #mu_jac, _ = self.model.predictive_gradients_spec(x,batches=1,batch_size=np.shape(x)[0],i=0,j=0)
                #mu_jac = mu_jac[:,:,0] I am not sure why this is here?!?!?!
                fx = self.inputs.pdf(x)
                fy = self.fy_interp(mu)
                fy_jac = self.fy_interp.derivative()(mu)
                tmp = fx * fy_jac / fy**2
                w_jac = fx_jac / fy[:,None] - tmp[:,None] * mu_jac

        return w_jac

    def _prepare_likelihood(self):
        """Prepare likelihood ratio for evaluation."""

        if self.inputs.input_dim <= 2:
            n_samples = int(1e5)
        else: 
            n_samples = int(1e6)
            
        batches = 1
        batch_size = n_samples / batches
        
        np.random.seed(np.size(self.model.Y)) #y: To ensure the sampling is not always the same - particularly useful for high dimensions  
        #pts1 = self.inputs.draw_samples(n_samples=int(n_samples*0.000001), 
        #                              sample_method="pdf")
        #pts2 = self.inputs.draw_samples(n_samples=int(n_samples*0.999999), 
        #                              sample_method="uni")
        #pts3 = self.inputs.draw_samples(n_samples=int(n_samples*0), 
         #                             sample_method="lhs")
        #pts = np.append(pts1,pts2,axis=0)
        #pts = np.append(pts,pts3,axis=0)

        pts = self.inputs.draw_samples(n_samples=n_samples, 
                                       sample_method="uni")

        fx = self.inputs.pdf(pts)

        if self.weight_type == "importance":
            # mu = np.zeros((n_samples,))
            # for batch in range(0,batches):
            #     inds = np.linspace(batch*batch_size, (batch_size*(batch+1))-1, int(batch_size)).astype(int)
            #     mu[inds] = self.model.predict_spec(pts[inds,:])[0].flatten()

            if self.surrogate_model == 'DON':
                mu = self.model.predict_spec(pts,batches,batch_size,0,0)[0].flatten()
                #mu = self.model.predict(pts)[0].flatten()
            elif self.surrogate_model == 'GP':
                mu = self.model.predict(pts)[0].flatten()
            
            #mu = self.model.predict(pts)[0].flatten()
            # Saving the variables to see what is going on here
            #sio.savemat('/scratch/epickeri/MMT/test_results.mat', {'mu':mu, 'fx':fx, 'inds':inds, 'pts':pts})
            if self.model.normalizer:
                mu = self.model.normalizer.normalize(mu)
            x, y = custom_KDE(mu, weights=fx).evaluate()
            # plt.plot(mu); plt.show()
            # plt.semilogy(x,y); plt.show()
            self.fy_interp = InterpolatedUnivariateSpline(x, y, k=1)

        if self.fit_gmm:
            if self.weight_type == "nominal":
                w_raw = fx
            elif self.weight_type == "importance":
                w_raw = fx/self.fy_interp(mu)
            self.gmm = self._fit_gmm(pts, w_raw, self.kwargs_gmm)

        return self

    @staticmethod
    def _fit_gmm(pts, w_raw, kwargs_gmm):
        """Fit Gaussian Mixture Model using scikit's GMM framework.

        Parameters
        ----------
        pts : array
            Sample points. 
        w_raw : array
            Raw likelihood ratio at sample points.
        kwargs_gmm : dict
            A dictionary of keyword arguments for scikit's GMM routine.

        Returns
        -------
        gmm : scikit Gaussian Mixture Model
            A GMM object approximating the likelihood ratio.

        """
        # Sample and fit
        sca = np.sum(w_raw)
        rng = np.random.default_rng()
        aa = rng.choice(pts, size=20000, p=w_raw/sca)
        gmm = GMM(**kwargs_gmm)
        gmm = gmm.fit(X=aa)
        # Rescale
        gmm_y = np.exp(gmm.score_samples(pts))
        scgmm = np.sum(gmm_y)
        gmm.weights_ *= (sca/w_raw.shape[0] * gmm_y.shape[0]/scgmm)
        return gmm

    @staticmethod
    def check_weight_type(weight_type):
        assert(weight_type.lower() in ["nominal", "importance"])
        return weight_type.lower()
