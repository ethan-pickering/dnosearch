import numpy as np
from scipy.interpolate import interp1d


class Oscillator:

    def __init__(self, noise, tf, nsteps, u_init,
                 delta=1.5, alpha=1.0, beta=0.1, x1=0.5, x2=1.5):
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.x1 = x1
        self.x2 = x2
        self.noise = noise
        self.tf = tf
        self.nsteps = nsteps
        self.u_init = u_init

    def rhs(self, u, t):
        u0, u1 = u
        f0 = u1
        f1 = -self.delta*u1 - self.f_nl(u0) + self.sample_noise(t) 
        f = [f0, f1]
        return f

    def f_nl(self, u0):
        if np.abs(u0) <= self.x1:
            return self.alpha * u0
        elif np.abs(u0) >= self.x2:
            return self.alpha * self.x1 * np.sign(u0) \
                   + self.beta * (u0 - self.x2 * np.sign(u0))**3
        else:
            return self.alpha * self.x1 * np.sign(u0)

    def solve(self, theta):
        self.sample_noise = self.noise.get_sample_interp(theta) 
        time = np.linspace(0, self.tf, self.nsteps+1)
        solver = ODESolver(self.rhs)
        solver.set_ics(self.u_init)
        u, t = solver.solve(time)
        return u, t


class Noise:

    def __init__(self, domain, sigma=0.1, ell=4.0):
        self.ti = domain[0]
        self.tf = domain[1]
        self.tl = domain[1] - domain[0]
        self.R = self.get_covariance(sigma, ell)
        self.lam, self.phi = self.kle(self.R)

    def get_covariance(self, sigma, ell):
        m = 500 + 1
        self.t = np.linspace(self.ti, self.tf, m)
        self.dt = self.tl/(m-1)
        R = np.zeros([m, m])
        for i in range(m):
            for j in range(m):
                tau = self.t[j] - self.t[i]
                R[i,j] = sigma*np.exp(-tau**2/(2*ell**2)) 
        return R*self.dt

    def kle(self, R):
        lam, phi = np.linalg.eigh(R)
        phi = phi/np.sqrt(self.dt)
        idx = lam.argsort()[::-1]
        lam = lam[idx]
        phi = phi[:,idx]
        return lam, phi

    def get_eigenvalues(self, trunc=None):
        return self.lam[0:trunc]

    def get_eigenvectors(self, trunc=None):
        return self.phi[:,0:trunc]

    def get_sample(self, xi):
        nRV = np.asarray(xi).shape[0] 
        phi_trunc = self.phi[:,0:nRV]
        lam_trunc = self.lam[0:nRV]
        lam_sqrtm = np.diag(np.sqrt(lam_trunc))
        sample = np.dot(phi_trunc, np.dot(lam_sqrtm, xi))
        return sample

    def get_sample_interp(self, xi):
        sample = self.get_sample(xi.ravel())
        sample_int = interp1d(self.t, sample, kind='cubic')
        return sample_int


class ODESolver:

    def __init__(self, f):
        self.f = lambda u, t: np.asarray(f(u, t), float)

    def set_ics(self, U0):
        U0 = np.asarray(U0)
        self.neq = U0.size
        self.U0 = U0

    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        K1 = dt*f(u[k], t[k])
        K2 = dt*f(u[k] + 0.5*K1, t[k] + 0.5*dt)
        K3 = dt*f(u[k] + 0.5*K2, t[k] + 0.5*dt)
        K4 = dt*f(u[k] + K3, t[k] + dt)
        u_new = u[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        return u_new

    def solve(self, time):
        self.t = np.asarray(time)
        n = self.t.size
        self.u = np.zeros((n,self.neq))
        self.u[0] = self.U0
        for k in range(n-1):
            self.k = k
            self.u[k+1] = self.advance()
        return self.u[:k+2], self.t[:k+2]

