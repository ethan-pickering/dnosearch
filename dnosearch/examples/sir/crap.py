#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:40:00 2022

@author: ethanpickering
"""

iter_num = 0
# T = 130
# dt = 0.125*2
# gamma = 0.15
# delta = 0
# N_people = 10*10**7
# I0 = 50

T = 45
dt = 0.1
gamma = 0.25
delta = 0
N_people = 10*10**7
I0 = 50

# T = 130/2.5
# dt = 0.25/2.5
# gamma = 0.12
# delta = 0
# N_people = 6*10**7
# I0 = 10

n_init = 50
ndim = dim
rank = dim

np.random.seed(seed)
noise_var = 0
my_map = BlackBox(map_def, noise_var=noise_var)
mean, cov = np.zeros(ndim), np.ones(ndim)
domain = [ [-6, 6] ] * ndim
inputs = GaussianInputs(domain, mean, cov)

if iter_num == 0:
    Theta = inputs.draw_samples(n_init, 'grd')


#Theta = inputs.draw_samples(50, "grd")
#noise = Noise([0,1], sigma=0.1, ell=0.1)
noise = Noise([0,1], sigma=0.1, ell=1)

# Need to determine U
nsteps = int(T/dt)
x_vals = np.linspace(0, 1, nsteps+1)
x_vals = x_vals[0:-1]

# DeepONet only needs a coarse version of the signal    
coarse = 4

# Create the X to U map, which is actually theta to U
multiplier = 3*10**-9 # Special for the map
#multiplier = 2*5*10**-9 # Special for the map


def Theta_to_U(Theta,nsteps,coarse,rank):
    U1 = noise.get_sample(np.transpose(Theta))
    
    NN_grid = np.linspace(0,1,nsteps)
    Noise_grid = np.linspace(0,1,np.shape(U1)[0])

    U = np.zeros((np.shape(Theta)[0],nsteps))
    for i in range(0,np.shape(Theta)[0]):
        interp_func = InterpolatedUnivariateSpline(Noise_grid, U1[:,i], k=1)
        U[i,:] = interp_func(NN_grid)
    
    coarser_inds = np.linspace(0,nsteps-1,int(nsteps/coarse)).astype(int)
    U = U[:,coarser_inds]
    return U


def Theta_to_Z(Theta,rank):
    if Theta.shape[1] == rank:
        Z = np.ones((Theta.shape[0], 1))
    else:
        Z = Theta[:,(2*rank):Theta.shape[1]]
    return Z         

if iter_num == 0:

    # Determine the training data
    Y = np.zeros((n_init**2,))
    Us = Theta_to_U(Theta,nsteps,1,2)+2.55
    Us = Us*multiplier

    for i in range(0,n_init**2):
        I_temp = map_def(Us[i,:],gamma,delta,N_people,I0,T,dt, np.zeros(np.shape(Us[i,:])))
        Y[i] = I_temp[-1]

    Y = Y.reshape(n_init**2,1)

plt.pcolor(Theta[:,0].reshape(n_init,n_init), Theta[:,1].reshape(n_init,n_init), Y.reshape(n_init,n_init))
plt.colorbar()
plt.show()

sio.savemat('/Users/ethanpickering/Documents/git/publish/dnosearch/dnosearch/examples/sir/test.mat', {'Y':Y})


# Determine Bounds for evaluzting the metric
test_pts = n_init
x_max = np.max(Y)
x_min = np.min(Y)
x_int = np.linspace(x_min,x_max,10000) # Linearly space points
x_int_standard = np.linspace(0,10**8,10000) # Static for pt-wise comparisons

# Create the weights/exploitation values
px = inputs.pdf(Theta)
sc = scipy.stats.gaussian_kde(Y.reshape(test_pts**2,), weights=px)   # Fit a guassian kde using px input weights
py = sc.evaluate(x_int) # Evaluate at x_int
py[py<10**-16] = 10**-16 # Eliminate spuriously small values (smaller than numerical precision)
py_standard = sc.evaluate(x_int_standard) # Evaluate for pt-wise comparisons
py_interp = InterpolatedUnivariateSpline(x_int, py, k=1) # Create interpolation function

# Conctruct the weights
wx = px.reshape(test_pts**2,)/py_interp(Y).reshape(test_pts**2,)
wx = wx.reshape(test_pts**2,1)

plt.pcolor(Theta[:,0].reshape(n_init,n_init), Theta[:,1].reshape(n_init,n_init), wx.reshape(n_init,n_init))
plt.colorbar()
plt.show()
plt.semilogy(x_int,py)
plt.xlim([0,26000000])
plt.ylim([10**-9, 10**-6.75])


#%%


T = 130/2.5
dt = 0.25/2.5
gamma = 0.12
delta = 0
N_people = 6*10**7
I0 = 10

ndim = dim
rank = dim
#seed = 1
#ndim = 2
#n_init = 5
n_init = 50

np.random.seed(seed)
noise_var = 0
my_map = BlackBox(map_def, noise_var=noise_var)
mean, cov = np.zeros(ndim), np.ones(ndim)
domain = [ [-6, 6] ] * ndim
inputs = GaussianInputs(domain, mean, cov)

if iter_num == 0:
    Theta = inputs.draw_samples(n_init, 'grd')


#Theta = inputs.draw_samples(50, "grd")
noise = Noise([0,1], sigma=0.1, ell=0.1)

# Need to determine U
nsteps = int(T/dt)
x_vals = np.linspace(0, 1, nsteps+1)
x_vals = x_vals[0:-1]

# DeepONet only needs a coarse version of the signal    
coarse = 4

# Create the X to U map, which is actually theta to U
multiplier = 2*5*10**-9 # Special for the map


def Theta_to_U(Theta,nsteps,coarse,rank):
    U1 = noise.get_sample(np.transpose(Theta))
    
    NN_grid = np.linspace(0,1,nsteps)
    Noise_grid = np.linspace(0,1,np.shape(U1)[0])

    U = np.zeros((np.shape(Theta)[0],nsteps))
    for i in range(0,np.shape(Theta)[0]):
        interp_func = InterpolatedUnivariateSpline(Noise_grid, U1[:,i], k=1)
        U[i,:] = interp_func(NN_grid)
    
    coarser_inds = np.linspace(0,nsteps-1,int(nsteps/coarse)).astype(int)
    U = U[:,coarser_inds]
    return U


def Theta_to_X(Theta,rank):
    if Theta.shape[1] == rank:
        X = np.ones((Theta.shape[0], 1))
    else:
        X = Theta[:,(2*rank):Theta.shape[1]]
    return X         

if iter_num == 0:

    # Determine the training data
    Y = np.zeros((n_init**2,))
    Us = Theta_to_U(Theta,nsteps,1,2)*multiplier/1 + multiplier

    for i in range(0,n_init**2):
        I_temp = map_def(Us[i,:],gamma,delta,N_people,I0,T,dt, np.zeros(np.shape(Us[i,:])))
        #Y[i] = np.log10(I_temp[-1])/10 - 0.5
        Y[i] = I_temp[-1]

    Y = Y.reshape(n_init**2,1)
plt.pcolor(Theta[:,0].reshape(n_init,n_init), Theta[:,1].reshape(n_init,n_init), Y.reshape(n_init,n_init))
plt.colorbar()
plt.show()