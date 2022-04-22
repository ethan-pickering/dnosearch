#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:06:32 2022

@author: ethanpickering
"""

# DNOSearch Imports
import numpy as np
from dnosearch import (BlackBox, GaussianInputs, DeepONet, Oscillator)
from oscillator import Noise

# DeepONet Imports
import deepxde as dde

# Other Imports
import sys
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt

# Variables
iter_num    = 0 # Iteration number
dim         = 2 # Dimension of the stochastic excitation (infection rate)
acq         = 'US_LW' # Acquisition type - currently only Likelihood-weighted uncertatiny sampling
n_init      = 3 # Initial data points
epochs      = 1000  # Number of training epochs
b_layers    = 8 # Branch Layers
t_layers    = 1 # Trunk Layers
neurons     = 300 # Number of neurons per layer
init_method = 'pdf'# How initial data are pulled
N           = 8 # Number of DNO ensembles
seed        = 3 # Seed for initial condition consistency - NOTE due to gradient descent of the DNO, the seed will not provide perfectly similar results, but will be analogous
iters_max   = 100  # Iterations to perform

def map_def(beta,gamma,delta,N,I0,T,dt,f):    
    S = np.zeros((int(T/dt),));
    S[0] = N;
    I= np.zeros((int(T/dt),));
    I[0] = I0;
    R = np.zeros((int(T/dt),));
    for tt in range(0,np.size(S)-1):
        # Ordinary different equations of the model
        dS = (-beta[tt]*I[tt]*S[tt] + delta*R[tt] - f[tt]) * dt;
        dI = (beta[tt]*I[tt]*S[tt] - gamma*I[tt] + f[tt]) * dt;
        dR = (gamma*I[tt] - delta*R[tt]) * dt;
        # Simple integration
        S[tt+1] = S[tt] + dS;
        I[tt+1] = I[tt] + dI;
        R[tt+1] = R[tt] + dR;
    return I


def main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,iters_max):
    
    # T = 65
    # dt = 0.2
    # gamma = 0.1
    # delta = 0
    # N_people = 10**8
    # I0 = 50
    T = 45  
    dt = 0.1
    gamma = 0.25
    delta = 0
    N_people = 10*10**7
    I0 = 50
    
    ndim = dim
    udim = dim # The dimensionality of the U components of Theta
    
    np.random.seed(seed)
    noise_var = 0
    my_map = BlackBox(map_def, noise_var=noise_var)
    mean, cov = np.zeros(ndim), np.ones(ndim)
    domain = [ [-6, 6] ] * ndim
    inputs = GaussianInputs(domain, mean, cov)
    
    if iter_num == 0:
        Theta = inputs.draw_samples(n_init, init_method)
    
    
    #Theta = inputs.draw_samples(50, "grd")
    noise = Noise([0,1], sigma=0.1, ell=1)
    
    # Need to determine U
    nsteps = int(T/dt)
    x_vals = np.linspace(0, 1, nsteps+1)
    x_vals = x_vals[0:-1]
    
    # DeepONet only needs a coarse version of the signal    
    coarse = 4
    
    # Create the X to U map, which is actually theta to U
    multiplier = 3*10**-9 # Special for the map
    
    
    def Theta_to_U(Theta,nsteps,coarse,udim):
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
    
    
    def Theta_to_Z(Theta,udim):
        if Theta.shape[1] == udim:
            Z = np.ones((Theta.shape[0], 1))
        else:
            Z = Theta[:,(udim+1):Theta.shape[1]]
        return Z         
    
    if iter_num == 0:

        # Determine the training data
        Y = np.zeros((n_init,))
        Us = Theta_to_U(Theta,nsteps,1,2)+2.55
        Us = Us*multiplier
    
        for i in range(0,n_init):
            I_temp = map_def(Us[i,:],gamma,delta,N_people,I0,T,dt, np.zeros(np.shape(Us[i,:])))
            Y[i] = I_temp[-1]
    
        Y = Y.reshape(n_init,1)
    
    
    m       = int(nsteps/coarse) #604*2
    lr      = 0.001
    dim_x   = 1
    activation = "relu"
    branch      = [neurons]*(b_layers+1)
    branch[0]   = m
    trunk       = [neurons]*(t_layers+1)
    trunk[0]    = dim_x
    
    net = dde.maps.OpNN(
        branch,
        trunk,
        activation,
        "Glorot normal",
        use_bias=True,
        stacked=False,
    )
    save_period = 1000
         
    # These functions are defined for normalizing, standardizing, or flatenining interal to DeepONet

    def DNO_Y_transform(x):
        x_transform = np.log10(x)/10 - 0.5
        return x_transform

    def DNO_Y_itransform(x_transform):
        x = 10**((x_transform+0.5)*10)
        return x
    
    # Keeping track of the metric
    pys = np.zeros((iters_max,10000))
    
    # Loop through iterations
    for iter_num in range(0,iters_max):
        # Train the model
        np.random.seed(np.size(Y))
        model_dir = './'
        model_str = ''
        model = DeepONet(Theta, nsteps, Theta_to_U, Theta_to_Z, Y, net, lr, epochs, N, model_dir, seed, save_period, model_str, coarse, udim, DNO_Y_transform, DNO_Y_itransform)
        training_data = model.training() # Get the training/loss values from the learning process

        # Pull a fine set of test_pts in the domain
        test_pts = 150
        Theta_test = inputs.draw_samples(test_pts, "grd")
        # Predict
        Mean_Val, Var_Val = model.predict(Theta_test)
        
        # Determine Bounds for evaluzting the metric
        x_max = np.max(Mean_Val)
        x_min = np.min(Mean_Val)
        x_int = np.linspace(x_min,x_max,10000) # Linearly space points
        x_int_standard = np.linspace(0,10**8,10000) # Static for pt-wise comparisons

        # Create the weights/exploitation values
        px = inputs.pdf(Theta_test)
        sc = scipy.stats.gaussian_kde(Mean_Val.reshape(test_pts**2,), weights=px)   # Fit a guassian kde using px input weights
        py = sc.evaluate(x_int) # Evaluate at x_int
        py[py<10**-16] = 10**-16 # Eliminate spuriously small values (smaller than numerical precision)
        py_standard = sc.evaluate(x_int_standard) # Evaluate for pt-wise comparisons
        py_interp = InterpolatedUnivariateSpline(x_int, py, k=1) # Create interpolation function
        
        # Conctruct the weights
        wx = px.reshape(test_pts**2,)/py_interp(Mean_Val).reshape(test_pts**2,)
        wx = wx.reshape(test_pts**2,1)
        
        # Compute the acquisition values
        ax = wx*Var_Val  # This is simply w(\theta) \sigma^2(\theta) - note that x and \theta are used interchangably
        
        # Find the optimal acquisition point
        Theta_opt = Theta_test[np.argmax(ax),:]
        Theta_opt = Theta_opt.reshape(1,2)        
        
        # Calculate the associated U
        U_opt = Theta_to_U(Theta_opt,nsteps,1,2)+2.55
        U_opt = U_opt*multiplier
        U_opt = U_opt.reshape(np.size(U_opt),1)
    
        # Pass to the Map
        I_temp = map_def(U_opt,gamma,delta,N_people,I0,T,dt,np.zeros(np.shape(U_opt)))
        Y_opt = I_temp[-1]
        Y_opt = Y_opt.reshape(1,1)
    
        # Append the value for the next step
        Theta = np.append(Theta, Theta_opt, axis = 0)
        Y = np.append(Y, Y_opt, axis = 0)
        pys[iter_num,:] = py_standard
        sio.savemat('SIR_Seed_'+str(seed)+'_N'+str(N)+'_iter_'+str(iter_num)+'.mat', {'pys':pys, 'x_int_standard':x_int_standard, 'Theta':Theta, 'U_opt':U_opt, 'I_temp':I_temp, 'wx':wx, 'ax':ax, 'py':py, 'x_int':x_int, 'Y':Y, 'Mean_Val':Mean_Val, 'Var_Val':Var_Val, 'n_init':n_init, 'N':N, 'seed':seed, 'Theta_test':Theta_test, 'training_data':training_data})

    return

# Call the function
main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,iters_max)


