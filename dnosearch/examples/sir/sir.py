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

iter_num    = 0
dim         = 2
acq         = 'US_LW'
n_init      = 3
epochs      = 1000
b_layers    = 8
t_layers    = 1
neurons     = 300
init_method = 'pdf'
N           = 2
seed        = 3

def map_def(beta,gamma,delta,N,I0,T,dt,f):    
    S = np.zeros((int(T/dt),));
    S[0] = N;
    I= np.zeros((int(T/dt),));
    I[0] = I0;
    R = np.zeros((int(T/dt),));
    for tt in range(0,np.size(S)-1):
        # Equations of the model
        dS = (-beta[tt]*I[tt]*S[tt] + delta*R[tt] - f[tt]) * dt;
        dI = (beta[tt]*I[tt]*S[tt] - gamma*I[tt] + f[tt]) * dt;
        dR = (gamma*I[tt] - delta*R[tt]) * dt;
        S[tt+1] = S[tt] + dS;
        I[tt+1] = I[tt] + dI;
        R[tt+1] = R[tt] + dR;
    return I


def main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N):
    
    T = 75
    dt = 0.2
    gamma = 0.1
    delta = 0
    N_people = 10**8
    I0 = 50
    
    ndim = dim
    rank = dim
    
    np.random.seed(seed)
    noise_var = 0
    my_map = BlackBox(map_def, noise_var=noise_var)
    mean, cov = np.zeros(ndim), np.ones(ndim)
    domain = [ [-6, 6] ] * ndim
    inputs = GaussianInputs(domain, mean, cov)
    
    if iter_num == 0:
        Theta = inputs.draw_samples(n_init, init_method)
    
    
    #Theta = inputs.draw_samples(50, "grd")
    noise = Noise([0,1], sigma=0.1, ell=0.1)
    
    # Need to determine U
    nsteps = int(T/dt)
    x_vals = np.linspace(0, 1, nsteps+1)
    x_vals = x_vals[0:-1]
    
    # DeepONet only needs a coarse version of the signal    
    coarse = 4
    
    # Create the X to U map, which is actually theta to U
    multiplier = 2*10**-9 # Special for the map
    
    
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
        Y = np.zeros((n_init,))
        Us = Theta_to_U(Theta,nsteps,1,2)+2.15
        Us = Us*multiplier
    
        for i in range(0,n_init):
            I_temp = map_def(Us[i,:],gamma,delta,N_people,I0,T,dt, np.zeros(np.shape(Us[i,:])))
            Y[i] = np.log10(I_temp[-1])/10 - 0.5
    
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
    
    model_dir = './'
    save_str = 'coarse'+str(coarse)+'_InitMethod_'+init_method # This alters the string for the model saving
     
    
    def Mean_Real(Mean_Val):
        Real = 10**((Mean_Val+0.5)*10)
        return Real

    # Now we loop through the iterations
    iters_max = 10
    # Keeping track of the objective function
    py = np.zeros((10,10000))
    
    
    for iter_num in range(0,iters_max):
        # Train the model
        np.random.seed(np.size(Y))
        model_str = 'Rank'+str(np.shape(Theta)[1])+'_'+save_str+'_'+acq+'_Iter'+str(np.size(Y)-n_init+1)
        model_str = ''
        model = DeepONet(Theta, nsteps, Theta_to_U, Theta_to_Z, Y, net, lr, epochs, N, model_dir, seed, save_period, model_str, coarse, rank, Mean_Real)
    
        test_pts = 150
        Thetanew = inputs.draw_samples(test_pts, "grd")
        Mean_Val, Var_Val = model.predict(Thetanew)
        #Real_Val = 10**((Mean_Val+0.5)*10)
        Real_Val = Mean_Val
        
        x_max = np.max(Real_Val)
        x_min = np.min(Real_Val)
        x_int = np.linspace(x_min,x_max,10000)
        
        fx = inputs.pdf(Thetanew)
        sc = scipy.stats.gaussian_kde(Real_Val.reshape(test_pts**2,), weights=fx)
        y = sc.evaluate(x_int)
        y[y<10**-16] = 10**-16
        fy_interp = InterpolatedUnivariateSpline(x_int, y, k=1)
        wx = fx.reshape(test_pts**2,)/fy_interp(Real_Val).reshape(test_pts**2,)
        wx = wx.reshape(test_pts**2,1)
        
        # Find the optimal acquisition point
        Theta_opt = Thetanew[np.argmax(wx*Var_Val),:]
        Theta_opt = Theta_opt.reshape(1,2)        
        
        # Calculate the U
        U_opt = Theta_to_U(Theta_opt,nsteps,1,2)+2.15
        U_opt = U_opt*multiplier
        U_opt = U_opt.reshape(np.size(U_opt),1)
    
        # Pass to the Map
        I_temp = map_def(U_opt,gamma,delta,N_people,I0,T,dt,np.zeros(np.shape(U_opt)))
        Y_opt = np.log10(I_temp[-1])/10 - 0.5
        Y_opt = Y_opt.reshape(1,1)
    
        # Append the value for the next step
        Theta = np.append(Theta, Theta_opt, axis = 0)
        Y = np.append(Y, Y_opt, axis = 0)
        training_data = model.training()
        py[iter_num,:] = y
    
    
    sio.savemat(str(N)+'.mat', {'Theta':Theta, 'U_opt':U_opt, 'I_temp':I_temp, 'wx':wx, 'y':y, 'x_int':x_int, 'Y':Y, 'Mean_Val':Mean_Val, 'Real_Val':Real_Val, 'Var_Val':Var_Val, 'n_init':n_init, 'N':N, 'seed':seed, 'Thetanew':Thetanew, 'training_data':training_data})
    return

# Call the function
main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N)


