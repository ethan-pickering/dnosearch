#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:06:32 2022

@author: ethanpickering
"""

# GPSearch Imports
import numpy as np
from gpsearch import (BlackBox, GaussianInputs, DeepONet, custom_KDE, Oscillator)


from oscillator import Noise


# DeepONet Imports
import deepxde as dde
from utils import mean_squared_error_outlier, safe_test, trim_to_65535

# Other Imports
import sys
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt

iter_num    = int(sys.argv[2])
dim         = int(sys.argv[3])
acq         = sys.argv[4]
n_init      = int(sys.argv[5])
epochs      = int(sys.argv[6])
b_layers    = int(sys.argv[7])
t_layers    = int(sys.argv[8])
neurons     = int(sys.argv[9])
init_method = sys.argv[10]
N           = int(sys.argv[11])
upper_limit = int(sys.argv[12])


def map_def(beta,gamma,delta,N,I0,T,dt,f):    
    S = np.zeros((int(T/dt),));
    S[0] = N;
    I= np.zeros((int(T/dt),));
    I[0] = I0;
    R = np.zeros((int(T/dt),));
    for tt in range(0,np.size(S)-1):
        # Equations of the model
        #dS = (-beta[tt]*I[tt]*S[tt] + delta*R[tt]) * dt;
        dS = (-beta[tt]*I[tt]*S[tt] + delta*R[tt] - f[tt]) * dt;
        #dI = (beta*I(tt)*S(tt) - gamma*I(tt)) * dt;
        #dI = (beta[tt]*I[tt]*S[tt] - gamma*I[tt]) * dt;
        dI = (beta[tt]*I[tt]*S[tt] - gamma*I[tt] + f[tt]) * dt;
        dR = (gamma*I[tt] - delta*R[tt]) * dt;
        S[tt+1] = S[tt] + dS;
        I[tt+1] = I[tt] + dI;
        R[tt+1] = R[tt] + dR;
    return I


def main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,upper_limit):
    
    T = 75
    dt = 0.2
    gamma = 0.1
    delta = 0
    N_people = 10*10**7
    I0 = 50
    
    ndim = dim
    rank = dim
    #seed = 1
    #ndim = 2
    #n_init = 5
    
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
    
    
    def Theta_to_X(Theta,rank):
        if Theta.shape[1] == rank:
            X = np.ones((Theta.shape[0], 1))
        else:
            X = Theta[:,(2*rank):Theta.shape[1]]
        return X         
    
    if iter_num == 0:

        # Determine the training data
        Y = np.zeros((n_init,))
        Us = Theta_to_U(Theta,nsteps,1,2)+2.15
        Us = Us*multiplier
    
        for i in range(0,n_init):
            I_temp = map_def(Us[i,:],gamma,delta,N_people,I0,T,dt, np.zeros(np.shape(Us[i,:])))
            Y[i] = np.log10(I_temp[-1])/10 - 0.5
    
        Y = Y.reshape(n_init,1)
    
    #b_layers = 5
    #t_layers = 1
    
    m       = int(nsteps/coarse) #604*2
    #neurons = 200
    #epochs  = 1000
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
    
    M = 1
    save_period = 1000
    
    # bananas
    #model_dir = '/Users/ethanpickering/Documents/git/gpsearch_pickering/gpsearch/examples/sir/models/'
    model_dir = '/Users/ethanpickering/Documents/Pandemic_Model_Bin/models/'
    #acq = 'lhs'
    save_str = 'coarse'+str(coarse)+'_InitMethod_'+init_method+'_upperlimit'+str(upper_limit)  # This alters the string for the model saving
    base_dir = '/Users/ethanpickering/Dropbox (MIT)/Pandemic/Runs/'
    #model_dir = scratch_or_research+'epickeri/MMT/Data/Rank'+str(rank)+'/models/'
    #save_dir = scratch_or_research+'epickeri/MMT/Data/Rank'+str(rank)+'/DON_Search/'   # Not Sure this is used anymore             
    #save_str = 'coarse'+str(coarse)+'_lam'+str(lam)+'_BatchSize'+str(batch_size)+'_OptMethod_'+init_method+'_nguess'+str(n_guess)+'_'+objective  # This alters the string for the model saving
    save_path_data = base_dir+'SIR_T75_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_N'+str(N)+'_Iteration'+str(iter_num)+'.mat'
    load_path_data = base_dir+'SIR_T75_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_N'+str(N)+'_Iteration'+str(iter_num-1)+'.mat'

    
    if iter_num > 0:
        d = sio.loadmat(load_path_data)
        Theta = d['Theta']
        Y = d['Y']
        
    print(np.shape(Theta))
    print(np.shape(Y)) 
     
    # I dont think this does anything     
    np.random.seed(np.size(Y))
    
     
    model_str = 'Rank'+str(np.shape(Theta)[1])+'_'+save_str+'_'+acq+'_Iter'+str(np.size(Y)-n_init+1)
    model = DeepONet(Theta, nsteps, Theta_to_U, Theta_to_X, Y, net, lr, epochs, N, M, model_dir, seed, save_period, model_str, coarse, rank)
    
    test_pts = 150
    Thetanew = inputs.draw_samples(test_pts, "grd")
    Mean_Val, Var_Val = model.predict(Thetanew)
    Real_Val = 10**((Mean_Val+0.5)*10)
    if upper_limit == 1:
        Real_Val[Real_Val>N_people] = N_people

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

    sio.savemat(save_path_data, {'Theta':Theta, 'U_opt':U_opt, 'I_temp':I_temp, 'wx':wx, 'y':y, 'x_int':x_int, 'Y':Y, 'Mean_Val':Mean_Val, 'Real_Val':Real_Val, 'Var_Val':Var_Val, 'n_init':n_init, 'N':N, 'seed':seed, 'Thetanew':Thetanew, 'training_data':training_data})


if __name__ == "__main__":
    main(int(sys.argv[1]),iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,upper_limit)
