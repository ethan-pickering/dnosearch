#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 11:53:54 2022
# 
@author: ethanpickering
"""
# dnosearch Imports
import numpy as np
from dnosearch import (BlackBox, UniformInputs, DeepONet)
from oscillator import Noise


# DeepONet Imports
import deepxde as dde

# Other Imports
import sys
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.io as sio
import h5py
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import distance



def map_def(theta):
    d = sio.loadmat('./IntracycleData.mat')
    eff = d['eff']
    amp = d['amplitude']
    phi = d['phi']
    eff = eff.reshape(462,1) - 0.15
    amp = amp.reshape(462,1)
    phi = phi.reshape(462,1)
    val = griddata(np.append(amp,phi,axis=1), eff, theta, method='linear', fill_value=0)
    return val

seed        = 1
dim         = 2
acq         = 'US'
epochs      = 1000
b_layers    = 3
t_layers    = 1
neurons     = 50
init_method = 'lhs'
N           = 5
n_init      = 3
iter_num    = 0
n_keep      = 1
iters_max   = 25


ndim = dim
udim = ndim
branch_dim = ndim  # This refers to rank in the underlying codes
domain = [ [0.25, 13], [0, 6.1] ] # Domain of amplitude and the phase

inputs = UniformInputs(domain)
np.random.seed(seed) # Set the seet

if iter_num == 0:
    Theta = inputs.draw_samples(n_init, init_method)
    if init_method == 'grd':
        n_init = n_init**2 # If grd, we need to square the init values

noise_var = 0
my_map = BlackBox(map_def, noise_var=noise_var)


# Determine the input signal, which must be discretized
nsteps = 50 # Choose the number of steps of the signal -     
# DeepONet only needs a coarse version of the signal  
# Thus, we can coarsen it for computational advantages  
coarse = 1 # Lets keep it the same for now

# Converts the Theta_u Parameters to Branch functions: U
def Theta_to_U(Theta,nsteps,coarse,rank):
    U = np.zeros((np.shape(Theta)[0],nsteps))
    x = np.linspace(0,2*np.pi,nsteps)
    print(Theta)
    for j in range(0,np.shape(Theta)[0]):
        U[j,:] =  Theta[j,0]*np.sin(x+Theta[j,1]) / 13
        # For some reason... this was working when only the first index was used... which was the direct answer I believe
    return U

# Converts the Theta_z Parameters to Trunk Functions 
def Theta_to_Z(Theta,rank):
    if Theta.shape[1] == rank:
        Z = np.ones((Theta.shape[0], 1))
    else:
        Z = Theta[:,(2*rank):Theta.shape[1]]
    return Z         
    
if iter_num == 0:
    # Determine the training data
    Us = Theta_to_U(Theta,nsteps,1,ndim)
    Y = map_def(Theta).reshape(n_init,1)

def DNO_Y_transform(x):
    x_transform = x
    return x_transform

def DNO_Y_itransform(x_transform):
    x = x_transform
    return x        


# Set the Neural Operator Parameters
m       = int(nsteps/coarse) # Number of sensor inputs
lr      = 0.001 # Learning Rate
dim_x   = 1 # Dimensionality of the operator values 
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

model_dir = './model/'
save_str = 'coarse'+str(coarse)+'_InitMethod_'+init_method 
base_dir = './data/'
save_path_data = base_dir+'Wave_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_batch'+str(n_keep)+'_N'+str(N)+'_Iteration'+str(iter_num)+'.mat'
load_path_data = base_dir+'Wave_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_batch'+str(n_keep)+'_N'+str(N)+'_Iteration'+str(iter_num-1)+'.mat'
    
print(np.shape(Theta))
print(np.shape(Y)) 

model_str = 'Rank'+str(np.shape(Theta)[1])+'_'+save_str+'_'+acq+'_Iter'+str(np.size(Y)-n_init+1)

for iters in range(0,iters_max):
    model = DeepONet(Theta, nsteps, Theta_to_U, Theta_to_Z, Y, net, lr, epochs, N, model_dir, seed, save_period, model_str, coarse, udim, DNO_Y_transform, DNO_Y_itransform)

    test_pts = 100
    Thetanew = inputs.draw_samples(test_pts, "grd")
    Mean_Val, Var_Val = model.predict(Thetanew)
    
    plt.pcolor(Thetanew[:,0].reshape(test_pts,test_pts),Thetanew[:,1].reshape(test_pts,test_pts),Mean_Val.reshape(test_pts,test_pts))
    plt.title('Mean Model Prediction')
    plt.colorbar()
    #plt.plot(Theta[:,0], Theta[:,1], 'o')
    plt.title('Iterations:'+str(np.size(Y)-n_init))
    plt.show()
    plt.pcolor(Thetanew[:,0].reshape(test_pts,test_pts),Thetanew[:,1].reshape(test_pts,test_pts),Var_Val.reshape(test_pts,test_pts)/np.max(Var_Val))
    plt.title('Model Variance')
    plt.plot(Theta[:,0], Theta[:,1], 'o')
    Theta_opt = Thetanew[np.argmax(Var_Val)]
    plt.plot(Theta_opt[0], Theta_opt[1],  'o')
    plt.show() 
    plt.draw()
    
    # We will simply impose the US sampling technique
    Theta_opt = Thetanew[np.argmax(Var_Val)]
    Theta_opt = Theta_opt.reshape(1,ndim)
    #Us_opt = Theta_to_U(Theta_opt,nsteps,1,ndim)
    Y_opt = map_def(Theta_opt).reshape(1,1)
    
    Theta = np.append(Theta, Theta_opt, axis = 0)
    Y = np.append(Y, Y_opt, axis = 0)