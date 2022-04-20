#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:42:31 2022

Example for N-dimensional 

@author: ethanpickering
"""
    
import numpy as np
from dnosearch import (BlackBox, UniformInputs, DeepONet, custom_KDE, Oscillator)
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
from scipy.spatial import distance
import shutil

#%% The map we are defining here is
# Input   = sum_i^N sin(x+phi_0+theta_i)
# Output  = sum_i^N sin(x(end)+phi_0+theta_i) = sum_i^N sin(2*pi+phi_0+theta_i)

# This I-O relationshsip means we are interested in an identify mapping
# of the last point in the input signal

phi_0 = -np.pi/2
wavenumber = 1

def map_def(Theta):
    phi_0 = -np.pi/2
    wavenumber = 1
    f = np.zeros((np.shape(Theta)[0],1))
    Theta = Theta.reshape(np.size(Theta),1)
    for j in range(0,np.shape(Theta)[0]):
        for i in range(0,np.shape(Theta)[1]):
            f[j] = f[j] + np.sin(wavenumber*(2*np.pi+phi_0+Theta[j,i]))
    return f

# Set various parameters
seed     = 1
dim      = 1
acq      = 'US'
epochs   = 1000
b_layers = 3
t_layers = 1
neurons  = 50
init_method = 'lhs'
N        = 2
n_init   = 2
iter_num = 0
n_keep   = 1
# Use the functional representation or the parameter representations
func_or_param = 'param'
iters_max = 10



ndim    = dim
dim_u   = ndim  # Dimension of Theta_U, note that the dimension of Theta_Z is ndim - dim_u
mean, cov = np.zeros(ndim), np.ones(ndim)
domain = [ [0, 2*np.pi] ] * ndim

inputs = UniformInputs(domain)
np.random.seed(seed)

if iter_num == 0:
    Theta = inputs.draw_samples(n_init, init_method)

noise_var = 0
my_map = BlackBox(map_def, noise_var=noise_var)

if func_or_param == 'func':
    # Need to determine U
    nsteps = 20
    x_vals = np.linspace(0, 1, nsteps)
    x_vals = x_vals[0:-1]    
    # DeepONet only needs a coarse version of the signal
    coarse = 1 # Lets keep it the same for now
    
    def Theta_to_U(Theta,nsteps,coarse,dim_u):
        U = np.zeros((np.shape(Theta)[0],nsteps))
        phi_0 = -np.pi/2
        wavenumber = 1
        x = np.linspace(0,2*np.pi,nsteps) + phi_0
        print(Theta)
        for j in range(0,np.shape(Theta)[0]):
            for i in range(0,np.shape(Theta)[1]):
                U[j,:] = U[j,:] + np.sin(wavenumber*(x+phi_0+Theta[j,i]))
        return U

else:
    nsteps = ndim
    x_vals = np.linspace(0, 1, nsteps)
    x_vals = x_vals[0:-1]    
    # DeepONet only needs a coarse version of the signal
    coarse = 1 # Lets keep it the same for now
    def Theta_to_U(Theta,nsteps,coarse,dim_u):
        U = Theta[:,0:dim_u]
        return U
    


def Theta_to_Z(Theta,dim_u):
    if Theta.shape[1] == dim_u:
        Z = np.ones((Theta.shape[0], 1))
    else:
        Z = Theta[:,dim_u:Theta.shape[1]]
    return Z
    
if iter_num == 0:
    # Determine the training data
    Us = Theta_to_U(Theta,nsteps,1,ndim)
    
    #Y = map_def(Theta,phi_0,wavenumber).reshape(n_init,1)
    Y = my_map.evaluate(Theta,ndim)

m       = int(nsteps/coarse)
lr      = 0.001
dim_x   = 1
activation = "relu"
branch      = [neurons]*(b_layers+1)
branch[0]   = m
trunk       = [neurons]*(t_layers+1)
trunk[0]    = dim_x

# Old ersion
net = dde.maps.OpNN(
# New version
#net = dde.maps.DeepONet(
    branch,
    trunk,
    activation,
    "Glorot normal",
    use_bias=True,
    stacked=False,
)

save_period = 1000


# Below are saving directories needed for DeepONet, they can be changed to anything you wish
model_dir = './'
save_str = 'Wave_NN_model'
base_dir = './'
# The model str helps delineate between different seeds and acquisition functions, but can be anything.
model_str = ''
#model_str = 'Rank'+str(np.shape(Theta)[1])+'_'+save_str+'_'+acq+'_Iter'+str(np.size(Y)-n_init+1)

print(np.shape(Theta))
print(np.shape(Y))
 

for iters in range(0,iters_max):
    model = DeepONet(Theta, nsteps, Theta_to_U, Theta_to_Z, Y, net, lr, epochs, N, model_dir, seed, save_period, model_str, coarse, dim_u)
    
    test_pts = 100
    Thetanew = inputs.draw_samples(test_pts, "grd")
    Mean_Val, Var_Val = model.predict(Thetanew)
    
    
    plt.plot(Thetanew,Mean_Val)
    plt.title('Mean Model Prediction')
    plt.plot(Theta,Y,'o')
    plt.title('Iterations:'+str(np.size(Y)-n_init))
    plt.show()
    plt.plot(Thetanew,Var_Val/np.max(Var_Val), 'r')
    plt.title('Model Variance')
    plt.plot(Theta,np.zeros(np.size(Y)), 'o')
    Theta_opt = Thetanew[np.argmax(Var_Val)]
    plt.plot(Theta_opt, 1, 'o')
    plt.show()
    
    # We will simply impose the US sampling technique
    Theta_opt = Thetanew[np.argmax(Var_Val)]
    Theta_opt = Theta_opt.reshape(1,ndim)
    Y_opt = my_map.evaluate(Theta_opt,ndim)

    
    Theta = np.append(Theta, Theta_opt, axis = 0)
    Y = np.append(Y, Y_opt, axis = 0)

    # Remove the temporary folder for saving DeepONet files
    shutil.rmtree('./model/')