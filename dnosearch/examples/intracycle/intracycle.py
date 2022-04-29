#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:36:27 2022

X% CDF Optimizer for the Intracycle Data 

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

def main_init(seed,acq):

    #seed        = 1
    dim         = 2
    acquisition = acq
    epochs      = 1000
    b_layers    = 3
    t_layers    = 1
    neurons     = 50
    init_method = 'lhs'
    N           = 2
    n_init      = 3
    iter_num    = 0
    n_keep      = 1
    cdf_val     = 0.75
    iters_max   = 50

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
    
    # Creating the test points and the truth data
    test_pts = 100
    Thetanew = inputs.draw_samples(test_pts, "grd")
    Y_truth = map_def(Thetanew).reshape(test_pts**ndim,1)
    # Calculate the truth CDF cut off
    x_max = np.max(Y_truth)
    x_min = np.min(Y_truth)
    x_int = np.linspace(1.25*x_min,1.25*x_max,1024)
    
    sc = scipy.stats.gaussian_kde(Y_truth.reshape(test_pts**ndim,))
    y = sc.evaluate(x_int)
        
    # Lets create the CDF
    dx = np.diff(x_int)
    x_Y_truth = x_int
    dx = np.append(dx,0)
    area = dx*y
    cdf = np.zeros(1024,)
    for i in range(0,1024):
        cdf[i] = np.sum(area[0:i])
    
    temp = np.ones((1024,))*cdf
    temp[temp > cdf_val] = 0 # VERY IMPORTANT
    indice = np.argmax(temp)
    mu_cut_Y_truth = x_Y_truth[indice]
    Y_cdf = np.zeros(np.shape(Y_truth))
    Y_cdf = Y_truth.copy()
    Y_cdf[Y_truth < mu_cut_Y_truth] = 0
    
    
    
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
    
    M = 1 # Number of snapshot ensembles
    save_period = 1000
    
    # bananas
    model_dir = './model/'
    save_str = 'coarse'+str(coarse)+'_InitMethod_'+init_method 
    base_dir = './data/'
    save_path_data = base_dir+'Wave_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_batch'+str(n_keep)+'_N'+str(N)+'_Iteration'+str(iter_num)+'.mat'
    load_path_data = base_dir+'Wave_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_batch'+str(n_keep)+'_N'+str(N)+'_Iteration'+str(iter_num-1)+'.mat'
        
    print(np.shape(Theta))
    print(np.shape(Y)) 
    
    model_str = 'Rank'+str(np.shape(Theta)[1])+'_'+save_str+'_'+acq+'_Iter'+str(np.size(Y)-n_init+1)
    
    MSE_CDF = np.zeros((iters_max,))
    Means = np.zeros((test_pts**2,iters_max))
    Vars = np.zeros((test_pts**2,iters_max))
    Wxs = np.zeros((test_pts**2,iters_max))
    Acqs = np.zeros((test_pts**2,iters_max))
    Means_CDF = np.zeros((test_pts**2,iters_max))
    mu_cuts = np.zeros((iters_max,))
    x_ints = np.zeros((1024,iters_max))
    ys = np.zeros((1024,iters_max))
    cdfs = np.zeros((1024,iters_max))
    
    for iters in range(0,iters_max):
        model = DeepONet(Theta, nsteps, Theta_to_U, Theta_to_Z, Y, net, lr, epochs, N, model_dir, seed, save_period, model_str, coarse, udim, DNO_Y_transform, DNO_Y_itransform)
        
    
        Mean_Val, Var_Val = model.predict(Thetanew)
        
        x_max = np.max(Mean_Val)
        x_min = np.min(Mean_Val)
        x_int = np.linspace(1.25*x_min,1.25*x_max,1024)
        
        sc = scipy.stats.gaussian_kde(Mean_Val.reshape(test_pts**ndim,))
        y = sc.evaluate(x_int)
            
        n_guess = n_keep # Typically restarts but will fix later  mayeb
    
        # Lets create the CDF
        dx = np.diff(x_int)
        x_truth = x_int
        dx = np.append(dx,0)
        area = dx*y
        cdf = np.zeros(1024,)
        for i in range(0,1024):
            cdf[i] = np.sum(area[0:i])
    
        temp = np.ones((1024,))*cdf
        temp[temp > cdf_val] = 0 # VERY IMPORTANT
        indice = np.argmax(temp)
        mu_cut = x_truth[indice]
        
        #acquisition = 'cdf'
        beta = 1#10000**ndim
        var_vals = Var_Val
        mean_vals = Mean_Val
        n_monte = test_pts**ndim
        
        if acquisition == "us":
            scores = -  var_vals.reshape(n_monte,)
        elif acquisition == "uslw":
            print('Not implemented')
            #scores = - wx * var_vals.reshape(n_monte,)
        elif acquisition == "bandit_us":
            #gamma = np.max(mean_vals) / np.max(var_vals) # Dynamic gamma
            gamma = 1
            scores = - (mean_vals.reshape(n_monte,) +  gamma * var_vals.reshape(n_monte,))
        elif acquisition == "bandit_uslw":
            print('Not implemented')
            #scores = - (mean_vals.reshape(n_monte,) + np.max(mean_vals) / np.max(var_vals)  / np.max(wx) * wx * var_vals.reshape(n_monte,))
        elif acquisition == "cdf":
            mean_vals = mean_vals.reshape(n_monte,)
            wx = np.zeros(np.shape(mean_vals))
            wx[mean_vals > mu_cut] = beta
            wx[mean_vals <= mu_cut] = 0
            #scores = - wx * var_vals.reshape(n_monte,)
            scores = -wx.reshape(n_monte,)*Var_Val.reshape(n_monte,)
        
        sorted_idxs = np.argsort(scores,axis = 0)
        scores = scores.reshape(n_monte,)
    
        # New version where we impose the radius earlier
        sorted_scores = scores[sorted_idxs[0:n_monte]]
        sorted_x0 = Thetanew[sorted_idxs[0:n_monte], :]
        n_counter = 0
        
        x0_guess = np.zeros((n_guess,ndim))
        score_guess = np.zeros((n_guess,))
        
        x0_guess[0,:] = sorted_x0[0,:]
        score_guess[0] = sorted_scores[0]
    
        # Now we need to remove the optimal from consideration, and remove values within a radius of influence
        max_domain_distance = np.sqrt((inputs.domain[1][1]-inputs.domain[0][0])**2*ndim)
        r_val = 0.025*max_domain_distance
    
        for i in range(1,n_guess):
            # Now remove the optimal value
            sorted_x0 = np.delete(sorted_x0, 0, axis=0)
            sorted_scores = np.delete(sorted_scores, 0)
            distances = np.zeros((np.size(sorted_scores),))
            
            for j in range(0,min(1000,np.size(sorted_scores))):  # Now this is not perfect becuase it does not eliminate all points to be considered with the radius, but should be more robust
                distances[j] = distance.euclidean(x0_guess[i-1,:], sorted_x0[j,:])
            sorted_x0 = sorted_x0[distances > r_val,:]
            sorted_scores = sorted_scores[distances > r_val]
            x0_guess[i,:] = sorted_x0[0,:]
            score_guess[i] = sorted_scores[0]
            
        scores_opt = score_guess
        theta_opt = x0_guess
        
        Theta_opt = theta_opt.reshape(1,ndim)    
        Y_opt = map_def(Theta_opt).reshape(1,1)
                
        
        # plt.pcolor(Thetanew[:,0].reshape(test_pts,test_pts),Thetanew[:,1].reshape(test_pts,test_pts),Mean_Val.reshape(test_pts,test_pts))
        # plt.title('Mean Model Prediction')
        # plt.colorbar()
        # plt.plot(Theta[:,0], Theta[:,1], 'wo')
        # plt.title('Iterations:'+str(np.size(Y)-n_init))
        # plt.show()
        # plt.pcolor(Thetanew[:,0].reshape(test_pts,test_pts),Thetanew[:,1].reshape(test_pts,test_pts),wx.reshape(test_pts,test_pts))
        # plt.title('CDF Cut')
        # plt.show()
        # plt.pcolor(Thetanew[:,0].reshape(test_pts,test_pts),Thetanew[:,1].reshape(test_pts,test_pts),Var_Val.reshape(test_pts,test_pts)/np.max(Var_Val))
        # plt.title('Model Variance')
        # plt.plot(Theta[:,0], Theta[:,1], 'wo')
        # plt.show() 
        # plt.pcolor(Thetanew[:,0].reshape(test_pts,test_pts),Thetanew[:,1].reshape(test_pts,test_pts),-scores.reshape(test_pts,test_pts))
        # plt.title('Acquisition Function')
        # plt.plot(Theta_opt[0,0], Theta_opt[0,1],  'ro')
        # plt.show()
        
    
        Y_NN_cdf = Mean_Val.copy()
        Y_NN_cdf[Y_truth < mu_cut_Y_truth] = 0
        
        MSE_CDF[iters] = np.sum((Y_cdf - Y_NN_cdf)**2) / np.sum(Y_truth > mu_cut_Y_truth)
        print(MSE_CDF[-1])
        
        Theta = np.append(Theta, Theta_opt, axis = 0)
        Y = np.append(Y, Y_opt, axis = 0)
    
        Means[:,iters] = Mean_Val.reshape(test_pts**ndim,)
        Vars[:,iters] = Var_Val.reshape(test_pts**ndim,)
        Wxs[:,iters] = wx.reshape(test_pts**ndim,)
        Acqs[:,iters] = -scores.reshape(test_pts**ndim,)
        Means_CDF[:,iters] = Y_NN_cdf.reshape(test_pts**ndim,)
        mu_cuts[iters] = mu_cut 
        x_ints[:,iters] = x_int.reshape(1024,)
        ys[:,iters] = y.reshape(1024,)
        cdfs[:,iters] = cdf.reshape(1024,)
        
    # plt.plot(np.log10(MSE_CDF))
    
    sio.savemat('./data/Intracycle_Seed'+str(seed)+'_'+acquisition+'_'+str(cdf_val)+'_Opt_N'+str(N)+'.mat', 
                {'Means':Means, 'Vars':Vars, 'Wxs':Wxs, 'Acqs':Acqs, 'Means_CDF':Means_CDF, 'mu_cuts':mu_cuts, 'x_ints':x_ints, 'ys':ys, 'cdfs':cdfs, 
                 'MSE_CDF':MSE_CDF, 'Y':Y, 'Theta':Theta, 'Y_cdf':Y_cdf, 'Y_truth':Y_truth, 'Y_NN_cdf':Y_NN_cdf, 'mu_cut_Y_truth':mu_cut_Y_truth})
    
main_init(int(sys.argv[1]), sys.argv[2])    