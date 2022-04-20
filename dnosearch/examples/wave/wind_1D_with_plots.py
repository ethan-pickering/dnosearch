#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:06:37 2022

@author: ethanpickering
"""
    
# DNOSearch Imports
import numpy as np
from dnosearch import (BlackBox, GaussianInputs, DeepONet, custom_KDE)
#from utils import Noise

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
n_keep      = int(sys.argv[13])


def main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,upper_limit,n_keep):
    # # seed = 1
    # # dim = 2
    # # acq = 'CDF_US'
    # # epochs = 1000
    # b_layers = 5
    # t_layers = 1
    # neurons = 200
    # #init_method = 'pdf'
    # N = 2
    
    ndim = dim
    rank = ndim
    d1 = 3
    mean, cov = np.zeros(ndim), np.ones(ndim)
    domain = [ [-d1, d1] ] * ndim
    inputs = GaussianInputs(domain, mean, cov)
    #Theta = inputs.draw_samples(100, "grd")
    
    if iter_num == 0:
        Theta = inputs.draw_samples(n_init, init_method)
        
    # Now we neeed to create the positive semidefinit matrix    
    np.random.seed(seed)
    mu = np.ones((ndim,))
    
    
    # Create random positive definit Sigma matrix
    Q = np.random.randn(ndim,ndim)
    for i in range(0,ndim):            
       Q[i,i] = np.abs(Q[i,i])+1*(i+1);
    
    Sigma = np.matmul(np.transpose(Q),Q)
    map_def = GaussianInputs(domain, mu.reshape(ndim,), Sigma)
    #obj = map_def.pdf(Theta)
    #plt.pcolor(obj.reshape(100,100))
    
    noise_var = 0
    my_map = BlackBox(map_def, noise_var=noise_var)
    
    #Theta = inputs.draw_samples(50, "grd")
    noise = Noise([0,1], sigma=0.1, ell=0.5)
    
    # Need to determine U
    nsteps = 100
    x_vals = np.linspace(0, 1, nsteps+1)
    x_vals = x_vals[0:-1]
    
    # DeepONet only needs a coarse version of the signal    
    coarse = 4
    
    #y = mvnpdf(X,mu,round(Sigma,2));
    
    def Theta_to_U(Theta,nsteps,coarse,rank):
        U = Theta
        return U


    def Theta_to_X(Theta,rank):
        if Theta.shape[1] == rank:
            X = np.ones((Theta.shape[0], 1))
        else:
            X = Theta[:,(2*rank):Theta.shape[1]]
        return X         
    
    if iter_num == 0:

        # Determine the training data
        Us = Theta_to_U(Theta,nsteps,1,2)
        Y = map_def.pdf(Theta).reshape(n_init,1)/10**-1
    
    #b_layers = 5
    #t_layers = 1
    
    m       = int(ndim) #604*2
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
    model_dir = '/Users/ethanpickering/Documents/Wind/models/'
    #acq = 'lhs'
    save_str = 'coarse'+str(coarse)+'_InitMethod_'+init_method #+'_upperlimit'+str(upper_limit)  # This alters the string for the model saving
    base_dir = '/Users/ethanpickering/Dropbox (MIT)/Wind/Runs/'
    #model_dir = scratch_or_research+'epickeri/MMT/Data/Rank'+str(rank)+'/models/'
    #save_dir = scratch_or_research+'epickeri/MMT/Data/Rank'+str(rank)+'/DON_Search/'   # Not Sure this is used anymore             
    #save_str = 'coarse'+str(coarse)+'_lam'+str(lam)+'_BatchSize'+str(batch_size)+'_OptMethod_'+init_method+'_nguess'+str(n_guess)+'_'+objective  # This alters the string for the model saving
    save_path_data = base_dir+'Wind_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_batch'+str(n_keep)+'_N'+str(N)+'_Iteration'+str(iter_num)+'.mat'
    load_path_data = base_dir+'Wind_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_batch'+str(n_keep)+'_N'+str(N)+'_Iteration'+str(iter_num-1)+'.mat'

    
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
    
    Thetanew = inputs.draw_samples(100, "grd")
    Mean_Val, Var_Val = model.predict(Thetanew)

    x_max = np.max(Mean_Val)
    x_min = np.min(Mean_Val)
    x_int = np.linspace(0.9*x_min,1.1*x_max,1024)
    
    sc = scipy.stats.gaussian_kde(Mean_Val.reshape(100**2,))
    y = sc.evaluate(x_int)
    #y[y<10**-16] = 10**-16
    #fy_interp = InterpolatedUnivariateSpline(x_int, y, k=1)
    #wx = fx.reshape(test_pts**2,)/fy_interp(Mean_Val).reshape(test_pts**2,)
    #wx = wx.reshape(test_pts**2,1)
    
    
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
    temp[temp > 0.8] = 0
    indice = np.argmax(temp)
    mu_cut = x_truth[indice]
    
    acquisition = 'cdf'
    beta = 100
    var_vals = Var_Val
    mean_vals = Mean_Val
    n_monte = 100**2
    
    if acquisition == "us":
        scores = -  var_vals.reshape(n_monte,)
    elif acquisition == "uslw":
        scores = - wx * var_vals.reshape(n_monte,)
    elif acquisition == "bandit_us":
        scores = - (mean_vals.reshape(n_monte,) + np.max(mean_vals) / np.max(var_vals) * var_vals.reshape(n_monte,))
    elif acquisition == "bandit_uslw":
        scores = - (mean_vals.reshape(n_monte,) + np.max(mean_vals) / np.max(var_vals)  / np.max(wx) * wx * var_vals.reshape(n_monte,))
    elif acquisition == "bandit_cdf":
        mean_vals = mean_vals.reshape(n_monte,)
        wx = np.zeros(np.shape(wx))
        wx[mean_vals > mu_cut] = beta
        wx[mean_vals <= mu_cut] = 1
        scores = - (mean_vals.reshape(n_monte,) + np.max(mean_vals) / np.max(var_vals)  / np.max(wx) * wx * var_vals.reshape(n_monte,))
    elif acquisition == "cdf":
        mean_vals = mean_vals.reshape(n_monte,)
        wx = np.zeros(np.shape(mean_vals))
        wx[mean_vals > mu_cut] = beta
        wx[mean_vals <= mu_cut] = 1
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
    r_val = 0.025*max_domain_distance/2

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
    
    Theta_opt = theta_opt        
    
    # Find the optimal acquisition point
    Theta_opt = Theta_opt.reshape(n_keep,ndim)        
    
    # Calculate the U
    U_opt = Theta_to_U(Theta_opt,nsteps,1,2)
    U_opt = U_opt.reshape(np.shape(U_opt)[1],n_keep)

    # Pass to the Map
    Y_opt = map_def.pdf(Theta_opt).reshape(n_keep,1)/10**-1
    Y_opt = Y_opt.reshape(n_keep,1)

    # Append the value for the next step
    Theta = np.append(Theta, Theta_opt, axis = 0)
    Y = np.append(Y, Y_opt, axis = 0)
    training_data = model.training()

    # Now we need validation data.
    np.random.seed(100)
    test_pts = 10**5
    #Theta_test = inputs.draw_samples(test_pts, "lhs")
    Theta_test = inputs.draw_samples(100, "grd")
    Mean_Val_test, Var_Val_test = model.predict(Theta_test)
    
    if iter_num == 0:
        # Save the validation data
        obj = map_def.pdf(Theta_test)/10**-1
        mu_temp = np.zeros((ndim,))
        Y_mu = np.zeros((ndim,))
        for i in range(0,ndim):
            mu_temp[i] = 1
            Y_mu[i] = map_def.pdf(mu_temp.reshape(1,ndim))/10**-1
            
        sio.savemat(base_dir+'Wind_Dim'+str(ndim)+'_Validation_.mat', {'Theta_test':Theta_test, 'obj':obj, 'Y_mu':Y_mu})

    mu_temp = np.zeros((ndim,))
    Y_mup = np.zeros((ndim,))
    for i in range(0,ndim):
        mu_temp[i] = 1
        Y_mup[i], var_val_temp = model.predict(mu_temp.reshape(1,ndim))
    
    sio.savemat(save_path_data, {'Y_mup':Y_mup, 'Theta':Theta, 'cdf':cdf, 'U_opt':U_opt, 'wx':wx, 'scores':scores, 'y':y, 'x_int':x_int, 'Y':Y, 'Mean_Val':Mean_Val, 'Var_Val':Var_Val, 'Mean_Val_test':Mean_Val_test, 'Var_Val_test':Var_Val_test, 'n_keep':n_keep, 'n_init':n_init, 'N':N, 'seed':seed, 'Thetanew':Thetanew, 'Theta_test':Theta_test, 'training_data':training_data})


if __name__ == "__main__":
    main(int(sys.argv[1]),iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,upper_limit,n_keep)

