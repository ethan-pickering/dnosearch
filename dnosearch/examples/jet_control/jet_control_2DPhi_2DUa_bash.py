#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 07:54:39 2022

@author: ethanpickering
"""

# DNOSearch Imports
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
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times"]})

# Variables
iter_num    = int(sys.argv[2]) # Iteration number
dim         = int(sys.argv[3]) # Dimension of the stochastic excitation (infection rate)
acq         = sys.argv[4] # Acquisition type - currently only Likelihood-weighted uncertatiny sampling
n_init      = int(sys.argv[5]) # Initial data points
epochs      = int(sys.argv[6]) # Number of training epochs
b_layers    = int(sys.argv[7]) # Branch Layers
t_layers    = int(sys.argv[8]) # Trunk Layers
neurons     = int(sys.argv[9]) # Number of neurons per layer
init_method = sys.argv[10] # How initial data are pulled
N           = int(sys.argv[11])  # Number of DNO ensembles

print('Print plots is set to True')
print_plots  =True


# The map we are defining here is 
# Input   = sum_i^N sin(x+phi_0+theta_i) 
# Output  = sum_i^N sin(x(end)+phi_0+theta_i) = sum_i^N sin(2*pi+phi_0+theta_i)

# This I-O relationshsip means we are interested in an identify mapping
# of the last point in the input signal

def map_def(U):    
    nsteps = 50
    y=np.zeros(np.shape(U)[0])
    for i in range(0,1):
        y = y + (U[:,i*nsteps-1])**10
    return y


def main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,print_plots):
    # Seed above is for initial condition consistency - NOTE due to gradient descent of the DNO, the seed will not provide perfectly similar results, but will be analogous
    
    ndim = dim
    udim = ndim # The dimensionality of the U components of Theta
    mean, cov = np.zeros(ndim), np.ones(ndim)
    domain = [[0, 2*np.pi], [-0.5, 0.5]]
    
    inputs = UniformInputs(domain)
    np.random.seed(seed)
    
    if iter_num == 0:
        Theta = inputs.draw_samples(n_init, init_method)
    
    # Need to determine U
    nsteps = 50 # discretely define function (sin val)
    
    # DeepONet only needs a coarse version of the signal    
    coarse = 1 # Lets keep it the same for now
    
    ##!!! NOTE THE BOTH OF THESE NEED TO BE CHANGED INSIDE: Theta_to_U as well.
    phi_0 = -np.pi/2 # original
    wavenumber = 1 # cool to look at higher wave numbers, e.g., 4
    
    def Theta_to_U(Theta,nsteps,coarse,rank):
        U = np.zeros((np.shape(Theta)[0],nsteps))
        phi_0 = -np.pi/2 # original
        wavenumber = 1 # 1
        x = np.linspace(0,2*np.pi,nsteps) + phi_0
        print(Theta)
        
        for j in range(0,np.shape(Theta)[0]):
            for i in range(0,1):
                U[j,0+(i*nsteps):((i+1)*nsteps)] =  Theta[j,i+1] + np.sin(wavenumber*(x+phi_0+Theta[j,i]))            
        return U
    
    
    def Theta_to_Z(Theta,rank):
        if Theta.shape[1] == rank:
            Z = np.ones((Theta.shape[0], 1))
        else:
            Z = Theta[:,(2*rank):Theta.shape[1]]
        return Z
    
    if iter_num == 0:
        # Determine the training data
        Us = Theta_to_U(Theta,nsteps,1,ndim)
        Y = map_def(Us).reshape(n_init,1)
        Y = Y.reshape(n_init,1)
    
    # Data Paths
    save_path_data = './data/Jet_2DPhi_2DUa_'+str(acq)+'_Seed_'+str(seed)+'_N'+str(N)+'_iter_'+str(iter_num)+'.mat'
    load_path_data = './data/Jet_2DPhi_2DUa_'+str(acq)+'_Seed_'+str(seed)+'_N'+str(N)+'_iter_'+str(iter_num-1)+'.mat'

    if iter_num > 0: # Will load in data
        d = sio.loadmat(load_path_data)
        Theta = d['Theta']
        Y = d['Y']    

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
        x_transform = x/60
        return x_transform

    def DNO_Y_itransform(x_transform):
        x = x_transform*60
        return x
    
    
    # Train the model
    np.random.seed(np.size(Y)) # Randomize the seed based on Y size for consistency
    
    # Where to save the DeepONet models
    model_dir = './'
    model_str = ''
    model = DeepONet(Theta, nsteps, Theta_to_U, Theta_to_Z, Y, net, lr, epochs, N, model_dir, seed, save_period, model_str, coarse, udim, DNO_Y_transform, DNO_Y_itransform)

    # Pull a fine set of test_pts in the domain
    test_pts = 75
    Theta_test = inputs.draw_samples(test_pts, "grd")
    # Predict
    Mean_Val, Var_Val = model.predict(Theta_test)
    
    # Determine Bounds for evaluzting the metric
    x_max = np.max(Mean_Val)
    x_min = np.min(Mean_Val)
    x_int = np.linspace(x_min,x_max,100) # Linearly space points
    x_int_standard = np.linspace(0,ndim,100) # Static for pt-wise comparisons

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
    
    if acq == 'LCB_LW':
        kappa = 1
        ax = Mean_Val + kappa * wx * (Var_Val)**(1/2) * np.max(Mean_Val) / np.max(wx*(Var_Val)**(1/2)) 
    elif acq == 'US_LW':
        ax = wx*Var_Val  # This is simply w(\theta) \sigma^2(\theta) - note that x and \theta are used interchangably
    
    # Find the optimal acquisition point
    Theta_opt = Theta_test[np.argmax(ax),:]
    Theta_opt = Theta_opt.reshape(1,udim)        
    
    # Calculate the associated U
    U_opt = Theta_to_U(Theta_opt,nsteps,1,udim)
    U_opt = U_opt.reshape(1,np.size(U_opt))

    # Pass to the Map
    Y_opt = map_def(U_opt).reshape(1,1)
    Y_opt = Y_opt.reshape(1,1)

    # Append the value for the next step
    Theta = np.append(Theta, Theta_opt, axis = 0)
    Y = np.append(Y, Y_opt, axis = 0)
    #pys[iter_num,:] = py_standard
    sio.savemat('./data/Jet_2DPhi_2DUa_'+str(acq)+'_Seed_'+str(seed)+'_N'+str(N)+'_iter_'+str(iter_num)+'.mat', {'py_standard':py_standard, 'x_int_standard':x_int_standard, 'Theta':Theta, 'U_opt':U_opt, 'wx':wx, 'ax':ax, 'py':py, 'x_int':x_int, 'Y':Y, 'Mean_Val':Mean_Val, 'Var_Val':Var_Val, 'n_init':n_init, 'N':N, 'seed':seed, 'Theta_test':Theta_test})

    
    if print_plots:
 
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.1)
        (ax1, ax2), (ax3, ax4) = gs.subplots()#(sharex='col', sharey='row')
        fig.suptitle('2D U = U1 Phi x a, Jet Control Search, Iteration '+str(iter_num))
        ax1.pcolor(Theta_test[:,0].reshape(test_pts, test_pts), Theta_test[:,1].reshape(test_pts, test_pts), Mean_Val.reshape(test_pts, test_pts))
        #ax1.set_aspect('equal')
        #ax1.colorbar()
        ax1.annotate('Mean Solution',
        xy=(3, 5), xycoords='data',
        xytext=(0.7, 0.95), textcoords='axes fraction',
        horizontalalignment='right', verticalalignment='top',color='white')
        #ax1.set_ylabel('$\theta_2$') 
        
        ax2.pcolor(Theta_test[:,0].reshape(test_pts, test_pts), Theta_test[:,1].reshape(test_pts, test_pts), Var_Val.reshape(test_pts, test_pts))
        ax2.plot(Theta[0:np.size(Y)-1,0], Theta[0:np.size(Y)-1,1], 'wo')
        #ax2.set_aspect('equal')
        #ax2.colorbar()
        ax2.annotate('Variance',
        xy=(3, 5), xycoords='data',
        xytext=(0.7, 0.95), textcoords='axes fraction',
        horizontalalignment='right', verticalalignment='top',color='white') 
        
        ax3.pcolor(Theta_test[:,0].reshape(test_pts, test_pts), Theta_test[:,1].reshape(test_pts, test_pts), wx.reshape(test_pts, test_pts))
        #ax3.set_aspect('equal')
        #ax3.colorbar()
        ax3.annotate('Danger Scores',
        xy=(3, 5), xycoords='data',
        xytext=(0.7, 0.95), textcoords='axes fraction',
        horizontalalignment='right', verticalalignment='top',color='white') 
        #ax3.set_ylabel('$\theta_2$') 
        #ax3.set_xlabel('$\theta_1$') 

        ax4.pcolor(Theta_test[:,0].reshape(test_pts, test_pts), Theta_test[:,1].reshape(test_pts, test_pts), ax.reshape(test_pts, test_pts))
        ax4.plot(Theta[-1,0], Theta[-1,1], 'ro')
        #ax4.colorbar()
        #ax4.set_aspect('equal')
        ax4.annotate('Acquisition',
        xy=(3, 5), xycoords='data',
        xytext=(0.7, 0.95), textcoords='axes fraction',
        horizontalalignment='right', verticalalignment='top',color='white') 
        #ax4.set_xlabel('$\theta_1$')
        #ax4.set_xlim([-6,6])
        #ax4.set_ylim([-6,6])
        plt.draw()
        
        plt.savefig('./plots/Jet_2DPhi_2DUa_'+str(acq)+'_Seed_'+str(seed)+'_N'+str(N)+'_iter_'+str(iter_num)+'.jpg',  dpi=150)

    sio.savemat(save_path_data, {'py_standard':py_standard,'x_int_standard':x_int_standard, 'Theta':Theta, 'U_opt':U_opt, 
                                 'wx':wx, 'ax':ax, 'py':py, 'x_int':x_int, 'Y':Y, 'Mean_Val':Mean_Val, 
                                 'Var_Val':Var_Val, 'n_init':n_init, 'N':N, 'seed':seed, 'Theta_test':Theta_test})


if __name__ == "__main__":
    main(int(sys.argv[1]),iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,print_plots)
