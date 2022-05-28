#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:12:55 2021

@author: ethanpickering
Performing a DNO search of the MMT Model of the Nonlinear Schrodinger Equation
"""

# DNOSearch Imports
import numpy as np
from dnosearch import (BlackBox, GaussianInputs, DeepONet)
from oscillator import Noise
import deepxde as dde

# NLS Import
from complex_noise import Noise_MMT

# Other Imports
import sys
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.io as sio
import h5py
import matplotlib.pyplot as plt
from scipy.spatial import distance


iter_num    = int(sys.argv[2])
rank        = int(sys.argv[3])
acq         = sys.argv[4]
lam         = float(sys.argv[5])
batch_size  = int(sys.argv[6])
n_init      = int(sys.argv[7])
epochs      = int(sys.argv[8])
b_layers    = int(sys.argv[9])
t_layers    = int(sys.argv[10])
neurons     = int(sys.argv[11])
n_guess     = int(sys.argv[12])
init_method = sys.argv[13]
model       = sys.argv[14]
N           = int(sys.argv[15])


tf = 1
noise = Noise_MMT([0, tf], rank)

# Create the Theta_u to U map for saving localling to be called by Matlab MMT files
def Save_U(Theta,nsteps,rank):
    y0 = np.zeros((1,nsteps),dtype=np.complex_)
    xr = Theta[0:rank]
    xi = Theta[rank:(2*rank)]
    x = xr + 1j*xi
    y0 = noise.get_sample(x)
    return y0

# Mapping Theta to U with only one input
def map_def(theta_rank):
    rank = int(theta_rank[1])
    theta = theta_rank[0].reshape(rank*2,)
    nsteps = 512
    y0 = Save_U(theta,nsteps,rank)
    return y0

def main(seed,iter_num,rank,acq,lam,batch_size,n_init,epochs,b_layers,t_layers,neurons,n_guess,init_method,model,N):

    
    if iter_num == 0:
        ndim = rank*2
        np.random.seed(seed)
        noise_var = 0
        nsteps = 512
        my_map = BlackBox(map_def, noise_var=noise_var)

        mean, cov = np.zeros(ndim), np.ones(ndim)
        domain = [ [-6, 6] ] * ndim
        inputs = GaussianInputs(domain, mean, cov)
        Theta = inputs.draw_samples(n_init, init_method)

        save_y0_file = './IC/Rank'+str(rank)+'_'+model+'_Seed'+str(seed)+'_Acq'+acq+'_Iter'+str(iter_num)+'_Lam'+str(lam)+'_BatchSize'+str(batch_size)+'_N'+str(N)+'_savedata_y0.mat'

        y0 = np.zeros((nsteps,n_init),dtype=np.complex_)
        for i in range(0,n_init):
            y0[:,i] = Save_U(Theta[i,:],nsteps,rank).reshape(nsteps,)
            # Save the y0 file to run with matlab
        sio.savemat(save_y0_file, {'y0':y0, 'Theta':Theta})        
    else:
        ndim = rank*2
        np.random.seed(seed)
        noise_var = 0
        my_map = BlackBox(map_def, noise_var=noise_var)

        mean, cov = np.zeros(ndim), np.ones(ndim)
        domain = [ [-6, 6] ] * ndim
        inputs = GaussianInputs(domain, mean, cov)
        

    
        # Need to determine U
        nsteps = 512 # Reset the number of steps for deeponet
        x_vals = np.linspace(0, 1, nsteps+1)
        x_vals = x_vals[0:-1]
            
        #Y = my_map.evaluate(Theta,1)    
        coarse = 4
        # Create the X to U map, which is actually theta to U
        def Theta_to_U(Theta,nsteps,coarse,rank):
           # We can also coarsen the steps 512 is likely extra fine for Deeponet
            Theta = np.atleast_2d(Theta)
            U = np.zeros((np.shape(Theta)[0],2*int(nsteps/coarse)),dtype=np.complex_)
    
            # Determine real and imaginary inds
            dim = int(np.shape(Theta)[1]/2)
            xr = Theta[:,0:(dim)]
            xi = Theta[:,dim:dim*2]
            x = xr + 1j*xi
            Us = np.transpose(noise.get_sample(x))
            coarser_inds = np.linspace(0,nsteps-1,int(nsteps/coarse)).astype(int)
    
            real_inds = np.linspace(0,nsteps/coarse*2-2,int(nsteps/coarse)).astype(int)
            imag_inds = np.linspace(1,nsteps/coarse*2-1,int(nsteps/coarse)).astype(int)
            
            U[:,real_inds] = np.real(Us[:,coarser_inds])
            U[:,imag_inds] = np.imag(Us[:,coarser_inds])
            return U
        
        def Theta_to_Z(Theta,rank):
            Z = np.ones((Theta.shape[0], 1))
            return Z          
        
        
        
        validation_file = './IC/Rank'+str(rank)+'_Xs1.mat'
        # Get previous iteration data
#        if batch_size > 1:
        save_path_data_prev     = './results/Rank'+str(rank)+'_'+model+'_'+acq+'_Seed'+str(seed)+'_N'+str(N)+'_Batch_'+str(batch_size)+'_Init_'+init_method+'_Iteration'+str(iter_num-1)+'.mat'
        save_path_data          = './results/Rank'+str(rank)+'_'+model+'_'+acq+'_Seed'+str(seed)+'_N'+str(N)+'_Batch_'+str(batch_size)+'_Init_'+init_method+'_Iteration'+str(iter_num)+'.mat'
#        else:
#            save_path_data_prev     = './results/Rank'+str(rank)+'_'+model+'_'+acq+'_Seed'+str(seed)+'_N'+str(N)+'_Iteration'+str(iter_num-1)+'.mat'
#            save_path_data          = './results/Rank'+str(rank)+'_'+model+'_'+acq+'_Seed'+str(seed)+'_N'+str(N)+'_Iteration'+str(iter_num)+'.mat'
        
        save_y0_file = './IC/Rank'+str(rank)+'_'+model+'_Seed'+str(seed)+'_Acq'+acq+'_Iter'+str(iter_num)+'_Lam'+str(lam)+'_BatchSize'+str(batch_size)+'_N'+str(N)+'_savedata_y0.mat'
        load_Y_file  = './IC/Rank'+str(rank)+'_'+model+'_Seed'+str(seed)+'_Acq'+acq+'_Iter'+str(iter_num-1)+'_Lam'+str(lam)+'_BatchSize'+str(batch_size)+'_N'+str(N)+'_savedata_Y.mat' 
        save_y0_file_prev = './IC/Rank'+str(rank)+'_'+model+'_Seed'+str(seed)+'_Acq'+acq+'_Iter'+str(iter_num-1)+'_Lam'+str(lam)+'_BatchSize'+str(batch_size)+'_N'+str(N)+'_savedata_y0.mat'


        if iter_num == 1:
            d = sio.loadmat(save_y0_file_prev)
            Theta = d['Theta']
            d = sio.loadmat(load_Y_file)
            Y = d['Y']
            print(np.shape(Y))
            
        else: 
            d = sio.loadmat(save_path_data_prev)
            Theta = d['Theta']
            Y = d['Y']
        
            # Load in the files
            d = sio.loadmat(save_y0_file_prev)
            theta_opt = d['theta_opt']
            d = sio.loadmat(load_Y_file)
            Y_opt = d['Y']
        
            Theta = np.vstack((Theta, theta_opt))
            Y = np.vstack((Y, Y_opt))
            print(np.shape(Theta))
            print(np.shape(Y))
    
        np.random.seed(np.size(Y))
    
        m       = int(nsteps/coarse*2) #604*2
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
        save_period = epochs
         
        # These functions are defined for normalizing, standardizing, or flatenining interal to DeepONet
        def DNO_Y_transform(x):
            x_transform = x
            return x_transform
    
        def DNO_Y_itransform(x_transform):
            x = x_transform
            return x
        
        # Train the model
        np.random.seed(np.size(Y)) # Randomize the seed based on Y size for consistency
        
        # Where to save the DeepONet models
        model_dir = './'
        model_str = 'Rank'+str(rank)+'_'+model+'_'+acq+'_Seed'+str(seed)+'_N'+str(N)+'_Batch_'+str(batch_size)+'_Init_'+init_method
        model = DeepONet(Theta, nsteps, Theta_to_U, Theta_to_Z, Y, net, lr, epochs, N, model_dir, seed, save_period, model_str, coarse, rank, DNO_Y_transform, DNO_Y_itransform)
        #training_data = model.training() # Get the training/loss values from the learning process
    
    
        # Create acquisition function and batch new points for acquisition        
        if ndim > 5:
            n_monte = 10**6
            print('Note that n_monte is set to 10^6 for memory limits on standard computers, but 10^7 was used for larger dimensions, results may be slightly impacted by this coarser representation')
            #n_end   = 10**4
        else:
            n_monte = 10**(ndim+1)
            #n_end = 10**(ndim)
        
        Theta_test = inputs.draw_samples(n_monte, "uni") # This is random uniform sampling of the space
        Theta_test = np.atleast_2d(Theta_test)
        
        Mean_Val, Var_Val = model.predict(Theta_test)
        
        if acq == 'US_LW':
            # Determine Bounds for evaluzting the metric
            x_max = np.max(Mean_Val)
            x_min = np.min(Mean_Val)
            x_int = np.linspace(x_min,x_max,10000) # Linearly space points
            x_int_standard = np.linspace(0,1,10000) # Static for pt-wise comparisons
        
            # Create the weights/exploitation values
            px = inputs.pdf(Theta_test)
            sc = scipy.stats.gaussian_kde(Mean_Val.reshape(n_monte,), weights=px)   # Fit a guassian kde using px input weights
            py = sc.evaluate(x_int) # Evaluate at x_int
            py[py<10**-16] = 10**-16 # Eliminate spuriously small values (smaller than numerical precision)
            py_standard = sc.evaluate(x_int_standard) # Evaluate for pt-wise comparisons
            py_interp = InterpolatedUnivariateSpline(x_int, py, k=1) # Create interpolation function
            
            # Conctruct the weights
            wx = px.reshape(n_monte,)/py_interp(Mean_Val).reshape(n_monte,)
            wx = wx.reshape(n_monte,1)
            
            # Compute the acquisition values
            ax = wx*Var_Val  # This is simply w(\theta) \sigma^2(\theta) - note that x and \theta are used interchangably
        elif acq == 'US':
            ax = Var_Val    
        else:
            print('Unkown Acquisition Function')
                
        scores = -ax
        sorted_idxs = np.argsort(scores,axis = 0)
        scores = scores.reshape(n_monte,)

        sorted_scores = scores[sorted_idxs[0:n_monte]]
        sorted_Theta = Theta_test[sorted_idxs[0:n_monte], :]
        
        Theta_guess = np.zeros((batch_size,ndim))
        score_guess = np.zeros((batch_size,))
        
        Theta_guess[0,:] = sorted_Theta[0,:]
        score_guess[0] = sorted_scores[0]

        # Now we need to remove the optimal from consideration, and remove values within a radius of influence
        max_domain_distance = np.sqrt((inputs.domain[1][1]-inputs.domain[0][0])**2*ndim)
        r_val = 0.025*max_domain_distance

        for i in range(1,batch_size):
            # Now remove the optimal value
            sorted_Theta = np.delete(sorted_Theta, 0, axis=0)
            sorted_scores = np.delete(sorted_scores, 0)
            distances = np.zeros((np.size(sorted_scores),))
            
            for j in range(0,min(1000,np.size(sorted_scores))):  # Now this is not perfect becuase it does not eliminate all points to be considered with the radius, but should be more robust
                distances[j] = distance.euclidean(Theta_guess[i-1,:], sorted_Theta[j,:])
            sorted_Theta = sorted_Theta[distances > r_val,:]
            sorted_scores = sorted_scores[distances > r_val]
            Theta_guess[i,:] = sorted_Theta[0,:]
            score_guess[i] = sorted_scores[0]
            
        scores_opt = score_guess
        theta_opt = Theta_guess
        theta_opt = np.atleast_2d(theta_opt)
        
        # Here we need to save the optimized value now such that it can be passed to the matlab function
        y0 = np.zeros((nsteps,batch_size),dtype=np.complex_)
        for i in range(0,batch_size):
            theta_rank = (theta_opt[i,:],np.shape(theta_opt)[1]/2)
            y0[:,i] = my_map.evaluate(theta_rank,1).reshape(nsteps,)            
  
        sio.savemat(save_y0_file, {'y0':y0, 'theta_opt':theta_opt})
        
        # Evaluate validation data, optimized data, and the results on the given data
        # Now we pull in the validation file
        d = sio.loadmat(validation_file)
        Theta_valid = d['Xs']        
        comp_batches = 1 # This is useful if the test data is very large and iterative computation saves memory
        n_samples = np.shape(Theta_valid)[0] #10**5
        
        comp_batch_size = n_samples / comp_batches # Note this is different than the ''batch_size'' for new acquisition points

        Mean_Val = np.zeros((n_samples,1))
        Var_Val = np.zeros((n_samples,1))
        US_LW_Val = np.zeros((n_samples,1))
        
        for batch in range(0,comp_batches):
            # Direct computation
            inds = np.linspace(batch*comp_batch_size, (comp_batch_size*(batch+1))-1, int(comp_batch_size)).astype(int)
            Mean_Val[inds,:], Var_Val[inds,:] = model.predict(Theta_valid[inds,:])
        
        Opt_Mean_Val, Opt_Var_Val = model.predict(theta_opt)
        Training_Mean_Val, Training_Var_Val = model.predict(Theta)

        if acq == 'US_LW':
            
            n_Y = np.size(Y)
            US_LW_Val = inputs.pdf(Theta_valid).reshape(n_samples,)/py_interp(Mean_Val).reshape(n_samples,) * Var_Val.reshape(n_samples,)
            Training_US_LW_Val = inputs.pdf(Theta).reshape(n_Y,)/py_interp(Training_Mean_Val).reshape(n_Y,) * Training_Var_Val.reshape(n_Y,)
            Opt_US_LW_Val = inputs.pdf(theta_opt).reshape(batch_size,)/py_interp(Opt_Mean_Val).reshape(batch_size,) * Opt_Var_Val.reshape(batch_size,)
            

        #training_data = model.training()
        
        # Calculate the approximated PDF for Validation points 
        x_int = np.linspace(0,1,1000) # Static for pt-wise comparisons
        px = inputs.pdf(Theta_valid)
        sc = scipy.stats.gaussian_kde(Mean_Val.reshape(n_samples,), weights=px)   # Fit a guassian kde using px input weights
        py = sc.evaluate(x_int) # Evaluate at x_int
        py[py<10**-16] = 10**-16 # Eliminate spuriously small values (smaller than numerical precision)

        print('Saving PDF only, for lite saving.')
        if rank == 1:
            sio.savemat(save_path_data, {'x_int':x_int, 'py':py,'Theta':Theta, 'Y':Y, 'Mean_Val':Mean_Val, 'Var_Val':Var_Val})
        else:
            sio.savemat(save_path_data, {'x_int':x_int, 'py':py,'Theta':Theta, 'Y':Y})
        # Below is a heavier save to track all of the variables
        #sio.savemat(save_path_data, {'Theta':Theta, 'Y':Y, 'Mean_Val':Mean_Val, 'Var_Val':Var_Val, 'US_LW_Val':US_LW_Val, 'n_init':n_init, 'N':N, 'seed':seed, 'Theta_valid':Theta_valid, 
        #                                  'Training_Mean_Val':Training_Mean_Val, 'Training_Var_Val':Training_Var_Val, 'Training_US_LW_Val':Training_US_LW_Val, 'Opt_Mean_Val':Opt_Mean_Val, 'Opt_Var_Val':Opt_Var_Val, 'Opt_US_LW_Val':Opt_US_LW_Val})

    return


if __name__ == "__main__":
    main(int(sys.argv[1]),iter_num,rank,acq,lam,batch_size,n_init,epochs,b_layers,t_layers,neurons,n_guess,init_method,model,N)
