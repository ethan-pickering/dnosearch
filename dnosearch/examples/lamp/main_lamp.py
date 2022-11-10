#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:06:32 2022
    Active learning via NNs of a 2D stochastic ship dynamics problem
@author: stephenguth
"""

# DNOSearch Imports
import numpy as np
from dnosearch import (BlackBox, GaussianInputs, DeepONet)
#from oscillator import Noise

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
import sklearn as sk
import os
import lamp_helper_functions as hf


# Variables

if len(sys.argv) <= 1:
    #
    # Call direktly with Python
    #
    seed                = 3 # Seed for initial condition consistency - NOTE due to gradient descent of the DNO, the seed will not provide perfectly similar results, but will be analogous
    initial_iter_num    = 0 # Iteration number
    dim                 = 10 # Dimension of the KL subspace
    acq                 = 'KUS_LW' # Acquisition type - currently only Likelihood-weighted uncertatiny sampling
    n_init              = 3 # Initial data points
    epochs              = 1000  # Number of training epochs
    b_layers            = 8 # Branch Layers
    t_layers            = 1 # Trunk Layers
    neurons             = 300 # Number of neurons per layer
    init_method         = 'pdf'# How initial data are pulled
    N                   = 2 # Number of DNO ensembles
    iters_max           = 2  # Iterations to perform
    run_name            = 'temp'
    activation_func     = 'relu'
    sampling_sigma      = 0
else :
    #
    # Call via bash
    #
    seed                = int(sys.argv[1])
    initial_iter_num    = int(sys.argv[2]) # Iteration number
    dim                 = int(sys.argv[3]) # Dimension of the KL subspace
    acq                 = sys.argv[4] # Acquisition type - currently only Likelihood-weighted uncertatiny sampling
    n_init              = int(sys.argv[5]) # Initial data points
    epochs              = int(sys.argv[6]) # Number of training epochs
    b_layers            = int(sys.argv[7]) # Branch Layers
    t_layers            = int(sys.argv[8]) # Trunk Layers
    neurons             = int(sys.argv[9]) # Number of neurons per layer
    init_method         = sys.argv[10] # How initial data are pulled
    N                   = int(sys.argv[11])  # Number of DNO ensembles
    iters_max           = 1 # we should check to be sure this is good?
    run_name            = sys.argv[12]
    activation_func     = sys.argv[13]
    sampling_sigma      = float(sys.argv[14])


print_plots = True




def main(seed,initial_iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,
         N,iters_max,print_plots,run_name='test_run',activation_func='relu', sampling_sigma=0):
    
    print('\n\n\n\n\n')
    print('Starting LAMP AS procedure, using DeepONet and Pickering variance.')
    print('Run name:  {}   Round {}'.format(run_name, initial_iter_num))
    print('Dim={}, acq={}'.format(dim, acq))
    print('\n\n\n\n\n')
    
    #
    # File IO
    #
    
    data_path = './LAMP_10D_Data/'
    gpr_pdf_path = './LAMP_10D_Data/'
    
    output_path = './{}/'.format(run_name)
    err_save_path = '{}errors/'.format(output_path)
    model_dir =  '{}'.format(output_path)
    intermediate_data_dir =  '{}data/'.format(output_path)
    as_dir = '{}as_quantities/'.format(output_path)
    fig_save_path = '{}plots/'.format(output_path)
    
    model_suffix = '-{}-40'.format(dim)
    
    hf.make_dirs(output_path, err_save_path, model_dir, as_dir, fig_save_path, intermediate_data_dir)

    #
    # Data IO
    #
    
    wTT, wDD, wVV = hf.load_wave_data(data_path, model_suffix)
    vTTlhs, vZZlhs, vAAlhs = hf.load_vbm_lhs_data(data_path, model_suffix, trim=False)
    #vTTmc, vZZmc, vAAmc = hf.load_vbm_mc_data(data_path, model_suffix)
    vTTmc, vZZmc, vAAmc = hf.load_vbm_lhs_data(data_path, model_suffix,  trim=False)
    qq_xx, qq_pp, mm_xx, mm_pp = hf.load_gpr_precomputed(gpr_pdf_path, dim)
    
    #sigma_n_filename = '{}{}-40-sigma-n-list.txt'.format(gpr_pdf_path, dim)
    #sigma_n_list = np.loadtxt(sigma_n_filename)
    
    n_lhs_data = vAAlhs.shape[0]
    n_wave_t = wTT.shape[0]
    #n_vbm_t = vZZlhs.shape[1]
    
    # Data Paths
    save_suffix = 'Seed_'+str(seed)+'_N'+str(N)+'_iter_'+str(initial_iter_num)+'.mat'
    load_suffix = 'Seed_'+str(seed)+'_N'+str(N)+'_iter_'+str(initial_iter_num-1)+'.mat'
    
    save_path_data = '{}LAMP_{}'.format(intermediate_data_dir, save_suffix)
    load_path_data = '{}LAMP_{}'.format(intermediate_data_dir, load_suffix)
    
    save_error_data = '{}LAMP_Errors_{}'.format(err_save_path, save_suffix)
    load_error_data = '{}LAMP_Errors_{}'.format(err_save_path, load_suffix)
    
    #
    # Number of output modes to retain, and how many to use in the RR rondel
    # Roughly, n_q_modes ought be between 2 and 3 times dim, and rr_constant
    # ought be between 1 and 1.5 times dim
    #
    # If this gets too big, we run into memory problem w.r.t. how many NN we
    # need to store in memory (per round) simultaneously
    #
    
    n_q_modes = np.minimum(2*dim+2, 10)
    rr_constant = np.minimum(1*dim+2, 6)
    
    #
    # Dim stuff
    #
    
    ndim = dim
    udim = dim # The dimensionality of the U components of Theta
    
    vAAlhs = vAAlhs[:, 0:dim]
    
    # DeepONet only needs a coarse version of the signal    
    coarse = 4
    
    u_decimation_factor = 5
    y_decimation_factor = 2
    
    #
    as_target_quantity = 'mode-coefficient'
    #as_target_quantity = 'vbm-interval-extrema'
    
    
    np.random.seed(seed)
    
    #
    # PCA transform of VBM!
    #
    
    QQ, w_vbm, v_vbm, vv_var = hf.pca_transform_z_2_q(vZZlhs, vZZmc, sklearn_pca_algo = False, n_q_modes=n_q_modes)
    
    #
    # Set up the input space
    #
    
    mean, cov = np.zeros(ndim), np.ones(ndim)
    zstar = 4.5
    domain = [ [-zstar, zstar] ] * ndim
    inputs = GaussianInputs(domain, mean, cov)
        
    #
    # Convert from wave episode coefficient form to wave episode time series
    #
    def Theta_to_U(alpha, nsteps, coarse, udim): 
        
        
        if alpha.shape[0] == 0 :
            U1 = np.zeros([n_wave_t, 1])
            for k in range(0, udim):
                U1 = U1 + alpha[k]*np.sqrt(wDD[k])*wVV[:, k]
        else :
            U1 = np.zeros([alpha.shape[0], n_wave_t])
            for k in range(0, udim):
                for j in range(0, alpha.shape[0]):
                    U1[j, :] = U1[j, :] + alpha[j, k]*np.sqrt(wDD[k])*wVV[:, k]
                    
        coarser_inds = np.linspace(0,nsteps-1,int(nsteps/coarse)).astype(int)
        U = U1[:,coarser_inds]
        return U/u_decimation_factor
    
    #
    # Dummy function, map everything to 1
    #
    
    def Theta_to_Z(Theta,udim):
        if Theta.shape[1] == udim:
            Z = np.ones((Theta.shape[0], 1))
        else:
            Z = Theta[:,(udim+1):Theta.shape[1]]
        return Z  
    
    #
    # Lambda the DNO transforms, to include the decimation faktor parameter
    #
    
    cur_DNO_Y_transform = lambda x : hf.DNO_Y_transform(x, decimation_factor=y_decimation_factor)
    cur_DNO_Y_itransform = lambda x : hf.DNO_Y_itransform(x, decimation_factor=y_decimation_factor)
    
    #
    # Initialize data matrices
    #
    
    if initial_iter_num == 0 :
        #
        # Draw the initial data points
        # 
        ii_pick = np.random.randint(0, high=n_lhs_data-1, size=n_init)
        Theta = vAAlhs[ii_pick, :]
        Y = QQ[ii_pick, :]
        
        # Keeping track of the error metric
        pys = np.zeros((iters_max, qq_xx.shape[0]))
        log10_errors_mode = np.zeros((iters_max, n_q_modes))
        log10_errors_vbm = np.zeros((iters_max, 1))
        log10_errors_vbm2 = np.zeros((iters_max, 1))
        
    else :
        #
        # Load current AS state from saved data
        # (Only get here if you launch from Bash and this isn't the first iterate)
        #
        d = sio.loadmat(load_path_data)
        Theta = d['Theta']
        Y = d['Y']
        
        #
        # Load previosu error metrics!
        #
        
        previous_pys = d['pys']
        pys = np.zeros((previous_pys.shape[0]+1, qq_xx.shape[0]))
        pys[0:-1, :] = previous_pys
        
        d = sio.loadmat(load_error_data)
        previous_mode_err = d['log10_errors_mode']
        log10_errors_mode = np.zeros((previous_mode_err.shape[0]+1, n_q_modes))
        log10_errors_mode[0:-1, :] = previous_mode_err[:, :]
        
        previous_vbm_err = d['log10_errors_vbm']
        log10_errors_vbm = np.zeros((previous_vbm_err.shape[0]+1, 1))
        log10_errors_vbm[0:-1, 0] = previous_vbm_err[:, 0]
        
        previous_vbm_err2 = d['log10_errors_vbm2']
        log10_errors_vbm2 = np.zeros((previous_vbm_err2.shape[0]+1, 1))
        log10_errors_vbm2[0:-1, 0] = previous_vbm_err2[:, 0]
    
    #
    # Start the NN things
    #
    # Only parameter here that I (SJ) have touched direktly is the activation function,
    # for which I found that sometimes 'tanh' works as well or better than relu
    #
    
    m       = int(n_wave_t/coarse)
    lr      = 0.001
    dim_x   = 1
    #activation = "relu"
    #activation  = "tanh"
    activation  = activation_func        # Read this in from runner
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
    print(n_wave_t)
    
    ##########################################
    # Loop through iterations
    ##########################################
    
    for iter_num in range(initial_iter_num,initial_iter_num+iters_max):
        model_list = np.empty([n_q_modes,],dtype=object)
        
        #
        # Train a NN for each mode separately, using the particular model
        # coefficients as the NN output
        #
        
        print('Starting NN training.')
        
        for cur_mode in range(0, n_q_modes) :
        
            #
            # Setup the data using current round robin index
            #

            cur_Y = Y[:, cur_mode]
            cur_Y = np.reshape(cur_Y, [cur_Y.shape[0], 1])
            
            # Train the model
            np.random.seed(np.size(Y))
            
            model_str = '_r{}'.format(cur_mode)
            model = DeepONet(Theta, n_wave_t, Theta_to_U, Theta_to_Z, cur_Y, net, 
                             lr, epochs, N, model_dir, seed, save_period, model_str, 
                             coarse, udim, cur_DNO_Y_transform, cur_DNO_Y_itransform)
            
            model_list[cur_mode] = model

        #
        # Create the set of test points
        #
        # OLD: Pull a fine set of test_pts in the domain
        # NEW: Check the parameters of the precomputed LHS points
        #
        
        Theta_test = vAAlhs
        
        
        
        #
        # Evaluate the model for Active Sampling!
        #
        # Save values outside of the loop that we might need later
        # (for diagnostic plots, mostly)
        #
        
        print('Starting acquisition function evaluations for Active Sampling.')
        
        if as_target_quantity == 'mode-coefficient' :
            ax_list = np.empty([n_q_modes,],dtype=object)
            cur_q_index = np.mod(iter_num, rr_constant)
            cur_extrema_class = 'none'
            
            for cur_mode in range(0, n_q_modes) :
            
                Mean_Val, Var_Val, wx, ax, py, py_standard, x_int, x_int_standard = hf.acq_calculation_rom(model_list, 
                                                            Theta_test, inputs, qq_xx=qq_xx, acq_rule=acq,
                                                            cur_mode=cur_mode, as_target_quantity='mode-coefficient', n_q_modes=n_q_modes,
                                                            v_vbm=v_vbm, w_vbm=w_vbm, vv_var=vv_var)
                ax_list[cur_mode] = ax
                
            ax_cur = ax_list[cur_q_index]
            
            
        elif as_target_quantity == 'vbm-interval-extrema' :
            #
            # Added b/c Ethan and I were discussing alternative statistics with
            # different likelihood ratios.  This was not the way.
            #
            if np.mod(iter_num, 2) == 0:
                cur_extrema_class = 'vbm-interval-max'
            else:
                cur_extrema_class = 'vbm-interval-min'
            
            Mean_Val, Var_Val, wx, ax, py, py_standard, x_int, x_int_standard = hf.acq_calculation_rom(model_list, 
                                                        Theta_test, inputs, qq_xx=mm_xx, acq_rule=acq,
                                                        cur_mode=0, as_target_quantity=cur_extrema_class, n_q_modes=n_q_modes,
                                                        v_vbm=v_vbm, w_vbm=w_vbm, vv_var=vv_var)
            
            ax_cur = ax
            
            
        #
        # Optimize!
        #
         
        # Find the optimal acquisition point
        ii_opt = np.argmax(ax_cur)
        Theta_opt = Theta_test[ii_opt,:]
        Theta_opt = Theta_opt.reshape(1,ndim)        
        
        # Calculate the associated U
        U_opt = Theta_to_U(Theta_opt,n_wave_t,1,udim)
        U_opt = U_opt.reshape(np.size(U_opt),1)
        
        #
        # Old:     pass to map_def
        # New:     use precomputed values
        # Newest:  use precomputed values inside of map_def
        #

        Y_opt = hf.map_def(0, ii_opt, QQ, sample_strat='discrete-noiseless', sigma_n=0) #sigma_n=sigma_n_list[cur_q_index])
    
        # Append the value for the next step
        Theta = np.append(Theta, Theta_opt, axis = 0)
        Y = np.append(Y, np.reshape(Y_opt, [1, n_q_modes]), axis = 0)
        pys[iter_num,:] = py_standard
        
        
        #
        # Error calculations and plotting things
        #
        
        for cur_mode in range(0, np.minimum(n_q_modes, 4)) :

            cur_py_standard_truth = qq_pp[cur_mode, :]
            
            plot_pts = 75
            #Theta_plot = inputs.draw_samples(plot_pts, "grd")
            
            Theta_plot = np.zeros((plot_pts**2, ndim))
            for k in range(0, plot_pts):
                for j in range(0, plot_pts):
                    Theta_plot[k*plot_pts+j, 0] = k/plot_pts*zstar*2 - zstar
                    Theta_plot[k*plot_pts+j, 1] = j/plot_pts*zstar*2 - zstar
            
            Mean_Val, Var_Val, wx, ax, py, cur_py_standard, x_int, x_int_standard = hf.acq_calculation_rom(model_list, 
                                                        Theta_plot, inputs, qq_xx, cur_mode=cur_mode,
                                                        as_target_quantity=as_target_quantity, n_q_modes=n_q_modes,
                                                        v_vbm=v_vbm, w_vbm=w_vbm, vv_var=vv_var)
            
            x_int_standard = qq_xx
            
            log10_error = hf.calc_log_error(cur_py_standard, cur_py_standard_truth,
                                        dx=(x_int_standard[2] -x_int_standard[1]), 
                                        ii = range(31,256-32), trunc_thresh= 1*10**-4)
            log10_errors_mode[iter_num, cur_mode] = log10_error
            print('The log10 of the log-pdf error for mode {} is: '.format(cur_mode)+str(np.log10(log10_error)))
            
            if print_plots:   
                hf.plot_as_mode_diagnostics(Theta_plot, Theta, Mean_Val, Var_Val, Y, wx, ax, plot_pts, 
                                        x_int_standard, cur_py_standard_truth, cur_py_standard, 
                                        log10_errors_mode, fig_save_path =fig_save_path,
                                        iter_num=iter_num, cur_mode=cur_mode)
                
        #
        # MC sample to reconstruct the full VBM pdf
        # Using scipy.stats.gaussian_kde() in honor of Ethan's legacy code, but
        # it's probably no worse that deliberate histograming
        #
        
        n_mc_samples = 1*10**4
        Theta_mc = inputs.draw_samples(n_mc_samples, "pdf")
        S_list = sampling_sigma*np.ones((n_q_modes, 1))
        zz_scale = hf.sample_VBM_from_NN(Theta_mc, model_list, v_vbm, w_vbm, vv_var, 
                                         n_q_modes=n_q_modes, noise_model='gaussian', sigma_n_list=S_list)
        
        zz_ravel = zz_scale.ravel()
        sc = scipy.stats.gaussian_kde(zz_ravel)   # Fit a guassian kde using px input weights
        pz = sc.evaluate(mm_xx) # Evaluate at x_int
        pz[pz<10**-16] = 10**-16 # Eliminate spuriously small values (smaller than numerical precision)
        
        thresh1 = 1*10**-13
        thresh2 = 1*10**-12
        
        log10_error_vbm = hf.calc_log_error(pz, mm_pp,
                                    dx=(x_int_standard[2] -x_int_standard[1]), 
                                    ii = None, trunc_thresh= thresh1)
        log10_errors_vbm[iter_num, 0] = log10_error_vbm
        
        log10_error_vbm2 = hf.calc_log_error(pz, mm_pp,
                                    dx=(x_int_standard[2] -x_int_standard[1]), 
                                    ii = None, trunc_thresh= thresh2)
        log10_errors_vbm2[iter_num, 0] = log10_error_vbm2
        
        if print_plots:      
            hf.plot_as_vbm_diagnostics(mm_xx, mm_pp, pz, log10_errors_vbm, fig_save_path = fig_save_path, 
                                        cur_mode=cur_mode, iter_num=iter_num)
        
         
        #
        # Save the AS state stuff in case bash needs to reload Python
        #
        
        sio.savemat(save_path_data, {'pys':pys, 'x_int_standard':x_int_standard, 
                                     'Theta':Theta, 'U_opt':U_opt, 'wx':wx, 'ax':ax, 
                                     'py':py, 'x_int':x_int, 'Y':Y, 'Mean_Val':Mean_Val, 
                                     'Var_Val':Var_Val, 'n_init':n_init, 'N':N, 'seed':seed, 
                                     'Theta_test':Theta_test})
        
        #
        # Save some error stuff when we're all done
        #  
        
        sio.savemat(save_error_data, {'log10_errors_mode':log10_errors_mode, 
                                      'log10_errors_vbm':log10_errors_vbm, 
                                      'log10_errors_vbm2':log10_errors_vbm2})
        
        
        
    return

# Call the function
main(seed,initial_iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,
     init_method,N,iters_max,print_plots,run_name=run_name,
     activation_func=activation_func,sampling_sigma=sampling_sigma)


