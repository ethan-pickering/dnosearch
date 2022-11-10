#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:47:11 2022

@author: stevejon
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

# SJ imports
import sklearn as sk
import os



#
# Would be our link to the Matlab wrapper for LAMP, if we implement it
#

def map_def(alpha, ii, QQ, sample_strat='discrete-noisless', sigma_n=0):
    if sample_strat == 'discrete-noiseless' :
        return QQ[ii, :]
    
    if sample_strat == 'discrete-noisy' :
        return QQ[ii, :] + np.randn()*sigma_n
    
    print('Uh-oh, sample strategy {} not recognized!'.format(sample_strat))
    return 1


#####################
# IO Helper Functions
#####################

def make_dirs(output_path, err_save_path, model_dir, as_dir, fig_save_path, intermediate_data_dir):
    #
    # Why oh why can't mkdir() fail silently if a direktory already exists?
    #
    
    try:
        os.mkdir(output_path)
    except OSError as error:
        print(error)
    try:
        os.mkdir(err_save_path)
    except OSError as error:
        print(error)
    try:
        os.mkdir(model_dir)
    except OSError as error:
        print(error)    
    try:
        os.mkdir('{}/model'.format(model_dir)) # 'cause DeepONet has some built in pathing
    except OSError as error:
        print(error)
    try:
        os.mkdir(as_dir)
    except OSError as error:
        print(error)
    try:
        os.mkdir(fig_save_path)
    except OSError as error:
        print(error)
    try:
        os.mkdir(intermediate_data_dir)
    except OSError as error:
        print(error)
        
        
def load_wave_data(data_path, model_suffix):
    #
    # Load some precomputed LAMP data
    #
       
    wave_TT_filename = '{}TT{}.txt'.format(data_path, model_suffix)
    wave_DD_filename = '{}DD{}.txt'.format(data_path, model_suffix)
    wave_VV_filename = '{}VV{}.txt'.format(data_path, model_suffix)
       

    wTT = np.loadtxt(wave_TT_filename)
    wDD = np.loadtxt(wave_DD_filename)
    wVV = np.loadtxt(wave_VV_filename)
       
    return wTT, wDD, wVV

def load_vbm_lhs_data(data_path, model_suffix, trim=True):
    vbm_TT_lhs_filename = '{}kl-2d{}-tt.txt'.format(data_path, model_suffix)
    vbm_zz_lhs_filename = '{}kl-2d{}-vbmg.txt'.format(data_path, model_suffix)
    vbm_aa_lhs_filename = '{}kl-2d{}-design.txt'.format(data_path, model_suffix)
    
    vTTlhs = np.loadtxt(vbm_TT_lhs_filename)
    vZZlhs = np.loadtxt(vbm_zz_lhs_filename)
    vAAlhs = np.loadtxt(vbm_aa_lhs_filename)
    
    if trim :
        vZZlhs = vZZlhs[0:625, :]   # minor accounting error during LAMP problem design
        vAAlhs = vAAlhs[0:625, :]        
    
    return vTTlhs, vZZlhs, vAAlhs
        
def load_vbm_mc_data(data_path, model_suffix):
    vbm_TT_mc_filename = '{}kl-2d{}-test-tt.txt'.format(data_path, model_suffix)
    vbm_zz_mc_filename = '{}kl-2d{}-test-vbmg.txt'.format(data_path, model_suffix)
    vbm_aa_mc_filename = '{}kl-2d{}-test-design.txt'.format(data_path, model_suffix)
    
    vTTmc = np.loadtxt(vbm_TT_mc_filename)
    vZZmc = np.loadtxt(vbm_zz_mc_filename)
    vAAmc = np.loadtxt(vbm_aa_mc_filename)
    
    return vTTmc, vZZmc, vAAmc

def load_gpr_precomputed(gpr_pdf_path, ndim):
    if (ndim > 6) :
        qdim = 6
    else :
        qdim = ndim
    
    qq_xx_filename = '{}{}-40-modes-bins.txt'.format(gpr_pdf_path, qdim)
    qq_pp_filename = '{}{}-40-modes-hist.txt'.format(gpr_pdf_path, qdim)
    
    if ndim >= 6 :
        # b/c the GPR VBM pdf is real bad for 6D, skip straight to the true MC
        # data, equivalent to \inf D
        mm_xx_filename = '{}mc-vbm-bins.txt'.format(gpr_pdf_path)
        mm_pp_filename = '{}mc-vbm-hist.txt'.format(gpr_pdf_path)
    else :
        mm_xx_filename = '{}{}-40-vbm-bins.txt'.format(gpr_pdf_path, ndim)
        mm_pp_filename = '{}{}-40-vbm-hist.txt'.format(gpr_pdf_path, ndim)
    
    qq_xx = np.loadtxt(qq_xx_filename)
    qq_xx = 1/2*(qq_xx[0:-1] + qq_xx[1::])  # b/c big dumb I saved it wrong
    qq_pp = np.loadtxt(qq_pp_filename)
    mm_xx = np.loadtxt(mm_xx_filename)
    mm_pp = np.loadtxt(mm_pp_filename)
    
    return qq_xx, qq_pp, mm_xx, mm_pp
    
############
# PCA Stuff
############

def project_onto_vector(x, v):
     a = np.dot(x, v) / np.dot(v, v)
     return a
 
    #
    # PCA transform of VBM!
    #
    # sklearn doesn't automatically normalize the PCA components, so we do that
    # by hand
    #
    # Actually, sklearn doesn't do PCA the same way I've been doing it, so I should
    # use my other method.  Probably smart PCA has various regularization stuff
    # for statisticians that I don't want
    #

def pca_transform_z_2_q(vZZlhs, vZZmc, sklearn_pca_algo = False, n_q_modes=6) :
    
    n_lhs_data = vZZlhs.shape[0]
    
    if sklearn_pca_algo :
        q_pca = sk.decomposition.PCA(n_components = n_q_modes)
        q_pca.fit(vZZmc)
        
        #print(q_pca.explained_variance_ratio_)
        #print(q_pca.singular_values_)
        
        q_lambda_mat = q_pca.get_covariance()
        q_lambda_list = np.zeros([n_q_modes,])
        for k in range(0, n_q_modes):
            q_lambda_list[k] = q_lambda_mat[k ,k]
        
        QQ_raw = q_pca.transform(vZZlhs)
        QQ = np.zeros(QQ_raw.shape)
        
        for k in range(0, n_q_modes):
            QQ[:, k] = QQ_raw[:, k] / np.sqrt( q_lambda_list[k])
        
    else : 
        vv_var = np.var(vZZmc.ravel())
        vv_norm = vZZmc/np.sqrt(vv_var)
        
        CC = np.matmul(np.transpose(vv_norm), vv_norm)
        CC = CC/n_lhs_data
        w_vbm, v_vbm = np.linalg.eig(CC)
        
        QQ = np.zeros([n_lhs_data, n_q_modes])
        vZZlhs_norm = vZZlhs/np.sqrt(vv_var)
        
        for k in range(0, n_q_modes):
             aa = project_onto_vector(vZZlhs_norm, v_vbm[:, k])            
             QQ[:, k] = aa/np.sqrt(w_vbm[k])
             
    return QQ, w_vbm, v_vbm, vv_var


#######################
# DNO Transform Things
#######################

    #    
    # These functions are defined for normalizing, standardizing, or flatenining interal to DeepONet
    #
    # Ethan sez:  decimation_factor = 2 is good, but might even be too low
    # 

def DNO_Y_transform(x, decimation_factor = 3):
    x_transform = x/decimation_factor
    return x_transform

def DNO_Y_itransform(x_transform, decimation_factor = 3):
    x = x_transform*decimation_factor
    return x

#####################
# Error calculations
#####################

def calc_log_error(pq_true, pq_surr, dx=1, ii = range(50,2750), trunc_thresh= 1*10**-4):
     #
     # Can handle the long tail problem by truncating at finite x, or 
     # truncating at finite px.  Currently, we do the latter
     #
     # This calcs log10-MAE, but we log10 it a second time for plotting? 
     #
     trunc_pq_true = np.maximum(pq_true, trunc_thresh)
     trunc_pq_surr = np.maximum(pq_surr, trunc_thresh)
     eps = np.sum(np.abs(np.log10(trunc_pq_surr) - np.log10(trunc_pq_true)))*(dx)  
     #eps = np.sum(np.abs(np.log10(pq_surr[ii]) - np.log10(pq_true[ii])))/(dx)  
     return eps
 
    
#####################
# Active Samping Things
#####################
    
#
# Peel out the Active Sampling calculation into a function, so that we
# can call it on different sets of points for AS and for error calculations
#

def acq_calculation_rom(model_list, Theta_test, inputs, qq_xx=np.linspace(-10,10,10000), 
                    input_rule='grd', sigma_n=0, numerical_eps=1*10**-16, acq_rule='US_LW',
                    cur_mode=0, as_target_quantity='mode-coefficient', n_q_modes=2,
                    v_vbm=1, w_vbm=1, vv_var=1):
    
    test_pts = Theta_test.shape[0] 
    Mean_Val, Var_Val = acq_evaluate_rom(model_list, Theta_test,sigma_n=sigma_n,
                        cur_mode=cur_mode, as_target_quantity=as_target_quantity, n_q_modes=n_q_modes,
                        v_vbm=v_vbm, w_vbm=w_vbm, vv_var=vv_var)
            
    # Determine Bounds for evaluating the metric
    x_max = np.max(Mean_Val)
    x_min = np.min(Mean_Val)
    x_int = np.linspace(x_min,x_max,10000) # Linearly space points
    #x_int_standard = np.linspace(-10,10,10000) # Static for pt-wise comparisons
    x_int_standard = qq_xx

    # Create the weights/exploitation values
    if input_rule=='pdf' :
        px = np.ones([Theta_test.shape[0],])
    else :
        px = inputs.pdf(Theta_test)
        
    sc = scipy.stats.gaussian_kde(Mean_Val.reshape(test_pts,), weights=px)   # Fit a guassian kde using px input weights
    py = sc.evaluate(x_int) # Evaluate at x_int
    py[py<numerical_eps] = numerical_eps # Eliminate spuriously small values (smaller than numerical precision)
    py_standard = sc.evaluate(x_int_standard) # Evaluate for pt-wise comparisons
    py_interp = InterpolatedUnivariateSpline(x_int, py, k=1) # Create interpolation function
    
    # Construct the weights
    wx = px.reshape(test_pts)/py_interp(Mean_Val ).reshape(test_pts)
    wx = wx.reshape(test_pts,1)
    
    # Compute the acquisition values
    if  acq_rule=='US_LW' :
        ax = wx.reshape(test_pts, )*Var_Val.reshape(test_pts, )  # This is simply w(\theta) \sigma^2(\theta) - note that x and \theta are used interchangably
    elif acq_rule == 'US' :
        ax = Var_Val
    elif acq_rule == 'US_SJ' :
        ax = wx*(Var_Val**6)
    elif acq_rule =='KUS_LW' :
        sig02 = np.min(Var_Val.reshape(test_pts, ))
        print('Estimated sig02:  {}'.format(sig02))
        ax = wx.reshape(test_pts, )*( Var_Val.reshape(test_pts, ) - sig02)
    elif acq_rule == 'RAND' :
        ax = np.random.randn(test_pts, 1)
        ax.reshape((test_pts, ))
        
    
    return Mean_Val, Var_Val, wx, ax, py, py_standard, x_int, x_int_standard



#
# Fall through a case structure that computes the AS scalar quantity differently
# depending on what we're looking for
#
# n_vbm_ensemble:   how many draws from the NN `posterior' should be called in
#                   order to compute the sample variance of the VBM.  Unlike the
#                   NN ensemble, this quantity should be more than de minimis
#
# nn_ensemble_var_rule:  at first I was calculating the 
#
def acq_evaluate_rom(model_list, Theta_test, sigma_n=0,
                    cur_mode=0, as_target_quantity='mode-coefficient', n_q_modes=2,
                    v_vbm=1, w_vbm=1, vv_var=1, n_vbm_ensemble = 25, nn_ensemble_var_rule = 'sum-var'):
    test_pts = Theta_test.shape[0] 
    n_t = v_vbm.shape[1]
    
    if as_target_quantity == 'mode-coefficient':
        cur_mean, Var_Val = model_list[cur_mode].predict(Theta_test)
        
        if sigma_n != 0 :
            Mean_Val = cur_mean + np.random.randn(cur_mean.shape[0], 1)*sigma_n
        else:
            Mean_Val = cur_mean
            
    elif (as_target_quantity == 'vbm-interval-max') | (as_target_quantity == 'vbm-interval-min'):
        mu_list = np.zeros((test_pts, n_q_modes))
        var_list = np.zeros((test_pts, n_q_modes))
        
        for k in range(0, n_q_modes):
            Mean_Val, Var_Val = model_list[cur_mode].predict(Theta_test)
            mu_list[:, k] = Mean_Val.ravel()
            var_list[:, k] = Var_Val.ravel()
            
        if nn_ensemble_var_rule == 'mc' :
            
            Mean_Val = np.zeros((test_pts, ))
            Var_Val = np.zeros((test_pts, ))
                
            for k in range(0, test_pts):
                q_list = np.zeros((n_vbm_ensemble, n_q_modes))
                vbm_list = np.zeros((n_vbm_ensemble, n_t))
                for j in range(0, n_vbm_ensemble) :
                    q_list[j, :] = mu_list[k, :] + np.random.randn(n_q_modes,)*np.sqrt( var_list[k, :])
                    vbm_list[j, :] = calc_VBM_from_Q( q_list[j, :], v_vbm, w_vbm, vv_var, n_q_modes=n_q_modes)
                    
                if as_target_quantity == 'vbm-interval-max' :
                    cur_maxes = np.amax(vbm_list, axis=1)
                    cur_mean= np.mean(cur_maxes)
                    cur_val = np.var(cur_maxes)
                else : # as_target_quantity == 'vbm-interval-min' :
                    cur_mins = np.amin(vbm_list, axis=1)
                    cur_mean= np.mean(cur_mins)
                    cur_val = np.var(cur_mins)
                
                Mean_Val[k] = cur_mean
                Var_Val[k] = cur_val
                
        elif nn_ensemble_var_rule == 'sum-var' : 
            cur_vbm = calc_VBM_from_Q(mu_list, v_vbm, w_vbm, vv_var, n_q_modes=n_q_modes)
            cur_vbm_var = calc_VBM_from_Q(var_list, v_vbm, w_vbm, vv_var, n_q_modes=n_q_modes)
            
            cur_maxes = np.amax(cur_vbm, axis=1)
            ii = np.argmax(cur_vbm, axis=1)
            
            Mean_Val = cur_maxes
            Var_Val = np.zeros(cur_vbm_var.shape[0], )
            for k in range(0, cur_vbm_var.shape[0]) :
                Var_Val[k] = cur_vbm_var[k, ii[k]]
            
        
    else:
        print('Uh-oh!  {} not recognized!'.format(as_target_quantity))
        

    return Mean_Val, Var_Val





############
# VBM Things
############

def sample_VBM_from_NN(Theta_mc, model_list, v_vbm, w_vbm, vv_var, n_q_modes=2,
                       noise_model='none', sigma_n_list=0):

    n_mc_samples = Theta_mc.shape[0]
    qq_mc = np.zeros([n_mc_samples, n_q_modes])
    for cur_mode in range(0, n_q_modes) :
        qq_mu, qq_sig2 = model_list[cur_mode].predict(Theta_mc)
        if noise_model == 'gaussian' :
            qq_sample = qq_mu + np.random.randn(qq_mu.shape[0], 1)*sigma_n_list[cur_mode]
        else : # 'none'
            qq_sample = qq_mu
             
        qq_mc[:, cur_mode] = np.reshape(qq_sample, [n_mc_samples,])
    
    zz = np.matmul(qq_mc, np.transpose(v_vbm[:, 0:n_q_modes]*np.sqrt(w_vbm[0:n_q_modes])))
    zz_scale = zz*np.sqrt(vv_var)
    return zz_scale

def sample_VBM_interval_extrema_from_NN(Theta_mc, model_list, v_vbm, w_vbm, vv_var, n_q_modes=2,
                       noise_model='none', sigma_n_list=0):
    zz_scale = sample_VBM_from_NN(Theta_mc, model_list, v_vbm, w_vbm, vv_var, n_q_modes=n_q_modes,
                           noise_model=noise_model, sigma_n_list=sigma_n_list)
    interval_max = np.maximum(zz_scale)
    interval_min = np.minimum(zz_scale)
    return interval_min, interval_max

def calc_VBM_from_Q(q_list, v_vbm, w_vbm, vv_var, n_q_modes=0):
    if n_q_modes==0:
        n_q_modes = q_list.shape[0]
    zz = np.matmul(q_list, np.transpose(v_vbm[:, 0:n_q_modes]*np.sqrt(w_vbm[0:n_q_modes])))
    zz_scale = zz*np.sqrt(vv_var)
    return zz_scale

def calc_VBM_var_from_Q_var(q_var_list, v_vbm, w_vbm, vv_var, n_q_modes=0):
    if n_q_modes==0:
        n_q_modes = q_var_list.shape[0]
    vv = np.matmul(q_var_list, np.transpose((v_vbm[:, 0:n_q_modes]**2)*w_vbm[0:n_q_modes]))
    vv_scale = vv*vv_var
    return vv_scale


##############################
# Plotting things
##############################

def plot_as_mode_diagnostics(Theta_plot, Theta, Mean_Val, Var_Val, Y, wx, ax, plot_pts, 
                        x_int_standard, cur_py_standard_truth, cur_py_standard, 
                        log10_errors_mode, fig_save_path ='.', iter_num=0, cur_mode=0):
    pxx = Theta_plot[:,0].reshape(plot_pts, plot_pts)
    pyy = Theta_plot[:,1].reshape(plot_pts, plot_pts)
 
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.3)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = gs.subplots()#(sharex='col', sharey='row')
    fig.suptitle('2D LAMP Search, Iteration '+str(iter_num))
    ax1.pcolor(pxx, pyy, Mean_Val.reshape(plot_pts, plot_pts))
    ax1.set_aspect('equal')
    ax1.annotate('Mean k={}'.format(cur_mode),
    xy=(-3, 5), xycoords='data',
    xytext=(0.7, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',color='white')
    #ax1.set_ylabel('$\theta_2$') 
    
    ax2.pcolor(pxx, pyy, Var_Val.reshape(plot_pts, plot_pts))
    ax2.plot(Theta[0:np.size(Y)-1,0], Theta[0:np.size(Y)-1,1], 'wo')
    ax2.set_aspect('equal')
    ax2.annotate('Variance k={}'.format(cur_mode),
    xy=(-3, 5), xycoords='data',
    xytext=(0.7, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',color='white') 
    
    ax3.pcolor(pxx, pyy, wx.reshape(plot_pts, plot_pts))
    ax3.set_aspect('equal')
    ax3.annotate('Danger Scores k={}'.format(cur_mode),
    xy=(-3, 5), xycoords='data',
    xytext=(0.7, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',color='white') 
    #ax3.set_ylabel('$\theta_2$') 
    #ax3.set_xlabel('$\theta_1$') 

    ax4.pcolor(pxx, pyy, ax.reshape(plot_pts, plot_pts))
    ax4.plot(Theta[-1,0], Theta[-1,1], 'ro')
    ax4.set_aspect('equal')
    ax4.annotate('Acquisition k={}'.format(cur_mode),
    xy=(-3, 5), xycoords='data',
    xytext=(0.7, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',color='white') 
    #ax4.set_xlabel('$\theta_1$')
    #ax4.set_xlim([-6,6])
    #ax4.set_ylim([-6,6])

    ax5.semilogy(x_int_standard, cur_py_standard_truth, label ='True PDF' )
    ax5.semilogy(x_int_standard, cur_py_standard, label='NN Approx.')
    ax5.set_xlim([-5, 5])
    ax5.set_ylim([10**-4,1])
    ax5.legend(loc='lower left')
    ax5.annotate('mode PDFs k={}'.format(cur_mode),
    xy=(-3, 5), xycoords='data',
    xytext=(0.7, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',color='white') 
    ax5.set_xlabel('q')
    
    ax6.plot(np.linspace(0,iter_num,iter_num+1),np.log10(log10_errors_mode[0:iter_num+1, cur_mode]), label='Error')
    ax6.annotate('Log pdf Error k={}'.format(cur_mode),
    xy=(-3, 5), xycoords='data',
    xytext=(0.7, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',color='white')
    ax6.legend(loc='lower left')
    ax6.set_xlabel('Iterations')
    
    filename = '{}plot_as_quantities_k={}_n={}'.format(fig_save_path, cur_mode, iter_num)
    plt.savefig(filename)
    #plt.show()
    #plt.close()


def plot_as_vbm_diagnostics(mm_xx, mm_pp, pz, log10_errors_vbm, fig_save_path = '.', 
                            cur_mode=0, iter_num=0):
    
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, hspace=0.15, wspace=0.3)
    (ax1, ax2) = gs.subplots()#(sharex='col', sharey='row')
    
    ax1.semilogy(mm_xx, mm_pp, label ='True PDF' )
    ax1.semilogy(mm_xx, pz, label='NN Approx.')
    ax1.set_xlim([-2.5*10**9, 2.5*10**9])
    ax1.set_ylim([10**-14,10**-8])
    ax1.legend(loc='lower left')
    ax1.annotate('VBM PDFs'.format(cur_mode),
    xy=(-3, 5), xycoords='data',
    xytext=(0.7, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',color='white') 
    ax1.set_xlabel('q')
    
    ax2.plot(np.linspace(0,iter_num,iter_num+1),np.log10(log10_errors_vbm[0:iter_num+1]), label='Error')
    ax2.annotate('Log pdf Error VBM'.format(cur_mode),
    xy=(-3, 5), xycoords='data',
    xytext=(0.7, 0.95), textcoords='axes fraction',
    horizontalalignment='right', verticalalignment='top',color='white')
    ax2.legend(loc='lower left')
    ax2.set_xlabel('Iterations')
    
    filename = '{}plot_as_quantities_vbm_n={}'.format(fig_save_path, iter_num)
    plt.savefig(filename)
    #plt.show()
    #plt.close()

