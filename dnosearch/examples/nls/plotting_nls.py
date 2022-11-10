#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 21:07:44 2022

@author: ethanpickering
"""

from dnosearch import (GaussianInputs)
import scipy.io as sio
import numpy as np
import scipy
import matplotlib.pyplot as plt


print('This script is meant to be run in an IDE, but can be adjusted to print/save plots if desired.')
print('To plot, lite 3 seed results are provided in the data folder and the directory points there currently')
data_dir = '../../../../../data/nls/Lite_Results/results/'
print('If the desire is to plot results ran from this scripts on your computer, choose the following directory: ./results/')
#data_dir = './results/'

# Plot a three seed version of Figure 2a

# Load in the truth data
def Calc_Truth_PDF(rank):
    Theta_validation_file = '../../../../../data/nls/truth_data/Rank'+str(rank)+'_Xs1.mat'
    if rank==1:
        Y_validation_file = '../../../../../data/nls/truth_data/Rank'+str(rank)+'_lam-0.5_t50_1_1024_MaxAbsRe.mat'
    else:
        Y_validation_file = '../../../../../data/nls/truth_data/Rank'+str(rank)+'_lam-0.5_t50_1_100000_MaxAbsRe.mat'

    d = sio.loadmat(Theta_validation_file)
    Theta = d['Xs']
    d = sio.loadmat(Y_validation_file)
    Y = d['Ys']
    
    # Define the input propabilities
    ndim = rank*2
    mean, cov = np.zeros(ndim), np.ones(ndim)
    domain = [ [-6, 6] ] * ndim
    inputs = GaussianInputs(domain, mean, cov)
    ptheta = inputs.pdf(Theta)
    
    # Calculate the resulting output distribution
    sc = scipy.stats.gaussian_kde(Y.reshape(np.size(Y),), weights=ptheta)   # Fit a guassian kde using px input weights
    y_int = np.linspace(0,1,1000)
    py = sc.evaluate(y_int) # Evaluate at x_int
    py[py<10**-16] = 10**-16 # Eliminate spuriously small values (smaller than numerical precision)
    
    # Id there is a desire to plot the true pdf
    #plt.plot(y_int, np.log10(py)); plt.title('Truth PDF for '+str(ndim)+'D')
    #plt.ylabel('PDF'); plt.xlabel('y')
    #plt.show()
    return y_int, py, ptheta


def PDF_Error(rank,acq,model,y_int,py,ptheta,seeds,iterations,N,batch,init):
    Error = np.zeros((iterations,seeds))
    for j in range(1,seeds+1):
        for i in range(1,iterations+1):
            d = sio.loadmat(data_dir+'Rank'+str(rank)+'_'+model+'_'+acq+'_Seed'+str(j)+'_N'+str(N)+'_Batch_'+str(batch)+'_Init_'+init+'_Iteration'+str(i)+'.mat')
            #print(d)
            pya = d['py']
            #sca = scipy.stats.gaussian_kde(Ya.reshape(np.size(Ya),), weights=ptheta)   # Fit a guassian kde using px input weights
            #pya = sca.evaluate(y_int) # Evaluate at x_int
            #pya[pya<10**-16] = 10**-16 # Eliminate spuriously small values (smaller than numerical precision)
            py_diff = np.abs(np.log10(pya.reshape(1000,))-np.log10(py.reshape(1000,)))
            Error[i-1,j-1] = np.sum(py_diff[0:-1])*(y_int[1]-y_int[0])
            # plt.plot(y_int.reshape(1000,), np.log10(py.reshape(1000,)))
            # plt.plot(y_int.reshape(1000,), np.log10(pya.reshape(1000,)))
            # plt.title(str(i))
            # plt.show()
    return Error

#%% Figure 3a
rank=1
seeds = 1
iterations = 300
N = 10
batch = 1
init = 'lhs'
y_int, py, ptheta = Calc_Truth_PDF(rank)
    

Error_GP_LHS = PDF_Error(rank,'lhs','GP',y_int,py,ptheta,seeds,iterations,N,batch,init)
Error_NN_LHS = PDF_Error(rank,'lhs','DON',y_int,py,ptheta,seeds,iterations,N,batch,init)

Error_GP_US = PDF_Error(rank,'US','GP',y_int,py,ptheta,seeds,iterations,N,batch,init)
Error_NN_US = PDF_Error(rank,'US','DON',y_int,py,ptheta,seeds,iterations,N,batch,init)

Error_GP_US_LW = PDF_Error(rank,'US_LW','GP',y_int,py,ptheta,seeds,iterations,N,batch,init)
Error_NN_US_LW = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,iterations,N,batch,init)

plt.plot(np.median(np.log10(Error_NN_LHS),axis=1), 'r:', label='NN LHS');
plt.plot(np.median(np.log10(Error_NN_US),axis=1), 'r--', label='NN US');
plt.plot(np.median(np.log10(Error_NN_US_LW),axis=1), 'r',label='NN USLW' );
plt.plot(np.median(np.log10(Error_GP_LHS),axis=1), 'b:', label='GP LHS');
plt.plot(np.median(np.log10(Error_GP_US),axis=1), 'b--', label='GP US');
plt.plot(np.median(np.log10(Error_GP_US_LW),axis=1), 'b',label='GP USLW');
plt.title('Lite Version of Figure 3a (1 Seed, N=10)')
plt.legend()
plt.show()

#%% Figure 3b
rank=2
seeds = 1
iterations = 300
N = 10
batch = 1
init = 'lhs'
y_int, py, ptheta = Calc_Truth_PDF(rank)
    

Error_GP_LHS = PDF_Error(rank,'lhs','GP',y_int,py,ptheta,seeds,iterations,N,batch,init)
Error_NN_LHS = PDF_Error(rank,'lhs','DON',y_int,py,ptheta,seeds,iterations,N,batch,init)

Error_GP_US = PDF_Error(rank,'US','GP',y_int,py,ptheta,seeds,iterations,N,batch,init)
Error_NN_US = PDF_Error(rank,'US','DON',y_int,py,ptheta,seeds,iterations,N,batch,init)

Error_GP_US_LW = PDF_Error(rank,'US_LW','GP',y_int,py,ptheta,seeds,iterations,N,batch,init)
Error_NN_US_LW = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,177,N,batch,init)

plt.plot(np.median(np.log10(Error_NN_LHS),axis=1), 'r:', label='NN LHS');
plt.plot(np.median(np.log10(Error_NN_US),axis=1), 'r--', label='NN US');
plt.plot(np.median(np.log10(Error_NN_US_LW),axis=1), 'r',label='NN US-LW' );
plt.plot(np.median(np.log10(Error_GP_LHS),axis=1), 'b:', label='GP LHS');
plt.plot(np.median(np.log10(Error_GP_US),axis=1), 'b--', label='GP US');
plt.plot(np.median(np.log10(Error_GP_US_LW),axis=1), 'b',label='GP US-LW');
plt.title('Lite Version of Figure 3b (1 Seed, N=10)')
plt.legend()
plt.show()

#%% Figure 3c
rank=3
seeds = 1
iterations = 300
N = 10
batch = 1
init = 'lhs'
y_int, py, ptheta = Calc_Truth_PDF(rank)
    

Error_GP_LHS = PDF_Error(rank,'lhs','GP',y_int,py,ptheta,seeds,iterations,N,batch,init)
Error_NN_LHS = PDF_Error(rank,'lhs','DON',y_int,py,ptheta,seeds,iterations,N,batch,init)

Error_GP_US = PDF_Error(rank,'US','GP',y_int,py,ptheta,seeds,iterations,N,batch,init)
Error_NN_US = PDF_Error(rank,'US','DON',y_int,py,ptheta,seeds,iterations,N,batch,init)

Error_GP_US_LW = PDF_Error(rank,'US_LW','GP',y_int,py,ptheta,seeds,iterations,N,batch,init)
Error_NN_US_LW = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,iterations,N,batch,init)

plt.plot(np.median(np.log10(Error_NN_LHS),axis=1), 'r:', label='NN LHS');
plt.plot(np.median(np.log10(Error_NN_US),axis=1), 'r--', label='NN US');
plt.plot(np.median(np.log10(Error_NN_US_LW),axis=1), 'r',label='NN US-LW' );
plt.plot(np.median(np.log10(Error_GP_LHS),axis=1), 'b:', label='GP LHS');
plt.plot(np.median(np.log10(Error_GP_US),axis=1), 'b--', label='GP US');
plt.plot(np.median(np.log10(Error_GP_US_LW),axis=1), 'b',label='GP US-LW');
plt.title('Lite Version of Figure 3c (1 Seed, N=10)')
plt.legend()
plt.show()

#%% Figure 3d
rank=4
seeds = 1
iterations = 300
N = 10
batch = 1
init = 'lhs'
y_int, py, ptheta = Calc_Truth_PDF(rank)
    

Error_1 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,300,N,1,init)
Error_5 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,60,N,5,init)
Error_10 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,30,N,10,init)
Error_25 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,12,N,25,init)

plt.plot(np.linspace(1,300,300),np.median(np.log10(Error_1),axis=1), 'b', label='1');
plt.plot(np.linspace(1,300,60),np.median(np.log10(Error_5),axis=1), 'r', label='5');
plt.plot(np.linspace(1,300,30),np.median(np.log10(Error_10),axis=1), 'k',label='10' );
plt.plot(np.linspace(1,300,12),np.median(np.log10(Error_25),axis=1), 'g', label='25');
plt.title('Lite Version of Figure 3d (1 Seed, N=10)')
plt.legend()
plt.show()

#%% Figure 3e
rank=4
seeds = 1
iterations = 100
batch = 50
init = 'lhs'
y_int, py, ptheta = Calc_Truth_PDF(rank)
    

Error_2 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,2,50,init)
Error_4 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,4,50,init)
Error_8 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,8,50,init)
Error_16 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,16,50,init)

plt.plot(np.median(np.log10(Error_2),axis=1), 'b', label='N=2');
plt.plot(np.median(np.log10(Error_4),axis=1), 'r', label='N=4');
plt.plot(np.median(np.log10(Error_8),axis=1), 'k',label='N=8' );
plt.plot(np.median(np.log10(Error_16),axis=1), 'g', label='N=16');
plt.title('Lite Version of Figure 3e (1 Seed, batch=50)')
plt.legend()
plt.show()

#%% Figure 3f
print('Note this script requires more than one seed, here only one seed is provided and several more are necessary to run for useful results.')
rank=4
seeds = 1
iterations = 100
batch = 50
init = 'lhs'
y_int, py, ptheta = Calc_Truth_PDF(rank)
    

Error_2 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,2,50,init)
Error_4 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,4,50,init)
Error_8 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,8,50,init)
Error_16 = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,16,50,init)

plt.plot(np.var(np.log10(Error_2),axis=1), 'b', label='N=2');
plt.plot(np.var(np.log10(Error_4),axis=1), 'r', label='N=4');
plt.plot(np.var(np.log10(Error_8),axis=1), 'k',label='N=8' );
plt.plot(np.var(np.log10(Error_16),axis=1), 'g', label='N=16');
plt.title('Lite Version of Figure 3f (>1 Seed?, batch=50)')
plt.legend()
plt.show()

#%% Figure 4a
print('This will only be representative with several independent experiments.')
rank=10
seeds = 1
iterations = 100
batch = 50
y_int, py, ptheta = Calc_Truth_PDF(rank)
    
samples = np.linspace(50,50000,100)
samples_log = np.log10(samples)

Error_lhs = PDF_Error(rank,'lhs','DON',y_int,py,ptheta,seeds,100,2,50,'lhs')
Error_NN_USLW_lhs = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,2,50,'lhs')
Error_NN_USLW_pdf = PDF_Error(rank,'US_LW','DON',y_int,py,ptheta,seeds,100,2,50,'pdf')
Error_NN_US_lhs = PDF_Error(rank,'US','DON',y_int,py,ptheta,seeds,100,2,50,'lhs')
Error_NN_US_pdf = PDF_Error(rank,'US','DON',y_int,py,ptheta,seeds,100,2,50,'pdf')

plt.plot(samples,np.median(np.log10(Error_lhs),axis=1), 'ko', label='LHS');
plt.plot(samples,np.median(np.log10(Error_NN_US_lhs),axis=1), 'go', label='US: lhs');
plt.plot(samples,np.median(np.log10(Error_NN_US_pdf),axis=1), 'mo', label='US: pdf');
plt.plot(samples,np.median(np.log10(Error_NN_USLW_lhs),axis=1), 'ro', label='US_LW: lhs');
plt.plot(samples,np.median(np.log10(Error_NN_USLW_pdf),axis=1), 'bo', label='US_LW: pdf');
plt.title('Lite Version of Figure 4a (1 Seed, batch=50)')
plt.legend()
plt.show()

#%% Figure 4b
plt.plot(samples_log,np.median(np.log10(Error_lhs),axis=1), 'ko', label='LHS');
plt.plot(samples_log,np.median(np.log10(Error_NN_US_lhs),axis=1), 'go', label='US: lhs');
plt.plot(samples_log,np.median(np.log10(Error_NN_US_pdf),axis=1), 'mo', label='US: pdf');
plt.plot(samples_log,np.median(np.log10(Error_NN_USLW_lhs),axis=1), 'ro', label='US_LW: lhs');
plt.plot(samples_log,np.median(np.log10(Error_NN_USLW_pdf),axis=1), 'bo', label='US_LW: pdf');
plt.title('Lite Version of Figure 4b (1 Seed, batch=50)')
plt.legend()
plt.show()

#%% Figure 4c
from complex_noise import Noise_MMT
tf = 1
rank=10
noise = Noise_MMT([0, tf], rank)

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

j = 1
i=100
iterations = 100
batch = 50
acq = 'US_LW'
model = 'DON'
init='lhs'
N = 2
d = sio.loadmat(data_dir+'Rank'+str(rank)+'_'+model+'_'+acq+'_Seed'+str(j)+'_N'+str(N)+'_Batch_'+str(batch)+'_Init_'+init+'_Iteration'+str(i)+'.mat')
Y = d['Y']
Theta = d['Theta']


x = np.linspace(0,1,128)
inds = np.linspace(0,254,128).astype(int)


Us_initial = Theta_to_U(Theta[0:10,:], 512, 4, rank)
for i in range(0,10):
    alpha_val = Y[i]/np.max(Y[0:50])
    plt.plot(x, np.transpose(np.real(Us_initial[i,inds])), 'k', alpha=alpha_val[0], label=str(np.round(alpha_val[0],2)))
plt.legend()
plt.title('Figure 4c left')
plt.xlabel('x')
plt.ylabel('Re(u)')
plt.show()

Us_sampled = Theta_to_U(Theta[4800:4810,:], 512, 4, rank)

for i in range(0,10):
    alpha_val = Y[4800+i]/np.max(Y[4800:4810])
    plt.plot(x, np.transpose(np.real(Us_sampled[i,inds])), 'k', alpha=alpha_val[0], label=str(np.round(alpha_val[0],2)))
plt.legend()
plt.title('Figure 4c right')
plt.xlabel('x')
plt.ylabel('Re(u)')
plt.show()




#%% Figure 4d
j = 1
i=100
iterations = 100
init='pdf'
d = sio.loadmat(data_dir+'Rank'+str(rank)+'_'+model+'_'+acq+'_Seed'+str(j)+'_N'+str(N)+'_Batch_'+str(batch)+'_Init_'+init+'_Iteration'+str(i)+'.mat')
Y = d['Y']
Theta = d['Theta']


x = np.linspace(0,1,128)
inds = np.linspace(0,254,128).astype(int)

Us_initial = Theta_to_U(Theta[0:10,:], 512, 4, rank)
for i in range(0,10):
    alpha_val = Y[i]/np.max(Y[0:50])
    plt.plot(x, np.transpose(np.real(Us_initial[i,inds])), 'k', alpha=alpha_val[0], label=str(np.round(alpha_val[0],2)))
plt.legend()
plt.title('Figure 4d left')
plt.xlabel('x')
plt.ylabel('Re(u)')
plt.show()

Us_sampled = Theta_to_U(Theta[4800:4810,:], 512, 4, rank)

for i in range(0,10):
    alpha_val = Y[4800+i]/np.max(Y[4800:4810])
    plt.plot(x, np.transpose(np.real(Us_sampled[i,inds])), 'k', alpha=alpha_val[0], label=str(np.round(alpha_val[0],2)))
plt.legend()
plt.title('Figure 4d right')
plt.xlabel('x')
plt.ylabel('Re(u)')
plt.show()


