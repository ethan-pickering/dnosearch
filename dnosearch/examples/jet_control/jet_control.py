#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:06:32 2022
    Active learning via NNs of a 2D stochastic SIR Pandemic Model
@author: ethanpickering
"""

# DNOSearch Imports
import numpy as np
from dnosearch import (BlackBox, GaussianInputs, DeepONet)
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
iter_num    = 0 # Iteration number
dim         = 2 # Dimension of the stochastic excitation (infection rate)
acq         = 'US_LW' # Acquisition type - currently only Likelihood-weighted uncertatiny sampling
n_init      = 3 # Initial data points
epochs      = 1000  # Number of training epochs
b_layers    = 8 # Branch Layers
t_layers    = 1 # Trunk Layers
neurons     = 300 # Number of neurons per layer
init_method = 'lhs'# How initial data are pulled
N           = 2 # Number of DNO ensembles
seed        = 3 # Seed for initial condition consistency - NOTE due to gradient descent of the DNO, the seed will not provide perfectly similar results, but will be analogous
iters_max   = 15  # Iterations to perform

print_plots =True


#%% The map we are defining here is 
# Input   = sum_i^N sin(x+phi_0+theta_i) 
# Output  = sum_i^N sin(x(end)+phi_0+theta_i) = sum_i^N sin(2*pi+phi_0+theta_i)

# This I-O relationshsip means we are interested in an identify mapping
# of the last point in the input signal

def map_def(Theta,phi_0,wavenumber):    
    f = np.zeros((np.shape(Theta)[0],1))
    #x = np.linspace(0,2*np.pi) + phi_0
    #Theta = Theta.reshape((np.shape(Theta)[0],1)) # This resize is not general
    print(Theta)
    for j in range(0,np.shape(Theta)[0]):
        for i in range(0,np.shape(Theta)[1]):
            # CHANGEED TO SQUARED
            f[j] = f[j] + np.sin(wavenumber*(2*np.pi+phi_0+Theta[j,i]))**3
    return f


def main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,iters_max,print_plots):
    
    T = 45  
    dt = 0.1
    gamma = 0.25
    delta = 0
    N_people = 10*10**7
    I0 = 50
    
    ndim = dim
    udim = dim # The dimensionality of the U components of Theta
    
    np.random.seed(seed)
    noise_var = 0
    my_map = BlackBox(map_def, noise_var=noise_var)
    mean, cov = np.zeros(ndim), np.ones(ndim)
    domain = [ [-6, 6] ] * ndim
    inputs = GaussianInputs(domain, mean, cov)
    
    if iter_num == 0:
        Theta = inputs.draw_samples(n_init, init_method)
    
    noise = Noise([0,1], sigma=0.1, ell=1)
    
    # Needed to determine U
    nsteps = int(T/dt)
    x_vals = np.linspace(0, 1, nsteps+1)
    x_vals = x_vals[0:-1]
    
    # DeepONet only needs a coarse version of the signal    
    coarse = 4
    
    # Create the X to U map, which is actually theta to U
    multiplier = 3*10**-9 # Special for the map
    
    
    def Theta_to_U(Theta,nsteps,coarse,udim):
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
    
    
    def Theta_to_Z(Theta,udim):
        if Theta.shape[1] == udim:
            Z = np.ones((Theta.shape[0], 1))
        else:
            Z = Theta[:,(udim+1):Theta.shape[1]]
        return Z         
    
    if iter_num == 0:

        # Determine the training data
        Y = np.zeros((n_init,))
        Us = Theta_to_U(Theta,nsteps,1,2)+2.55
        Us = Us*multiplier
    
        for i in range(0,n_init):
            I_temp = map_def(Us[i,:],gamma,delta,N_people,I0,T,dt, np.zeros(np.shape(Us[i,:])))
            Y[i] = I_temp[-1]
    
        Y = Y.reshape(n_init,1)
    
    
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
        x_transform = np.log10(x)/10 - 0.5
        return x_transform

    def DNO_Y_itransform(x_transform):
        x = 10**((x_transform+0.5)*10)
        return x
    
    # Keeping track of the metric
    pys = np.zeros((iters_max,10000))
    log10_errors = np.zeros((iters_max,))
    
    ##########################################
    # Loop through iterations
    ##########################################
    
    for iter_num in range(0,iters_max):
        # Train the model
        np.random.seed(np.size(Y))
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
        x_int = np.linspace(x_min,x_max,10000) # Linearly space points
        x_int_standard = np.linspace(0,10**8,10000) # Static for pt-wise comparisons

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
        ax = wx*Var_Val  # This is simply w(\theta) \sigma^2(\theta) - note that x and \theta are used interchangably
        
        # Find the optimal acquisition point
        Theta_opt = Theta_test[np.argmax(ax),:]
        Theta_opt = Theta_opt.reshape(1,2)        
        
        # Calculate the associated U
        U_opt = Theta_to_U(Theta_opt,nsteps,1,2)+2.55
        U_opt = U_opt*multiplier
        U_opt = U_opt.reshape(np.size(U_opt),1)
    
        # Pass to the Map
        I_temp = map_def(U_opt,gamma,delta,N_people,I0,T,dt,np.zeros(np.shape(U_opt)))
        Y_opt = I_temp[-1]
        Y_opt = Y_opt.reshape(1,1)
    
        # Append the value for the next step
        Theta = np.append(Theta, Theta_opt, axis = 0)
        Y = np.append(Y, Y_opt, axis = 0)
        pys[iter_num,:] = py_standard
        sio.savemat('SIR_Seed_'+str(seed)+'_N'+str(N)+'_iter_'+str(iter_num)+'.mat', {'pys':pys, 'x_int_standard':x_int_standard, 'Theta':Theta, 'U_opt':U_opt, 'I_temp':I_temp, 'wx':wx, 'ax':ax, 'py':py, 'x_int':x_int, 'Y':Y, 'Mean_Val':Mean_Val, 'Var_Val':Var_Val, 'n_init':n_init, 'N':N, 'seed':seed, 'Theta_test':Theta_test})

        if iter_num == 0: # Calulate the truth values
            d = sio.loadmat('./truth_data_py.mat')
            py_standard_truth = d['py_standard']
            py_standard_truth = py_standard_truth.reshape(10000,)
        
        log10_error = np.sum(np.abs(np.log10(py_standard[50:2750]) - np.log10(py_standard_truth[50:2750])))/(x_int_standard[2] -x_int_standard[1])  
        log10_errors[iter_num] = log10_error
        print('The log10 of the log-pdf error is: '+str(np.log10(log10_error)))
        
        if print_plots:
         
            fig = plt.figure()
            gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.3)
            (ax1, ax2, ax3), (ax4, ax5, ax6) = gs.subplots()#(sharex='col', sharey='row')
            fig.suptitle('2D Stochastic Pandemic Search, Iteration '+str(iter_num))
            ax1.pcolor(Theta_test[:,0].reshape(test_pts, test_pts), Theta_test[:,1].reshape(test_pts, test_pts), Mean_Val.reshape(test_pts, test_pts))
            ax1.set_aspect('equal')
            ax1.annotate('Mean Solution',
            xy=(-3, 5), xycoords='data',
            xytext=(0.7, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',color='white')
            #ax1.set_ylabel('$\theta_2$') 
            
            ax2.pcolor(Theta_test[:,0].reshape(test_pts, test_pts), Theta_test[:,1].reshape(test_pts, test_pts), Var_Val.reshape(test_pts, test_pts))
            ax2.plot(Theta[0:np.size(Y)-1,0], Theta[0:np.size(Y)-1,1], 'wo')
            ax2.set_aspect('equal')
            ax2.annotate('Variance',
            xy=(-3, 5), xycoords='data',
            xytext=(0.7, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',color='white') 
            
            ax3.pcolor(Theta_test[:,0].reshape(test_pts, test_pts), Theta_test[:,1].reshape(test_pts, test_pts), wx.reshape(test_pts, test_pts))
            ax3.set_aspect('equal')
            ax3.annotate('Danger Scores',
            xy=(-3, 5), xycoords='data',
            xytext=(0.7, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',color='white') 
            #ax3.set_ylabel('$\theta_2$') 
            #ax3.set_xlabel('$\theta_1$') 

            ax4.pcolor(Theta_test[:,0].reshape(test_pts, test_pts), Theta_test[:,1].reshape(test_pts, test_pts), ax.reshape(test_pts, test_pts))
            ax4.plot(Theta[-1,0], Theta[-1,1], 'ro')
            ax4.set_aspect('equal')
            ax4.annotate('Acquisition',
            xy=(-3, 5), xycoords='data',
            xytext=(0.7, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',color='white') 
            #ax4.set_xlabel('$\theta_1$')
            ax4.set_xlim([-6,6])
            ax4.set_ylim([-6,6])

            ax5.semilogy(x_int_standard, py_standard_truth, label ='True PDF' )
            ax5.semilogy(x_int_standard, py_standard, label='NN Approx.')
            ax5.set_xlim([0,2.75*10**7])
            ax5.set_ylim([10**-10,10**-6.75])
            ax5.legend(loc='lower left')
            #ax5.annotate('Output PDFs',
            #xy=(-3, 5), xycoords='data',
            #xytext=(0.7, 0.95), textcoords='axes fraction',
            #horizontalalignment='right', verticalalignment='top',color='white') 
            ax5.set_xlabel('New Infections')
            
            ax6.plot(np.linspace(0,iter_num,iter_num+1),np.log10(log10_errors[0:iter_num+1]), label='Error')
            #ax6.annotate('Log10 of log-pdf Error',
            #xy=(-3, 5), xycoords='data',
            #xytext=(0.7, 0.95), textcoords='axes fraction',
            #horizontalalignment='right', verticalalignment='top',color='white')
            ax6.legend(loc='lower left')
            ax6.set_xlabel('Iterations')
            plt.show()
                    
    sio.savemat('./data/SIR_Errors_Seed_'+str(seed)+'_N'+str(N)+'.mat', {'log10_errors':log10_errors})
    return

# Call the function
main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,iters_max,print_plots)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:13:25 2022

@author: ethanpickering
"""

# Testing the squared version u(x)**2 and u(x)**3
    
# GPSearch Imports
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
import seaborn as sns
import math

iter_num    = 0
dim         = 1
acq         = 'US' 
n_init      = 2 # init conditions
epochs      = 1000 # train 1000
b_layers    = 3 # branch
t_layers    = 1 # trunk
neurons     = 100 
init_method = 'lhs' # latin hypercube sampling
# originally N = 2
N           = 6 # add more models (8, 100, etc.)
upper_limit = 0 # dummy variable, need to remove later
n_keep      = 1 # will be used later, can you take more than one sample at a time (batching)

# iter_num    = int(sys.argv[2])
# dim         = int(sys.argv[3])
# acq         = sys.argv[4]
# n_init      = int(sys.argv[5])
# epochs      = int(sys.argv[6])
# b_layers    = int(sys.argv[7])
# t_layers    = int(sys.argv[8])
# neurons     = int(sys.argv[9])
# init_method = sys.argv[10]
# N           = int(sys.argv[11])
# upper_limit = int(sys.argv[12])
# n_keep      = int(sys.argv[13])
# norm_val    = float(sys.argv[14])




#%%


#def main(seed,iter_num,dim,acq,n_init,epochs,b_layers,t_layers,neurons,init_method,N,upper_limit,n_keep):
seed = 1 # independent experiments
dim = 1
acq = 'US'
epochs = 1000
b_layers = 3
t_layers = 1
neurons = 50
init_method = 'lhs'
N = 6
n_init = 2
iter_num = 0

ndim = dim
rank = ndim  # THIS NEEDS TO BE CHANGED TO SOMETHING USEFUL
mean, cov = np.zeros(ndim), np.ones(ndim)
domain = [ [0, 2*np.pi] ] * ndim

inputs = UniformInputs(domain)
#Theta = inputs.draw_samples(100, "grd")
np.random.seed(seed)

if iter_num == 0:
    Theta = inputs.draw_samples(n_init, init_method)

noise_var = 0 # can add in noise in the future; thing is though, don't call Blackbox
my_map = BlackBox(map_def, noise_var=noise_var)

#Theta = inputs.draw_samples(50, "grd")
#noise = Noise([0,1], sigma=0.1, ell=0.5)

# Need to determine U
nsteps = 50 # discretely define function (sin val)
# nsteps = 50 # original
#x_vals = np.linspace(0, 1, nsteps)
#x_vals = x_vals[0:-1]

# DeepONet only needs a coarse version of the signal    
coarse = 1 # Lets keep it the same for now

#y = mvnpdf(X,mu,round(Sigma,2));

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
        for i in range(0,np.shape(Theta)[1]):
            # CHANGED TO SQUARED
            #U[j,:] = U[j,:] + np.sin(wavenumber*(x+phi_0+Theta[j,i]))**3
            U[j,:] = U[j,:] + np.sin(wavenumber*(x+phi_0+Theta[j,i]))            
            # For some reason... this was working when only the first index was used... which was the direct answer I believe
    return U


def Theta_to_X(Theta,rank):
    if Theta.shape[1] == rank:
        X = np.ones((Theta.shape[0], 1))
    else:
        X = Theta[:,(2*rank):Theta.shape[1]]
    return X
    
if iter_num == 0:
    # Determine the training data
    Us = Theta_to_U(Theta,nsteps,1,ndim)
    Y = map_def(Theta,phi_0,wavenumber).reshape(n_init,1)
    

def get_corrs(wavenumber,test_pts,Mean_Val,Var_Val):
    """
    Only works for 1-dimension! Make sure that test_pts are divisible by wavenumber
    INPUTS: wavenumber, test_pts, Mean_Val, Var_Val
    OUTPUTS: m_corrs, v_corrs
    """
    m_corrs = 0
    v_corrs = 0
    
    for i in range(0,wavenumber-1):
        for j in range (i+1,wavenumber):
            m_corr = np.corrcoef(Mean_Val[i*test_pts//wavenumber:(i+1)*test_pts//wavenumber], 
                    Mean_Val[j*test_pts//wavenumber:(j+1)*test_pts//wavenumber],
                    rowvar=False)[0,1]
            v_corr = np.corrcoef(Var_Val[i*test_pts//wavenumber:(i+1)*test_pts//wavenumber], 
                    Var_Val[j*test_pts//wavenumber:(j+1)*test_pts//wavenumber],
                    rowvar=False)[0,1]
            m_corrs = m_corrs + m_corr
            v_corrs = v_corrs + v_corr
    
    # Calculate how many total combinations
    f = math.factorial
    combs = f(wavenumber) // f(2) // f(wavenumber-2)
    # For Python 3.8 and above, comb(wavenumber,2) should work

    m_corrs = m_corrs / (combs)
    v_corrs = v_corrs / (combs)
    return m_corrs, v_corrs
    

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
#model_dir = '/Users/ethanpickering/Documents/Wind/models/'
model_dir = '/Users/hchoi2/Documents/temp/models/'
#acq = 'lhs'
save_str = 'coarse'+str(coarse)+'_InitMethod_'+init_method #+'_upperlimit'+str(upper_limit)  # This alters the string for the model saving
#base_dir = '/Users/ethanpickering/Dropbox (MIT)/Wind/Runs/'
base_dir = '/Users/hchoi2/Documents/temp/runs/'
#model_dir = scratch_or_research+'epickeri/MMT/Data/Rank'+str(rank)+'/models/'
#save_dir = scratch_or_research+'epickeri/MMT/Data/Rank'+str(rank)+'/DON_Search/'   # Not Sure this is used anymore             
#save_str = 'coarse'+str(coarse)+'_lam'+str(lam)+'_BatchSize'+str(batch_size)+'_OptMethod_'+init_method+'_nguess'+str(n_guess)+'_'+objective  # This alters the string for the model saving
save_path_data = base_dir+'Wave_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_batch'+str(n_keep)+'_N'+str(N)+'_Iteration'+str(iter_num)+'.mat'
load_path_data = base_dir+'Wave_Dim'+str(ndim)+'_'+save_str+'_'+acq+'_Seed'+str(seed)+'_Init'+str(n_init)+'_batch'+str(n_keep)+'_N'+str(N)+'_Iteration'+str(iter_num-1)+'.mat'


# if iter_num > 0:
#     d = sio.loadmat(load_path_data)
#     Theta = d['Theta']
#     Y = d['Y']
    
print(np.shape(Theta))
print(np.shape(Y)) 


model_str = 'Rank'+str(np.shape(Theta)[1])+'_'+save_str+'_'+acq+'_Iter'+str(np.size(Y)-n_init+1)

iters_max = 10 # measure in 1D case: MSE between true and neural net at each iteration for multiple wavenumbers and experiments (seeds=2 or 3)
# iteration wise = 25
test_pts = 100
Thetanew = inputs.draw_samples(test_pts, "grd") # theta tests
Y_true = map_def(Thetanew,phi_0,wavenumber).reshape(test_pts**ndim,1)
MSE = np.zeros((iters_max,1))
mean_corr = np.zeros((iters_max,1))
var_corr = np.zeros((iters_max,1))

# def get_corrs(df):
#     col_correlations = pd.df.corr()
#     col_correlations.loc[:, :] = np.tril(col_correlations, k=-1)
#     cor_pairs = col_correlations.stack()
#     return cor_pairs.to_dict()

plot_dir = '/Users/hchoi2/Documents/temp/plots/cubed/'

for iters in range(0,iters_max):
    model = DeepONet(Theta, nsteps, Theta_to_U, Theta_to_X, Y, net, lr, epochs, N, M, model_dir, seed, save_period, model_str, coarse, rank)
    
    # Calculate the MSE
    Mean_Val, Var_Val = model.predict(Thetanew)
    MSE[iters] = np.mean((Y_true - Mean_Val)**2)
    
    # Wavenumber = 2, Testpts = 100, Dim = 1, N = 2:
    # Compute the correlation coefficients of each period
    # mean_corr[iters] = np.corrcoef(Mean_Val[0:test_pts//2],Mean_Val[test_pts//2:test_pts],rowvar=False)[0,1]
    # var_corr[iters] = np.corrcoef(Var_Val[0:test_pts//2],Var_Val[test_pts//2:test_pts],rowvar=False)[0,1]
    
    if dim == 1:
        # Plot predicted mean model and variance
        plt.plot(Thetanew,Mean_Val)
        plt.title('Mean Model Prediction')
        plt.plot(Theta,Y,'o')
        plt.title('Iterations:'+str(np.size(Y)-n_init))
        plt.xlabel(r'$\Theta$')
        #plt.savefig(plot_dir+'mean_NSTEPS_'+str(nsteps)+'_dim_' + str(dim) + '_wavenumber_' + str(wavenumber) + '_N_' + str(N) + '_iter_' + str(iters) + '.png')
        plt.show()
        plt.plot(Y_true)
        plt.title('True function testing')
        plt.show()
    
        # Plot predicted variance
        plt.plot(Thetanew,Var_Val/np.max(Var_Val), 'r')
        plt.title('Model Variance')
        plt.plot(Theta,np.zeros(np.size(Y)), 'o')
        Theta_opt = Thetanew[np.argmax(Var_Val)]
        plt.plot(Theta_opt, 1, 'o')
        plt.xlabel(r'$\Theta$')
        #plt.savefig(plot_dir+'var_NSTEPS_' + str(nsteps) + '_dim_' + str(dim) + '_wavenumber_' + str(wavenumber) + '_N_' + str(N) + '_iter_' + str(iters) + '.png')
        plt.show()
        
        # Get the correlation coefficients
        # mean_corr[iters] = get_corrs(wavenumber,test_pts,Mean_Val,Var_Val)[0]
        # var_corr[iters] = get_corrs(wavenumber,test_pts,Mean_Val,Var_Val)[1]
        
    
    if dim == 2:
        plt.pcolor(Mean_Val.reshape((test_pts, test_pts)))
        plt.title('Mean Model Prediction')
        plt.title('Iterations:'+str(np.size(Y)-n_init))
        plt.xlabel(r'$\Theta_1$')
        plt.ylabel(r'$\Theta_2$')
        plt.colorbar()
        #plt.savefig(plot_dir+'mean_dim_' + str(dim) + '_wavenumber_' + str(wavenumber) + '_N_' + str(N) + '_iter_' + str(iters) + '.png')
        plt.show()
        
        # Plot predicted variance
        #plt.pcolor(Var_Val/np.max(Var_Val))
        plt.pcolor(Var_Val.reshape((test_pts, test_pts))/np.max(Var_Val))
        plt.title('Model Variance')
        plt.xlabel(r'$\Theta_1$')
        plt.ylabel(r'$\Theta_2$')
        plt.colorbar()
        #plt.savefig(plot_dir+'var_dim_' + str(dim) + '_wavenumber_' + str(wavenumber) + '_N_' + str(N) + '_iter_' + str(iters) + '.png')
        plt.show()
        
        # mean_corr[iters] = np.corrcoef(Mean_Val.reshape(test_pts,test_pts)[0:test_pts//2,0:test_pts//2].reshape(test_pts//2*test_pts//2,1),Mean_Val.reshape(test_pts,test_pts)[test_pts//2:test_pts,test_pts//2:test_pts].reshape(test_pts//2*test_pts//2,1),rowvar=False)[0,1]
        # var_corr[iters] = np.corrcoef(Var_Val.reshape(test_pts,test_pts)[0:test_pts//2,0:test_pts//2].reshape(test_pts//2*test_pts//2,1),Var_Val.reshape(test_pts,test_pts)[test_pts//2:test_pts,test_pts//2:test_pts].reshape(test_pts//2*test_pts//2,1),rowvar=False)[0,1]

    # We will simply impose the US sampling technique
    Theta_opt = Thetanew[np.argmax(Var_Val)]
    Theta_opt = Theta_opt.reshape(1,ndim)
    #Us_opt = Theta_to_U(Theta_opt,nsteps,1,ndim)
    Y_opt = map_def(Theta_opt,phi_0,wavenumber).reshape(1,1)
    Theta = np.append(Theta, Theta_opt, axis = 0)
    Y = np.append(Y, Y_opt, axis = 0)

# Plot the MSE vs. iterations
plt.semilogy(MSE)
plt.title('MSE')
plt.xlabel('Iterations')
plt.show()

plt.plot(mean_corr)
plt.title('Mean correlation coefficient between ' + str(wavenumber) + ' periods')
plt.xlabel('Iterations')
plt.show()

plt.plot(var_corr)
plt.title('Variance correlation coefficient between ' + str(wavenumber) + ' periods')
plt.xlabel('Iterations')
plt.show()

