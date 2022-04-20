#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:11:36 2021
    For use with DNOSearch for actively training neural operators
@author: ethanpickering
"""

import numpy as np
import deepxde as dde
# Loss function for DeepONet - can be customized
def custom_mean_squared_error(y_true, y_pred):
    error = np.ravel((y_true - y_pred) ** 2)
    return np.mean(error)


class DeepONet(object):
    """ A class for training and ensemble of Deep Neural Operators via DeepONet model
    Parameters
    ---------
    Theta : float, n x d-Dimensional set of Random Parameters, (m is samples size, d is dimension)
        Theta is composed of Theta_u and Theta_z
        Theta_u is transformed to functions U and Theta_z is transformed to parameters Z
    nsteps : float, 
        number of total steps to be computed
    Theta_to_U: user defined function
        transforms Theta_u to input functions U
    Theta_to_Z: user defined function
        transforms Theta_z to input functions Z
    Y : float
        Observed Values
    net : FILL IN
    lr : float
        learning rate
    epochs : float
	number of epochs to train each neural net
    N : float
        number of randomly initialized ensemble members
    model_dir : string
        directory to save the DeepONet model
    seed : float
        provided for saving independent models in the same directory
    model_str : string
        special string for delineating models
    coarse : int
        coarsens the input function note that m = nsteps/coarse
    rank : int
     	delineates the length of Theta_u
    Attributes
    ----------
    Theta, nsteps, Theta_to_U, Theta_to_X, Y, net, lr, epochs, N, model_dir, seed, save_period, model_str, coarse, rank
    """
    
    def __init__(self, Theta, nsteps, Theta_to_U, Theta_to_Z, Y, net, lr, epochs, N, model_dir, seed, save_period, model_str, coarse, rank, Mean_Real):
        
        self.net = net
        self.lr = lr
        self.epochs = epochs
        self.N = N
        self.rank = rank
        self.normalizer = None
        self.model_dir = model_dir
        self.seed = seed
        self.model_str = model_str
        self.save_period = save_period
        self.coarse = coarse       
        self.Theta_to_U = Theta_to_U
        self.Theta_to_Z = Theta_to_Z   
        self.Mean_Real = Mean_Real        

        self.Theta = Theta
        self.nsteps = nsteps 
        # Transform to U and Z values       
        self.U = self.Theta_to_U(self.Theta, self.nsteps, self.coarse,self.rank)
        self.Z = self.Theta_to_Z(self.Theta,self.rank)
        self.Y = Y
        # Making Dummy Variable for testing since it is redundant right now
        self.Utest = self.U[0,:].reshape((1,np.size(self.U[0,:])))
        self.Ztest = self.Z[0,:].reshape((1,np.size(self.Z[0,:])))
        self.Ytest = self.Y[0,:].reshape((1,np.size(self.Y[0,:])))
        
        # Report the sizes of the functions
        print(np.shape(self.U))
        print(np.shape(self.Z))
        print(np.shape(self.Y))
        print(np.shape(self.Utest))
        print(np.shape(self.Ztest))
        print(np.shape(self.Ytest))

        # Initialize DeepONet dataset
        self.data = dde.data.OpDataSet(X_train=(self.U,self.Z), y_train=self.Y, X_test=(self.Utest,self.Ztest), y_test=self.Ytest)
        
        # Initilize a list of DeepONet models
        self.modelN = list()
        for i in range(0,N):
            self.modelN = np.append(self.modelN,dde.Model(self.data,net))
            # Compile model N
            self.modelN[i].compile("adam", lr=lr, metrics=[custom_mean_squared_error])
            checker = dde.callbacks.ModelCheckpoint(self.model_dir+"model/N"+str(i)+"seed"+str(self.seed)+"_"+self.model_str+"_model.ckpt", save_better_only=False, period=save_period)
            self.modelN[i].losshistory, self.modelN[i].train_state = self.modelN[i].train(epochs=self.epochs, callbacks=[checker]) #Training Model batch_size = 10000
        
        # Append all models together
        self.model = list()
        for i in range(0,N):
            self.model = np.append(self.model,self.modelN[i])
    
    # Thetanew are new parameter values
    def predict_basic(self, Thetanew):
        # This finds the ensemble mean and variance from N (initialized models)
        mean_vals = np.zeros((np.shape(Thetanew)[0], self.N))
        U = self.Theta_to_U(Thetanew, self.nsteps, self.coarse, self.rank)
        Z = self.Theta_to_Z(Thetanew, self.rank)
        for i in range(0,self.N):
            self.model[i].restore(self.model_dir+"model/N"+str(i)+"seed"+str(self.seed)+"_"+self.model_str+"_model.ckpt-" + str(self.epochs), verbose=0)
            temp = self.model[i].predict((U,Z))
            mean_vals[:,i] = temp.reshape(Thetanew.shape[0])
        mean_val = np.mean(mean_vals,axis = 1)
        var_val = np.var(mean_vals,axis = 1)
        
        return mean_val.reshape(np.shape(Thetanew)[0],1), var_val.reshape(np.shape(Thetanew)[0], 1)
    
    def predict_all(self, Thetanew):
        # This computes and reports all ensemble values
        prediction_vals = np.zeros((np.shape(Thetanew)[0], self.N))
        U = self.Theta_to_U(Thetanew, self.nsteps, self.coarse, self.rank) # This step can be memory intensive.
        Z = self.Theta_to_Z(Thetanew, self.rank)
        UZnew = (U,Z)
        for i in range(0,self.N):
            self.model[i].restore(self.model_dir+"model/N"+str(i)+"seed"+str(self.seed)+"_"+self.model_str+"_model.ckpt-" + str(self.epochs), verbose=0)
            temp = self.model[i].predict((UZnew))
            prediction_vals[:,i] = temp.reshape(np.shape(UZnew[0])[0],)
        return prediction_vals
    
    # Calls predict_all and computes mean and variance - should consider a normalizing layer here
    # A normalizing layer here would be able to take into account any transformations for learning the underlying functions
    def predict(self, Thetanew):
        prediction_vals = self.predict_all(Thetanew)
        real_vals = self.Mean_Real(prediction_vals)
        mean_vals = np.mean(real_vals,axis = 1).reshape(np.shape(Thetanew)[0],1)
        var_vals = np.var(real_vals,axis = 1).reshape(np.shape(Thetanew)[0],1)
        
        return mean_vals, var_vals
    
    def predictive_gradients(self, Thetanew, eps=1e-2):
    # Currently optimized to compute in y space, rather than x space
        mean0, var0 = self.predict(Thetanew)
        mean_jac    = np.zeros((Thetanew.shape[0],Thetanew.shape[1]))
        var_jac     = np.zeros((Thetanew.shape[0],Thetanew.shape[1]))
        EPS_0       = np.zeros((Thetanew.shape[0],Thetanew.shape[1]))
        Theta_perturb   = np.zeros((Thetanew.shape[0]*Thetanew.shape[1],Thetanew.shape[1]))

        # Vector version.
        # Contruct perturbations (Could use an identity matrix instead)
        for i in range(Thetanew.shape[0]):
            for ii in range(Thetanew.shape[1]):
                Theta_perturb[i*2+ii,:]     = Thetanew[i,:]
                Theta_perturb[i*2+ii,ii]    = Theta_perturb[i*2+ii,ii] + eps   
        # Compute Jacobian
        dmean, dvar = self.predict(Theta_perturb)
        dmean = dmean.reshape((Thetanew.shape[0],Thetanew.shape[1]))
        dvar = dvar.reshape((Thetanew.shape[0],Thetanew.shape[1]))
        mean_jac  = (dmean-mean0)/eps
        var_jac  = (dvar-var0)/eps
        
        return mean_jac, var_jac
    
    #
    def training(self):
        len_train_data = int(self.epochs/100)+1
        training_data = np.zeros((len_train_data, self.N))
        # This is hardcoded and must be changed - not exactly clear how to change due to DeepONet/tensorflow hardcode
        # The full set of data is always saved (at least as currently coded) in 100 iterations
        keep = np.linspace(0,self.epochs/100,len_train_data).astype(int)
        for i in range(0,self.N):
            self.model[i].restore(self.model_dir+"model/N"+str(i)+"seed"+str(self.seed)+"_"+self.model_str+"_model.ckpt-" + str(self.epochs), verbose=0)
            temp1 = self.model[i].losshistory
            temp2 = np.array(temp1.loss_train)
            training_data[:,i] = temp2[keep,:].reshape(len_train_data,) # This is currently hardcoded...

        return training_data
