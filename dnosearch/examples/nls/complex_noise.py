#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:29:26 2021

@author: ethanpickering
"""
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio

class Noise_MMT:

    def __init__(self, domain, nRV, sigma=1, ell=0.35):
        
        self.ti = domain[0]
        self.tf = domain[1]
        self.tl = domain[1] - domain[0]
        self.nRV = nRV
        try:
             d = sio.loadmat('./savedata_phi_lam.mat')
        except FileNotFoundError:
            try:
                d = sio.loadmat('otherdirector/savedata_phi_lam.mat')
            except FileNotFoundError:
                print('Loading local computer file')
                d = sio.loadmat('otherlocaldirectory/savedata_phi_lam.mat')
        phi = d['phi']
        lam = d['lam']
        self.phi = phi[:,0:self.nRV]
        self.lam = lam[0:self.nRV]
        
    def get_covariance(self, sigma, ell):
        m = 500 + 1
        self.t = np.linspace(self.ti, self.tf, m)
        self.dt = self.tl/(m-1)
        self.period = 1
        R = np.zeros([m, m],dtype=np.complex_)
        for i in range(m):
            for j in range(m):
                tau = self.t[j] - self.t[i]
                R[i,j] = sigma*np.exp(1j*2*np.sin(np.pi*np.abs(tau)/self.period)**2)*np.exp(-2*np.sin(np.pi*np.abs(tau)/self.period)**2/(ell**2))
                
        return R*self.dt

    def kle(self, R):
        lam, phi = np.linalg.eigh(R)
        phi = phi/np.sqrt(self.dt)
        idx = lam.argsort()[::-1]
        lam = lam[idx]
        phi = phi[:,idx]
        return lam, phi

    def get_eigenvalues(self, trunc=None):
        return self.lam[0:trunc]

    def get_eigenvectors(self, trunc=None):
        return self.phi[:,0:trunc]
    
    def get_sample(self, xi):
        xi = np.transpose(np.atleast_2d(xi))
        #nRV = np.asarray(xi).shape[0]         
        lam_sq = self.lam**(1/2)
        weights = np.transpose(lam_sq*xi)
        sample = np.inner(self.phi, weights);
        return sample
    
