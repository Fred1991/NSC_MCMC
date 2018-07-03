#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:15:40 2018

@author: jb
"""
#import timeit
from computeCI import mean_confidence_interval
#from sklearn.datasets import fetch_mldata
#from sklearn import datasets
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import LeaveOneOut
from pr import Pr_DensityGaussian 

def bandwidth(X):
    #bandwidths = 10 ** np.linspace(-1, 1, 100) # for digit mnist
    bandwidths = np.linspace(0.05, 0.15, 11)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=5)
    #grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut(len(X)))
    grid.fit(X);
    #kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(X)
    
    #stop = timeit.default_timer()
    #print('Runtime: ', stop - start)
    print('Best_Bandwidth: ', grid.best_params_)
    
    h = grid.best_params_.get('bandwidth')
    
    return h

def CI_compute(X, h):
    #time record
    #start = timeit.default_timer()

    #mnist = fetch_mldata('MNIST original')
    #digit = datasets.load_digits()
    
    #X = digit.data
    
    #compute CI
    Xp = np.zeros(len(X))
    for i in range(len(X)):
        Xp[i] = Pr_DensityGaussian(X[i], X, h)
        print(Xp[i])
        
    conf = [0.95, 0.75, 0.25]
    CIlow = np.zeros(len(conf)) 
    #mean, high = [], []   
    for j in range(len(conf)):   
        _, CIlow[j], _ =  mean_confidence_interval(Xp, confidence = conf[j])
        
    return CIlow

def mean_compute(X, h):

    Xp = np.zeros(len(X))
    for i in range(len(X)):
        Xp[i] = Pr_DensityGaussian(X[i], X, h)
        #print(Xp[i])
    mean = np.mean(Xp)         
    return mean

def logp_mean_compute(X, h):
    
    kde = KernelDensity(bandwidth=h, kernel='gaussian').fit(X)
    total_log_p = kde.score(X)
    mean = total_log_p / len(X)
    
    return mean