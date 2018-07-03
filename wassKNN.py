#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:00:18 2018

@author: jb
"""
from sklearn.neighbors import NearestNeighbors
#from scipy.stats import wasserstein_distance

#input k of KNN, samples, input x
def wKNN(k, s, x):
    #k = number of neighbors, s = samples
    #nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'brute', metric = wasserstein_distance).fit(s)
    nbrs2 = NearestNeighbors(n_neighbors = k , algorithm = 'brute').fit(s)
    #distance, indices = nbrs.kneighbors([x])
    distance2, indices2 = nbrs2.kneighbors([x])
    #print('distance:',distance,'indices:',indices)
    #print('distance2:',distance2,'indices2:',indices2)
    #return distance2, indices2
    #print('distance2:',distance2[0])
    #print('indices2:',indices2[0])
    disK = distance2[0][k - 1]
    #index = indices2[0][k - 1]
    #print('k-th distance:', disK, ' index:', index)
    return disK, indices2[0]
    
