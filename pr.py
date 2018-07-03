#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:17:46 2018

@author: jb
"""

import math
import numpy as np
from decimal import Decimal as D
from sklearn.neighbors import KernelDensity
#from wassKNN import wKNN

def Pr_DensityGaussian(x, sample, h):
    
    d = np.shape(sample)[1]
    N = np.shape(sample)[0]
    pr = 0

    for i in range(N):      
        #pr += ((N * (h ** d)) ** (-1)) * ((2 * math.pi) ** (-d / 2)) * np.exp((-1 / (2 * (h ** 2))) * np.dot(np.transpose(x - sample[i]), (x - sample[i])))
        a = (D(N * (h ** d))) ** (-1)
        b = D((2 * math.pi) ** (-d / 2))
        c = np.dot(np.transpose(x - sample[i]), (x - sample[i]))
        pr += a * b * c
    return pr

def Pr_KDE(x, sample, h):
    
    kde = KernelDensity(bandwidth=h, kernel='gaussian').fit(sample)
    #print(x)
    #print(np.shape(x))
    pr_log = kde.score_samples(x.reshape(1,-1))
  
    return pr_log