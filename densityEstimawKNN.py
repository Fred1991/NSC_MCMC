#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:16:23 2018

@author: jb
"""
#import math
import numpy as np
#from math import pow
from wassKNN import wKNN
from CV_H import bandwidth

class Gradient:
    
    def __init__(self, name):
        self.name = name
        self.grad = []
        self.N = []
        self.coef = []
        self.h = []
        self.sub = []
    
# =============================================================================
#     def GradGaussianKNN(self, sample, k, x):       
#         self.sum = 0
#         self.N = len(sample)
#         self.h, _ = wKNN(k, sample, x)
#         self.coef = -1 / (self.N * pow(self.h, (k + 2)))  
#         
#         for i in range(self.N):
#             self.sum += (sample[i] - x) * pow(2 * math.pi, -k / 2) * np.exp(- np.transpose(x - sample[i]) * ((x - sample[i]) / (2 * self.h ** 2)))        
#         
#         self.grad = self.sum * self.coef       
#         return self.grad
# =============================================================================
    def GradNormalKNN(self, sample, k, x, num_feature):      
        self.sum = 0
        self.sum1 = 0
        self.sum2 = 0
        self.h = 10
        _, self.sub = wKNN(k, sample, x)
        #print('h: ', self.h)
        self.coef = 1 / (self.h ** 2)
        
        for i in range(k):
            self.sum1 += (sample[self.sub[i]] - x) * np.exp((-1 / (2 * (self.h ** 2))) * np.dot(np.transpose(x - sample[self.sub[i]]), (x - sample[self.sub[i]])))
            self.sum2 += np.exp((-1 / (2 * (self.h ** 2))) * np.dot(np.transpose(x - sample[self.sub[i]]), (x - sample[self.sub[i]])))
        self.sum = self.sum1 / self.sum2
        self.grad = self.sum * self.coef 
        #negetive grad 
        return -self.grad


# =============================================================================
#     def GradNormalKNN(self, sample, k, x, num_feature):      
#         self.sum = 0
#         self.sum1 = 0
#         self.sum2 = 0
#         self.h, self.sub = wKNN(k, sample, x)
#         #print('h: ', self.h)
#         self.coef = 1 / (self.h ** 2)
#         
#         for i in range(k):
#             self.sum1 += (sample[self.sub[i]] - x) * np.exp((-1 / (2 * (self.h ** 2))) * np.dot(np.transpose(x - sample[self.sub[i]]), (x - sample[self.sub[i]])))
#             self.sum2 += np.exp((-1 / (2 * (self.h ** 2))) * np.dot(np.transpose(x - sample[self.sub[i]]), (x - sample[self.sub[i]])))
#         self.sum = self.sum1 / self.sum2
#         self.grad = self.sum * self.coef 
#         #negetive grad 
#         return -self.grad
# =============================================================================
    
    def GradSimpleKNN(self, sample, k, x, num_feature):      
        self.sum = 0
        self.h, self.sub = wKNN(k, sample, x)
        #print('self: ', self.sub)
        self.coef = (num_feature + 2) / ((self.h ** 2) * k)
        
        for i in range(k):
            self.sum += sample[self.sub[i]] - x
        
        self.grad = self.sum * self.coef 
        #negetive grad 
        return -self.grad
    
    
    def GradNormal(self, sample, x, num_feature, h):      
        self.sum = 0
        self.sum1 = 0
        self.sum2 = 0
        #self.h = 10
        #_, self.sub = wKNN(k, sample, x)
        #scott's factor: h = n**(-1/(d+4))
        #silerman's facor: h = (n*(d+2)/4)**(-1/(d+4))
        #h = bandwidth(sample)
        #h = 27
        
        self.coef = 1 / (h ** 2)
        
        for i in range(len(sample)):
            self.sum1 += (sample[i] - x) * np.exp((-1 / (2 * (h ** 2))) * np.dot(np.transpose(x - sample[i]), (x - sample[i])))
            self.sum2 += np.exp((-1 / (2 * (h ** 2))) * np.dot(np.transpose(x - sample[i]), (x - sample[i])))
        self.sum = self.sum1 / self.sum2
        self.grad = self.sum * self.coef 
        #negetive grad 
        return -self.grad