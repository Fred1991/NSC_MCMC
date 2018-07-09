#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:09:58 2018

@author: jb
"""
import numpy as np
from sampling import generating_logp, generating_logp_NB
from pr import Pr_DensityGaussian
from CV_H import bandwidth, CI_compute, mean_compute, logp_mean_compute
import matplotlib.pyplot as plt
import mnist
import timeit
from sklearn.neighbors import KernelDensity
import math

#size 60000 * 28 * 28
dimension = 28 * 28
images = mnist.train_images()
vector2D = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
#scaling
vector2Dscale = vector2D.copy() / 255
label = mnist.train_labels()

#digit ==> [class][0][image]
digit = [[] for _ in range(10)]
num_sampleInclass = []
for i in range(10):
    now = []
    now = [vector2Dscale[j] for j in range(len(label)) if label[j] == i]
    num_now = len(now)
    digit[i].append(now)
    num_sampleInclass.append(num_now)

# =============================================================================
# #init parameters
# newSample = []
# rejectSample = []
# status = []
# iterNumMax = 1000
# 
# #default damp and inverse mass
# mu = 1
# damp = 2
# 
# #bandwidth selection
# # =============================================================================
# # h = np.zeros(10)
# # for i in range(10):
# #     if i == 1:
# #         target = []
# #         target = np.asarray(digit[i][0]).copy()
# #         h[i] = bandwidth(target)
# # =============================================================================
# start = timeit.default_timer()
# num_subset = 128
# num_feature = 28 * 28
# u = 0.05
# h = 0.15
# alpha = pow(10,-2)
# target = np.asarray(digit[2][0])
# #mean = logp_mean_compute(target, h)
# #newSample, x_initial, xnow, rejectSample, status= generating_logp(iterNumMax, alpha, h, mean, num_subset, num_feature, target, mu, damp, u)
# newSample = generating_logp_NB(iterNumMax, alpha, h, num_subset, num_feature, target, mu, damp)
# stop = timeit.default_timer()
# print('Run time',stop - start)
# =============================================================================
    
    
    
    
#h = [29,15,35,35,37,31,32,27,24,25]
#h_scale = [0.13,0.06,0.15,0.15,0.15,0.12,0.12,0.1,0.1,0.1]
    
#generate new sample
    
# =============================================================================
# mean = mean_compute(target, h)
# newSample, x_initial, xnow, featureIndex, x1= generating2(iterNumMax, alpha, h, mean, num_subset, num_feature, target, mu, damp) 
# 
# #random pick 
# x = target[np.random.choice(len(target))]
# pr = Pr_DensityGaussian(x, target, h)
# print('pr:',pr)
# 
# #visualization  
# plt.figure(1)
# plt.title('Original')
# plt.imshow(x_initial.reshape((8, 8)), cmap=plt.cm.binary, interpolation='none')
# plt.figure(2)
# plt.title('New')
# plt.imshow(xnow.reshape((8, 8)), cmap=plt.cm.binary, interpolation='none')
# print('dif: x_initial - xnow \n', x_initial - x1)
# =============================================================================


# =============================================================================
# h_scale = [0.13, 0.06, 0.15, 0.15, 0.15, 0.12, 0.12, 0.1, 0.1, 0.1]
# num_subset = 128
# num_feature = 28 * 28
# iterNumMax = 1000
# mu = 1
# damp = 2
# u = 0.05
# for i in range(10):
#     newSample = []
#     rejectSample = []
#     status = []   
#     h = h_scale[i]
#     alpha = pow(10,-1)
#     target = np.asarray(digit[i][0])
#     mean = logp_mean_compute(target, h)
#     start = timeit.default_timer()
#     newSample, x_initial, xnow, rejectSample, status= generating_logp(iterNumMax, alpha, h, mean, num_subset, num_feature, target, mu, damp, u)
#     stop = timeit.default_timer()
#     #print('Run time',stop - start)
#     out_index = str(i)
#     file1 = open('newSample'+ out_index +'.txt', 'w')
#     for item in newSample:
#         file1.write("%s\n" % item)
#     file1.close()
# 
#     file2 = open('rejectSample'+ out_index +'.txt', 'w')
#     for item in rejectSample:
#         file2.write("%s\n" % item)
#     file2.close()
# 
#     file3 = open('status'+ out_index +'.txt', 'w')
#     for item in status:
#         file3.write("%s\n" % item)
#     file3.close()
#     
#     file4 = open('time', 'w')
#     file4.write("%s\n" % (stop-start))
# =============================================================================

# =============================================================================
# for i in range(5):#num_subset
#     for j in range(5):#num_feature
#         for k in range(10):#index_class
#             for l in range(3):#alpha
#                 mean = mean_compute(digit[k], h[k])
#                 alpha = 10 ** (-l - 1)
#                 num_subset = np.floor(num_sampleInclass[k] * 0.2 * (i + 1))
#                 num_feature = np.floor(dimension * 0.2 * (j + 1))
#                 target = np.asarray(digit[k][0])
#                 newsample = generating2(iterNumMax, alpha, h[k], mean, num_subset, num_feature, target, mu, damp)[0]
#                 newsample[i][j][k][l] =  generating2(iterNumMax, alpha, h, mean, )
# =============================================================================
# =============================================================================
# file1 = open('newSample.txt', 'w')
# for item in newSample:
#     file1.write("%s\n" % item)
# file1.close()
# 
# file2 = open('rejectSample.txt', 'w')
# for item in rejectSample:
#     file2.write("%s\n" % item)
# file2.close()
# 
# file3 = open('status.txt', 'w')
# for item in status:
#     file3.write("%s\n" % item)
# file3.close()
#     
# =============================================================================
h_scale = [0.13, 0.06, 0.15, 0.15, 0.15, 0.12, 0.12, 0.1, 0.1, 0.1]
num_subset = 128
num_feature = 28 * 28
iterNumMax = 600000
mu = 1
damp = 2
u = 0.05
for i in range(1):
    newSample = []
    rejectSample = []
    status = []   
    h = h_scale[i]
    alpha = pow(10,-2)
    target = np.asarray(digit[i][0])
    start = timeit.default_timer()
    newSample = generating_logp_NB(iterNumMax, alpha, h, num_subset, num_feature, target, mu, damp)
    stop = timeit.default_timer()
    #print('Run time',stop - start)
    out_index = str(i)
    file1 = open('newSample'+ out_index +'.txt', 'w')
    for item in newSample:
        file1.write("%s\n" % item)
    file1.close()
    
