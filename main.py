#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:09:58 2018

@author: jb
"""
#from sklearn.neighbors import NearestNeighbors
from sklearn import datasets
#from scipy.stats import wasserstein_distance
import numpy as np
import timeit
#import random
#from rsample import randomSub
#from wassKNN import wKNN
from sampling import generating, generating2
#from sklearn.preprocessing import StandardScaler
#from sklearn import preprocessing
from pr import Pr_DensityGaussian
from CV_H import bandwidth, CI_compute, mean_compute
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
#split
#mnist = fetch_mldata('MNIST original')
digitAll = datasets.load_digits()
digit = [[] for _ in range(10)]
for i in range(10):
    now = []
    now = [digitAll.data[j] for j in range(len(digitAll.target)) if digitAll.target[j] == i]
    digit[i].append(now)

# =============================================================================
# digit = [[] for _ in range(10)]
# for i in range(10):
#     now = []
#     now = [mnist.data[j] for j in range(len(mnist.target)) if mnist.target[j] == i]
#     digit[i].append(now)
# =============================================================================
# =============================================================================
# for i in range(10):
#     if i == 0:
#         start = 0
#         index = 0
#     di = datasets.load_digits(i + 1)
#     start = index
#     index = len(di.data)
#     num = index - start     
#     digit.append(di.data[start:start + num])
#     
# =============================================================================

# =============================================================================
# num_subset = 182
# num_feature = 64
# alpha = pow(10,-4)
# #with langevin with noise
# =============================================================================
    
# =============================================================================
# num_subset = 30
# #with subsample
# num_feature = 64
# alpha = pow(10,-4)
# #with langevin2 without noise
# =============================================================================
#init parameters
num_subset = 128
num_feature = 32
alpha = pow(10,-1)
newSample = []
iterNumMax = 100
mu = 1
damp = 2

# =============================================================================
# plot_generate = []
# for i in range(44):
#     target = np.asarray(digit[i % 10][0])
#     h = bandwidth(target)
#     mean = mean_compute(target, h)
#     newSample, x_initial, xnow, featureIndex, x1= generating2(iterNumMax, alpha, h, mean, num_subset, num_feature, target, mu, damp)
#     plot_generate.append(newSample[len(newSample)-1])
# =============================================================================
# =============================================================================
# #scaling tuning
# num_subset = 140
# num_feature = 40
# alpha = pow(10,-1)
# newSample = []
# iterNumMax = 2000
# mu = 1
# damp = 2
# =============================================================================
plot_generate = []
target = np.asarray(digit[5][0])
#bandwidth simple computing
#scott's factor: h = n**(-1/(d+4))
#silerman's facor: h = (n*(d+2)/4)**(-1/(d+4))
#target = target1/16
h = bandwidth(target)
#h=0.89
# =============================================================================
# bandwidths = 10 ** np.linspace(-1, 1, 100)
# grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=5)
#     #grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut(len(X)))
# grid.fit(digit.data);
# kde = grid.best_estimator_
# h = grid.best_params_.get('bandwidth')
# =============================================================================
# =============================================================================
# CIlow = CI_compute(target, h)
# print('CIlow:', CIlow)
# thres = CIlow[0] * 0.8
# =============================================================================
start = timeit.default_timer()
mean = mean_compute(target, h)
newSample, x_initial, xnow, featureIndex, x1= generating2(iterNumMax, alpha, h, mean, num_subset, num_feature, target, mu, damp) 

#visualize the generated sample
    
x = target[np.random.choice(len(target))]
#pr = Pr_DensityGaussian(x, digit.data, len(digit.data))
pr = Pr_DensityGaussian(x, target, h)
print('pr:',pr)
#print(kde.score_samples(x.reshape(1,-1)))
plt.figure(1)
plt.title('Original')
plt.imshow((x_initial).reshape((8, 8)), cmap=plt.cm.binary, interpolation='none')
plt.figure(2)
plt.title('New')
plt.imshow((xnow).reshape((8, 8)), cmap=plt.cm.binary, interpolation='none')
print('dif: x_initial - xnow ', x_initial - x1)


plot_generate.append(newSample[len(newSample)-1])
stop = timeit.default_timer()

print(stop - start)
# =============================================================================
# for i in range(len(newSample)):
#     plt.figure(i+3)
#     plt.imshow(newSample[i].reshape((8, 8)), cmap=plt.cm.binary, interpolation='none')
# =============================================================================
# =============================================================================
# #test sample
# xsample = np.random.randint(5, size=num_feature)
# xsampleF = xsample.astype(float)
# #print('xsampleF:', xsampleF)
# #subsample
# sample, featureIndex = randomSub(digit.data, num_subset, num_feature)
# 
# #KNN
# #wKNN(k, sample, xsampleF)
# =============================================================================

# =============================================================================
# for i in range(classNum):
#     for j in range(iterNum):
#         init = sampling()
# =============================================================================


# =============================================================================
# # plot real digits and resampled digits
# fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
# for j in range(11):
#     ax[4, j].set_visible(False)
#     for i in range(4):
#         im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
#                              cmap=plt.cm.binary, interpolation='nearest')
#         im.set_clim(0, 16)
#         im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
#                                  cmap=plt.cm.binary, interpolation='nearest')
#         im.set_clim(0, 16)
# 
# ax[0, 5].set_title('Selection from the input data')
# ax[5, 5].set_title('"New" digits drawn from the kernel density model')
# 
# plt.show()
# =============================================================================


