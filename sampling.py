#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:16:18 2018

@author: jb
"""
import numpy as np
from hamiltonMove import hamilton, langevin, langevin2
#from densityEstimawKNN import GradNormalKNN
from densityEstimawKNN import Gradient as gp
from pr import Pr_DensityGaussian, Pr_KDE
from rsample import randomSub
import matplotlib.pyplot as plt

def generating(iterNumMax, alpha, h, pr_thres, num_subset, num_feature, allsample):
    #select specific feature to update
# =============================================================================
#     print('featureIndex:', len(featureIndex))
#     print('x:', len(x))
#     xs = [x[j] for j in featureIndex]
#     #print('xs: ', len(xs))
#     s = np.shape(xs)
#     vInit = np.random.uniform(0,10,s)
# =============================================================================
    #print('vint: ',len(vInit))
    g = gp('GradNormal')
    
    #init pool of samples
    generatedSample = []
    x_initial = []
    #n_all, p_all = np.shape(allsample)
    vNext = np.zeros(num_feature)
    v = np.zeros(num_feature)
    for i in range(iterNumMax):
        sample, featureIndex, sample_x = randomSub(allsample, num_subset, num_feature)
        #xs = sample[np.random.choice(len(sample))]
        #init start V
        kk = 100
        v[featureIndex] = vNext[featureIndex]
        if i == 0:
# =============================================================================
#             x1 = sample_x[np.random.choice(len(sample_x))]
#             x2 = sample_x[np.random.choice(len(sample_x))]
#             vInit = (x1[featureIndex] - x2[featureIndex]) / kk
#             #vInit = np.random.uniform(0,11,num_feature) - 5
#             #v = vInit.copy() * kk/ np.linalg.norm(vInit)
#             v = vInit.copy()
# =============================================================================
            x = sample_x[np.random.choice(len(sample_x))]
            x1 = sample_x[np.random.choice(len(sample_x))]
            vInit = np.zeros(num_feature)
            vInit[featureIndex] = (x1[featureIndex] - x[featureIndex]) / kk
            #print('vInit:',vInit)
            v = vInit.copy()
            x_initial = x.copy()
        
        #xs = [x[j] for j in featureIndex]
        grad = g.GradNormal(sample_x, x, num_feature, h)
        x, vNext = hamilton(x, v, alpha, grad, i)           
        v = np.zeros(num_feature).copy()              
# =============================================================================
#         for index in featureIndex:
#             x[featureIndex[index]] = xs[index]
# =============================================================================
        probxnew = Pr_DensityGaussian(x, allsample, h)
        xnow = x.copy()
        print('pr:', probxnew)
# =============================================================================
#         u = np.random.uniform(0,1)
#         if (probxnew/pr_thres) >= u:
#             generatedSample.append(x)
# =============================================================================
        #CI method
        if probxnew >= pr_thres:
            generatedSample.append(x)
            
# =============================================================================
#     plt.figure(1)
#     plt.title('first step')
#     plt.imshow(x1.reshape((8, 8)), cmap=plt.cm.binary, interpolation='nearest')
# =============================================================================
    return generatedSample, x_initial, xnow, featureIndex, x1



def generating2(iterNumMax, yita, h, pr_thres, num_subset, num_feature, allsample, mu, damp):
 
    g = gp('GradNormal')    
    #init pool of samples
    generatedSample = []
    x_initial = []
    n_all, p_all = np.shape(allsample)
    #vNext = np.zeros(p_all)
    v = np.zeros(p_all)
    for i in range(iterNumMax):
        sample, featureIndex, sample_x = randomSub(allsample, num_subset, num_feature)
        #kk = 0.1
        kk = 1
        #v[featureIndex] = vNext[featureIndex]
        if i == 0:
            flag0 = np.random.choice(len(sample_x))
            x = sample_x[flag0]
# =============================================================================
#             x1 = sample_x[np.random.choice(len(sample_x))]
#             vInit = np.zeros(p_all)
#             vInit[featureIndex] = (x1[featureIndex] - x[featureIndex]) / kk
#             #print('vInit:',vInit)
#             v = vInit.copy()
#             x_initial = x.copy()
# =============================================================================
            x_initial = x.copy()
        flag = np.random.choice(len(sample_x))
        x1 = sample_x[flag]
        vInit = np.zeros(p_all)
        vInit[featureIndex] = (x1[featureIndex] - x[featureIndex]) / kk
        #print('vInit:',vInit)
        v = vInit.copy()
        #xs = [x[j] for j in featureIndex]
        #grad = g.GradNormal(sample_x, x, num_feature)
# =============================================================================
#         x, vNext = hamilton(x, v, alpha, grad, i)           
# =============================================================================
        x_input = x.copy()
        for j in range(300):
# =============================================================================
#             if j == 99:
#                 x, vNext = langevin2(x, v, yita, mu, damp, grad, j)   
#             else:
#                 x, v = langevin2(x, v, yita, mu, damp, grad, j)   
# =============================================================================
            #x_previous = x.copy()
            #print('xprevious:',x_previous)
            grad = g.GradNormal(sample_x, x, num_feature)
            x, v = langevin2(x, v, yita, mu, damp, grad, j)
            
            if j >= 100:
                probxnew = Pr_DensityGaussian(x, allsample, h)
                u = np.random.uniform(0,1)
                if (probxnew/pr_thres) >= u:
                    generatedSample.append(x)
                    break
            flag_now = j
        #print('x',x)
        #print('j:',j)
        probxnew = Pr_DensityGaussian(x, allsample, h)
        #x, vNext = langevin2(x, v, yita, mu, damp, grad, i)
        v = np.zeros(p_all).copy()              
        #probxnew = Pr_DensityGaussian(x, allsample, h)
        xnow = x.copy()
        print('pr:', probxnew)
        if j == flag_now:
            x = x_input.copy()
# =============================================================================
#         u = np.random.uniform(0,1)
#         if (probxnew/pr_thres) >= u:
#             generatedSample.append(x)
#             print('flag:',flag)
#         else:
#             x = x_input.copy()
# =============================================================================
            
# =============================================================================
#     plt.figure(1)
#     plt.title('first step')
#     plt.imshow(x1.reshape((8, 8)), cmap=plt.cm.binary, interpolation='nearest')
# =============================================================================
    print('flag0:',flag0)
    return generatedSample, x_initial, xnow, featureIndex, x1


def generating3(iterNumMax, yita, h, pr_thres, num_subset, num_feature, allsample, mu, damp, u):
 
    g = gp('GradNormal')    
    #init pool of samples
    generatedSample = []
    rejectSample = []
    status = []
    v0 = []
    x_initial = []
    n_all, p_all = np.shape(allsample)
    #vNext = np.zeros(p_all)
    v = np.zeros(p_all)
    for i in range(iterNumMax):
        sample, featureIndex, sample_x = randomSub(allsample, num_subset, num_feature)
        kk = 1
        #v[featureIndex] = vNext[featureIndex]
        if i == 0:
            flag0 = np.random.choice(len(sample_x))
            x = sample_x[flag0]
            x_initial = x.copy()
        flag = np.random.choice(len(sample_x))
        x1 = sample_x[flag]
        vInit = np.zeros(p_all)
        vInit[featureIndex] = (x1[featureIndex] - x[featureIndex]) / kk
        v = vInit.copy()
        v0.append(vInit)
        x_input = x.copy()
        
        for j in range(1000):
            grad = g.GradNormal(sample_x, x, num_feature)
            x, v = langevin2(x, v, yita, mu, damp, grad, j)

        probxnew = Pr_DensityGaussian(x, allsample, h)
        v = np.zeros(p_all).copy()              
        xnow = x.copy()
        print('pr:', probxnew)
        
        #importance sampling
        if (probxnew/pr_thres) >= u:
            generatedSample.append(x) 
            status.append(1)                   
        else:
            rejectSample.append(x)
            status.append(0)
            x = x_input.copy()
            

    return generatedSample, x_initial, xnow, rejectSample, status



def generating_logp(iterNumMax, yita, h, pr_thres, num_subset, num_feature, allsample, mu, damp, u):
 
    g = gp('GradNormal')    
    #init pool of samples
    generatedSample = []
    rejectSample = []
    status = []
    v0 = []
    x_initial = []
    n_all, p_all = np.shape(allsample)
    #vNext = np.zeros(p_all)
    v = np.zeros(p_all)
    for i in range(iterNumMax):
        sample, featureIndex, sample_x = randomSub(allsample, num_subset, num_feature)
        kk = 1
        #v[featureIndex] = vNext[featureIndex]
        if i == 0:
            flag0 = np.random.choice(len(sample_x))
            x = sample_x[flag0]
            x_initial = x.copy()
        flag = np.random.choice(len(sample_x))
        x1 = sample_x[flag]
        vInit = np.zeros(p_all)
        vInit[featureIndex] = (x1[featureIndex] - x[featureIndex]) / kk
        v = vInit.copy()
        v0.append(vInit)
        x_input = x.copy()
        
        for j in range(1000):
            grad = g.GradNormal(sample_x, x, num_feature)
            x, v = langevin2(x, v, yita, mu, damp, grad, j)

        probxnew = Pr_KDE(x, allsample, h)
        v = np.zeros(p_all).copy()              
        xnow = x.copy()
        print('pr:', probxnew)
        
        #importance sampling
        if (probxnew/pr_thres) >= u:
            generatedSample.append(x) 
            status.append(1)                   
        else:
            rejectSample.append(x)
            status.append(0)
            x = x_input.copy()
            

    return generatedSample, x_initial, xnow, rejectSample, status