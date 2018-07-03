#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:03:16 2018

@author: jb
"""

import mnist_reader
import matplotlib.pyplot as plt
import numpy as np
from CV_H import bandwidth


#class=10, 60000x28x28 train, 10000test
X_train, y_train = mnist_reader.load_mnist('/Users/jb/Documents/NSC_MCMC/fashion-mnist/data/fashion', kind='train')
#scaling
X_scale = X_train.copy() / 255
#fashion ==> [class][0][image]  
fashion = [[] for _ in range(10)]
for i in range(10):
    now = []
    now = [X_scale[j] for j in range(len(y_train)) if y_train[j] == i]
    fashion[i].append(now)

#bandwidth selection  
h = np.zeros(10)
for i in range(10):
    if i >= 6:
        target = []
        target = np.asarray(fashion[i][0]).copy()
        h[i] = bandwidth(target)    
    
#h = [25,21,27,22,27,32,32,22,33,25]
#h_scale = [0.11,0.8,0.8,0.1,0.11,0.11,0.12,0.09,0.11,0.11]    
    
# =============================================================================
# plt.figure(1)
# plt.title('Original')
# plt.imshow(X_train[50000].reshape((28, 28)), cmap=plt.cm.binary, interpolation='none')
# =============================================================================
