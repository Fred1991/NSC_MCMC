#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 09:55:30 2018

@author: jb
"""
import numpy as np
#import random

#input data, subset of sample, subset of feature
def randomSub(s, ns, nf):
    #get the size and dimension of the data
    num_sample = np.shape(s)[0]
    num_feature = np.shape(s)[1]
    #input data to get the random Rsize subset of both samples and their dimensions
    sample_x = s[np.random.choice(num_sample, ns, replace=False)]
    #print(np.shape(sample_iris))
    #at least one sample or one feature selected
    random_feature = np.random.choice(num_feature, nf, replace=False)
    #print('randomFeature:', random_feature)
    #new list 
    result = []
    #print('Num_sample:',len(sample_x),' Num_feature:',len(random_feature))
    for i in range(0,len(sample_x)):
        result.append(sample_x[i][random_feature])
    resultarray = np.asarray(result)
    return resultarray, random_feature, sample_x

# =============================================================================
# if __name__ == "__main__":
#     import sys
#     fib(int(sys.argv[1]))
# =============================================================================
