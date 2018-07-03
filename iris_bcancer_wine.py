#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:30:57 2018

@author: jb
"""

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
import numpy as np
from CV_H import bandwidth

iris = load_iris() # class=3,150x4,
wine = load_wine() # class=3,178x13,
cancer = load_breast_cancer() # class=2,569x30,

#data ==> [class][0][sample]
iris_data = [[] for _ in range(3)]
for i in range(3):
    now = []
    now = [iris.data[j] for j in range(len(iris.target)) if iris.target[j] == i]
    iris_data[i].append(now)

wine_data = [[] for _ in range(3)]
for i in range(3):
    now = []
    now = [wine.data[j] for j in range(len(wine.target)) if wine.target[j] == i]
    wine_data[i].append(now)

cancer_data = [[] for _ in range(2)]
for i in range(2):
    now = []
    now = [cancer.data[j] for j in range(len(cancer.target)) if cancer.target[j] == i]
    cancer_data[i].append(now)
    
#bandwidth selection  
h_iris = np.zeros(3)
h_wine = np.zeros(3)
h_cancer = np.zeros(2)

for i in range(3):
    target = []
    target = np.asarray(h_iris[i][0]).copy()
    h_iris[i] = bandwidth(target)    
    arget = []
    target = np.asarray(h_wine[i][0]).copy()
    h_wine[i] = bandwidth(target)  
    arget = []
    if i < 2:
        target = np.asarray(h_cancer[i][0]).copy()
        h_cancer[i] = bandwidth(target)  
    