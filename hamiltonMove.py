#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:54:08 2018

@author: jb
"""
import numpy as np

def hamilton(x, v, alpha, grad, iterNum):
    xNext = x + alpha * v
    vNext = v - alpha * grad - (3 / (iterNum + 1)) * (xNext - x) 
    return xNext, vNext
    
def hamilton2(x, v, alpha, grad, iterNum):    
    xNext = x + alpha * v
    vNext = v - alpha * grad - (3 / (iterNum + 1)) * (xNext - x) + alpha * np.random.normal(0,1,len(v))
    return xNext, vNext

def langevin(x, v, yita, mu, damp, grad, iterNum):
    xNext = x + yita * v
    vNext = v - damp * yita * v - yita * mu * grad + np.sqrt(2 * damp * mu * yita) * np.random.normal(0,1,len(v))
    return xNext, vNext

def langevin2(x, v, yita, mu, damp, grad, iterNum):
    xNext = x + yita * v
    vNext = v - damp * yita * v - yita * mu * grad 
    return xNext, vNext


    