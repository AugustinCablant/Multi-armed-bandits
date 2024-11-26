# -*- coding: utf-8 -*-
'''
Useful functions for bandit algorithms (especially KL-UCB)
'''

from math import log, sqrt, exp
import numpy as np

## A function that returns an argmax at random in case of multiple maximizers 

def randmax(A):
    maxValue=max(A)
    index = [i for i in range(len(A)) if A[i]==maxValue]
    return np.random.choice(index)


## Kullback-Leibler for Bernoulli distributions

eps = 1e-15

def klBern(x, y):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1-eps)
    y = min(max(y, eps), 1-eps)
    return x*log(x/y) + (1-x)*log((1-x)/(1-y))

def klGaussian(x,y,sigma=1):
    """KL divergence for Gaussian distribution (default stdev=1)"""
    return (x-y)^2/(2*sigma^2)


## to compute the kl-UCB index 

def klucb(x, level, div, upperbound, lowerbound=-float('inf'), precision=1e-6):
    """Generic klUCB index computation using binary search: 
    returns u>x such that div(x,u)=level where div is the KL divergence to be used.
    """
    l = max(x, lowerbound)
    u = upperbound
    while u-l>precision:
        m = (l+u)/2
        if div(x, m)>level:
            u = m
        else:
            l = m
    return (l+u)/2


def klucbBern(x, level, precision=1e-6):
    """returns u such that kl(x,u)=level for the Bernoulli kl-divergence."""
    upperbound = min(1.,x+sqrt(level/2)) 
    return klucb(x, level, klBern, upperbound, precision)

