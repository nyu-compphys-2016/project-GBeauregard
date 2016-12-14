from __future__ import division
import math
import numpy as np

def FindRoot(f,fprime,guess,accuracy):
    x = guess
    delta = 1
    while abs(delta)>accuracy:
        delta = f(x)/fprime(x)
        x -= delta
    return x
