from __future__ import division
import numpy as np

def leastsquares(x,y):
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(x)
    Ex = x.sum()/N
    Ey = y.sum()/N
    Exx = (x**2).sum()/N
    Exy = (x*y).sum()/N
    m = (Exy-Ex*Ey)/(Exx-Ex**2)
    c = (Exx*Ey-Ex*Exy)/(Exx-Ex**2)
    return m, c
