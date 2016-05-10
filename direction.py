#!/usr/bin/env python -tt
#
#
# direction.py
#
#

from __future__ import division
import numpy as np
import numpy.fft as fft
import scipy.optimize as sciopt

def dot_product_threshold(filename,
                          threshold = 0.1,
                          sigma = 0.35/2):
    
    dirs = np.loadtxt(filename)
    num_vecs = dirs.shape[0]
    cnt = 0
    
    locs = np.array([])
    vals = []
    
    for i in xrange(0,nv-1):
        for j in xrange(1,nv):
            dp = np.dot(dirs[i,:],dirs[j,:])
            
            if dp >= threshold:
                cnt = cnt + 1
                locs[cnt,:] = np.vstack([locs, [i, j]])
                vals[cnt] = np.vstack([vals, np.exp((dp**2 - 1)/(2*sigma**2))])
    
    return locs, vals

def func(x,a,b):
    return a + b*x

#def residuals(a,b):
#    return 
    
def least_Squares_Fitting(Ir,Iq,r,q,filename):
    
    dirs = np.loadtxt(filename)
    r = np.hstack([dirs[r,:],np.ones([len(r),1)])
    q = np.hstack([dirs[q,:],1])
    
    nrow, ncol = Iq.shape
    
    A = np.zeros(np.hstack([r.shape,3]))
    Irq = np.zeros(Ir.shape)
    
    for i in xrange(r.shape[0]):
        Irq[:,:,i] = Ir[:,:,i] - Iq
        A[i,:] = r[i,:] - q
    
    A = np.matrix(A);
    Aleft = np.linalg.solve((A.T*A),A.T)
    beta = np.zeros(np.hstack([Iq.shape,3])
    
    for i in xrange(nrow):
        for j in xrange(ncol):
            beta[i,j,:] = Aleft*np.matrix(Irq[i,j,:].flat)
            Gdiffsq[i,j] = 
    
    return beta, Gdiffsq