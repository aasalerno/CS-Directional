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

def dot_product_threshold_with_weights(filename,
                          threshold = 0.1,
                          sigma = 0.35/2):
    dirs = np.loadtxt(filename)
    num_vecs = dirs.shape[0]
    cnt = 0
    
    locs = np.array([])
    vals = []
    
    for i in xrange(0,num_vecs):
        for j in xrange(1,num_vecs):
            dp = np.dot(dirs[i,:],dirs[j,:])
            
            if dp >= threshold:
                cnt = cnt + 1
                locs[cnt,:] = np.vstack([locs, [i, j]])
                vals[cnt] = np.vstack([vals, np.exp((dp**2 - 1)/(2*sigma**2))])
    
    return locs, vals

def dot_product_with_mins(filename,
                          nmins = 4)
    '''
    This code exists to quickly calculate the closest directions in order to quickly get the values we need to calculate the mid matrix for the least squares fitting
    '''
    dirs = np.loadtxt(filename) # Load in the file
    num_vecs = dirs.shape[0] # Get the number of directions
    
    dp = np.zeros([num_vecs,num_vecs]) # Preallocate for speed
        
    for i in xrange(num_vecs):
        for j in xrange(1,num_vecs):
            dp[i,j] = np.dot(dirs[i,:],dirs[j,:]) # Do all of the dot products
    
    inds = np.argsort(dp) # Sort the data based on *rows*
    return inds[:,1:nmins+1], dirs

def func(x,a,b):
    return a + b*x

def calc_Mid_Matrix(filename,nmins):
    '''
    The purpose of this code is to create the middle matrix for the calculation:
        Gdiff**2 = del(I_{ijkrq}).T*M*del(I_{ijkrq})
        
    By having the M matrix ready, we can easily parse through the data trivially.
    
    We calculate M as [A*(A.T*A)**(-1)][(A.T*A)**(-1)*A.T]
    
    Where A is from (I_{ijkr} - I_{ijkq}) = A_rq * B_{ijkq}
    Note that there is a different M for each direction that we have to parse through
    
    The return is an m x 3 x 3 matrix where m is the number of directions that we have in the dataset.
    '''
    inds,dirs = dot_product_with_mins(filename,nmins)
    #dirs = np.loadtxt(filename)
    
    for i in xrange(dirs.shape[0]):
        
    
    
#def residuals(a,b):
#    return 
    
def least_Squares_Fitting(Ir,Iq,r,q,filename):
    
    dirs = np.loadtxt(filename)
    r = np.hstack([dirs[r,:],np.ones([len(r),1])])
    q = np.hstack([dirs[q,:],1])
    
    nrow, ncol = Iq.shape
    
    A = np.zeros(np.hstack([r.shape,3]))
    Irq = np.zeros(Ir.shape)
    
    for i in xrange(r.shape[0]):
        Irq[:,:,i] = Ir[:,:,i] - Iq
        A[i,:] = r[i,:] - q
    
    A = np.matrix(A);
    Aleft = np.linalg.solve((A.T*A),A.T)
    beta = np.zeros(np.hstack([Iq.shape,3]))
    
    for i in xrange(nrow):
        for j in xrange(ncol):
            beta[i,j,:] = Aleft*np.matrix(Irq[i,j,:].flat)
            Gdiffsq[i,j] = 
    
    return beta, Gdiffsq