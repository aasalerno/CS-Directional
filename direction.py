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
from numpy.linalg import inv



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

def dot_product_with_mins(dirs,
                          nmins = 4):
    '''
    This code exists to quickly calculate the closest directions in order to quickly get the values we need to calculate the mid matrix for the least squares fitting
    '''
    #dirs = np.loadtxt(filename) # Load in the file
    num_vecs = dirs.shape[0] # Get the number of directions
    
    dp = np.zeros([num_vecs,num_vecs]) # Preallocate for speed
        
    for i in xrange(num_vecs):
        for j in xrange(num_vecs):
            dp[i,j] = np.dot(dirs[i,:],dirs[j,:]) # Do all of the dot products
    
    inds = np.fliplr(np.argsort(abs(dp))) # Sort the data based on *rows*
    return inds[:,1:nmins+1]

def func(x,a,b):
    return a + b*x

def calc_Mid_Matrix(dirs,nmins):
    '''
    The purpose of this code is to create the middle matrix for the calculation:
        Gdiff**2 = del(I_{ijkrq}).T*M*del(I_{ijkrq})
        
    By having the M matrix ready, we can easily parse through the data trivially.
    
    We calculate M as [A*(A.T*A)**(-1)][(A.T*A)**(-1)*A.T]
    
    Where A is from (I_{ijkr} - I_{ijkq}) = A_rq * B_{ijkq}
    Note that there is a different M for each direction that we have to parse through
    
    The return is an nDirs x nMins x nMins matrix where m is the number of directions that we have in the dataset.
    '''
    
    for i in range(30):
        if dirs[i,2] < 0:
            dirs[i,:] = -dirs[i,:]

        inds = dot_product_with_mins(dirs,nmins)
        #dirs = np.loadtxt(filename)
        
        M = np.zeros([dirs.shape[0],nmins,nmins])
        
    for qDir in xrange(dirs.shape[0]):
        A = dirs[inds[qDir,:],:]-dirs[qDir,:]
        #datHold = datHold/np.linalg.norm(datHold) # Should I do this? Normalizes the vectors
        #A = np.hstack([A, datHold])
        # I apologize for how messy this is
        Ahat = np.dot(inv(np.dot(A.T,A)),A.T)
        M[qDir,:,:] = np.dot(Ahat.T,Ahat)
    
    return M
        
    
#def residuals(a,b):
#    return 
    
def least_Squares_Fitting(x,N,strtag,dirs,inds,M):
    
    x0 = x.copy().reshape(N)
    nmins = inds.shape[1]
    dirloc = strtag.index("diff")
    x0 = np.rollaxis(x0,dirloc)
    
    for q in xrange(dirs.shape[0]):
        r = inds[q,:]
    
        # Assume the data is coming in as image space data and pull out what we require
        Iq = x0[q,:,:]
        Ir = x0[r,:,:]
        nrow, ncol = Iq.shape
    
        #A = np.zeros(np.hstack([r.shape,3]))
        Irq = Ir - Iq # Iq will be taken from Ir for each part of axis 0
    
        #Aleft = np.linalg.solve((A.T*A),A.T)
        #beta = np.zeros(np.hstack([Iq.shape,3]))
        
        Gdiffsq = np.zeros(N)
        
        for i in xrange(nrow):
            for j in xrange(ncol):
                Gdiffsq[q,i,j] = np.dot(np.dot(Irq[:,i,j].reshape(1,nmins),M[q,:,:]),Irq[:,i,j].reshape(nmins,1))
    
    # This line puts the data back into the orientation that it was in before
    Gdiffsq = np.rollaxis(Gdiffsq,0,dirloc)
    
    return Gdiffsq