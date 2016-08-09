#
#
# grads.py
#
#
from __future__ import division
import numpy as np
import transforms as tf
import matplotlib.pyplot as plt


def gXFM(x,N,
         p = 1,
         l1smooth = 1e-15):
    '''
    In this code, we apply an approximation of what the 
    value would be for the gradient of the XFM (usually wavelet)
    on the data. The approximation is done using the form:
        
    |x| =. sqrt(x.conj()*x)
    
    Using this, we are tryin to come up with a form that is cts about
    all x.
    
    Because of how this is done, we need to ensure that it is applied on
    a slice by slice basis.
    
    Inputs:
    [np.array] x - data that we're looking at
    [int]      p - The norm of the value that we're using
    [float]  l1smooth - Smoothing value
    
    Outputs:
    [np.array] grad - the gradient of the XFM
    
    '''
    
    
    x0 = x.reshape(N)
    grad = np.zeros(N)
    #    for i in xrange(x.shape[2]):
    #        x1 = x[...,...,i]
    #        grad[...,...,i] = p*x1*(x1*x1.conj()+l1smooth)**(p/2-1)
    grad = p*x0*(x0*x0.conj()+l1smooth)**(p/2.0-1)
    return grad

def gObj(x,N,ph,
         data_from_scanner,
         samp_mask):
    '''
    Here, we are attempting to get the objective derivative from the
    function. This gradient is how the current data compares to the 
    data from the scanner in order to try and enforce data consistency.
    
    Inputs:
    [np.array] x - data that we're looking at
    [np.array] data_from_scanner - the original data from the scanner
    [int/boolean] samp_mask - Mask so we only compare the data from the regions of k-space that were sampled
    [int]      p - The norm of the value that we're using
    [float]  l1smooth - Smoothing value
    
    Outputs:
    [np.array] grad - the gradient of the XFM
    
    '''
    if len(x.shape) == 2:
        x = np.reshape(x,np.hstack([x.shape, 1]))

    #grad = np.zeros([x.shape])

    # Here we're going to convert the data into the k-sapce data, and then subtract
    # off the original data from the scanner. Finally, we will convert this data 
    # back into image space
    x0 = x.reshape(N)
    data_from_scanner.shape = N
    x_data = np.fft.fftshift(samp_mask)*tf.fft2c(x0,ph); # Issue, feeding in 3D data to a 2D fft alg...
    
    grad = 2*tf.ifft2c(data_from_scanner - x_data,ph);
    
    return grad

def gTV(x,N,strtag,dirWeight,dirs = None,nmins = 0, dirInfo = None, p = 1,l1smooth = 1e-15):
    
    if dirInfo:
        M = dirInfo[0]
        dIM = dirInfo[1]
        Ause = dirInfo[2]
        inds = dirInfo[3]
    else:
        M = None
        dIM = None
        Ause = None
        inds = None
    
    x0 = x.reshape(N)
    grad = np.zeros(np.hstack([len(strtag),N]))
    TV_data = tf.TV(x0,N,strtag,dirWeight,dirs,nmins,M)
    k = .5
    for i in xrange(len(strtag)):
       if strtag[i] == 'spatial':
           TV_dataRoll = np.roll(TV_data[i,:,:],1,axis=i)
           grad[i,:,:] = -np.tanh(k*(TV_data[i,:,:])) + np.tanh(k*(TV_dataRoll))
       elif strtag[i] == 'diff':
           for d in xrange(N[i]):
               
               dDirx = np.zeros(np.hstack([N,M.shape[1]])) # dDirx.shape = [nDirs,imx,imy,nmins]
               
                for ind_q in xrange(N[i]):
                    for ind_r in xrange(M.shape[1]):
                        dDirx[ind_q,:,:,ind_r] = x0[inds[ind_q,ind_r],:,:] - x0[ind_q,:,:]
                                      
               
                for comb in xrange(len(Ause[kk])):
                    colUse = Ause[dir][comb]
                    for qr in xrange(M.shape[1]):
                        grad[i,d,:,:] = np.dot(dIM[d,qr,colUse],dDirx[d,:,:,qr]) + grad[i,d,:,:] 
    
    # Need to make sure here that we're iterating over the correct dimension
    # As of right now, this assumes that we're working on a slice by slice basis
    # I'll have to implement 3D data work soon.
    
    grad = np.sum(grad,axis=0)
    return grad