#
#
# grads.py
#
#

import numpy as np
import transforms as tf

def gXFM(x,
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
    
    grad = np.zeros(x.shape)
    
    for i in xrange(x.shape[2]):
        x1 = x[...,...,i]
        grad[...,...,i] = p*x1*(x1*x1.conj()+l1smooth)**(p/2-1)
        
    return grad

def gObj(x,
         data_from_scanner,
         samp_mask,
         p = 1,
         l1smooth = 1e-15):
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
    
    grad = np.zeros([x.shape])
    
    
    X_data = tf.ifft2c(samp_mask*tf.fft2c(x));
    grad = data_from_scanner - x_data;
    
    return grad
