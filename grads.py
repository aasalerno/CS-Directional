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
    '''
    
    grad = np.zeros(x.shape)
    
    for i in xrange(x.shape[2]):
        x1 = x[...,...,i]
        grad[...,...,i] = p*x1*(x1*x1.conj()+l1smooth)**(p/2-1)
        
    return grad

def gObj(x,
         dat_scanner,
         p = 1,
         l1smooth = 1e-15):
    '''
    Here, we are attempting to get the objective derivative from the
    function. This gradient is how the current data compares to the 
    data from the scanner in order to try and enforce data consistency.
    '''
    if len(x.shape) == 2:
        x = np.reshape(x,np.hstack([x.shape, 1]))
    
    grad = np.zeros([x.shape])
    
    for i in xrange(x.shape[2]):
        x1 = x[...,...,i]
        dat = dat_scanner