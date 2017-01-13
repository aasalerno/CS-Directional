from __future__ import division
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'

import os.path
import transforms as tf
import scipy.ndimage.filters
import grads
import sampling as samp
import direction as d
import optimize as opt
EPS = np.finfo(float).eps

# ----------------------------------------------------- #
# ------- MAJOR FUNCTIONS FOR USE IN MAIN CODE -------- #
# ----------------------------------------------------- #

def objectiveFunction(x, N, lam1, lam2, data, k, strtag, ph, dirWeight=0, dirs=None,
           dirInfo=[None,None,None,None], nmins=0, wavelet='db4', mode="per", a=10):
    '''
    This is the optimization function that we're trying to optimize. We are optimizing x here, and testing it within the funcitons that we want, as called by the functions that we've created
    '''
    #dirInfo[0] is M
    #import pdb; pdb.set_trace()
    tv = 0
    xfm = 0
    data.shape = N
    x.shape = N
    
    obj = np.sum(objectiveFunctionDataCons(x,N,ph,data,k))
    
    if lam1 > 1e-6:
        tv = np.sum(objectiveFunctionTV(x,N,strtag,dirWeight,dirs,nmins,dirInfo=dirInfo,a=a))
    
    if lam2 > 1e-6:
        xfm = objectiveFunctionXFM(x,N,wavelet=wavelet,mode=mode,a=a)
    
    x.shape = (x.size,) # Not the most efficient way to do this, but we need the shape to reset.
    data.shape = (data.size,)
    #output
    #print('obj: %.2f' % abs(obj))
    #print('tv: %.2f' % abs(lam1*tv))
    #print('xfm: %.2f' % abs(lam2*xfm))
    return abs(obj + lam1*tv + lam2*xfm)


def derivativeFunction(x, N, lam1, lam2, data, k, strtag, ph, dirWeight=0.1, dirs=None,
                       dirInfo=[None,None,None,None], nmins=0, wavelet="db1", mode="per", a=1.0):
    '''
    This is the function that we're going to be optimizing via the scipy optimization pack. This is the function that represents Compressed Sensing
    '''
    #import pdb; pdb.set_trace()
    disp = 0
    gTV = 0
    gXFM = 0
    
    gDataCons = grads.gDataCons(x,N,ph,data,k) # Calculate the obj function
    if lam1 > 1e-6:
        gTV = grads.gTV(x,N,strtag,dirWeight,dirs,nmins,dirInfo=dirInfo,a=a) # Calculate the TV gradient
    if lam2 > 1e-6:
        gXFM = grads.gXFM(x,N,wavelet=wavelet,mode=mode,a=a)
    x.shape = (x.size,)
    #import pdb; pdb.set_trace();
    #if disp:
        #minval = np.min(np.hstack([gObj,lam1*gTV,lam2*gXFM]))
        #maxval = np.max(np.hstack([gObj,lam1*gTV,lam2*gXFM]))
        #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        #im1 = ax1.imshow(abs(gObj),interpolation='none', clim=(minval,maxval))
        #ax1.set_title('Data Cons. Term')
        #plt.colorbar(im1,ax=ax1)
        #im2 = ax2.imshow(abs(lam1*gTV),interpolation='none',clim=(minval,maxval))
        #ax2.set_title('lam1*TV Term')
        #plt.colorbar(im2,ax=ax2)
        #im3 = ax3.imshow(abs(lam2*gXFM),interpolation='none',clim=(minval,maxval))
        #ax3.set_title('lam2*XFM Term')
        #plt.colorbar(im3,ax=ax3)
        #im4 = ax4.imshow(abs(gObj + lam1*gTV + lam2*gXFM),interpolation='none')
        #ax4.set_title('Total Grad')
        #plt.colorbar(im4,ax=ax4)
        ##plt.show()
    
    return (gDataCons + lam1*gTV + lam2*gXFM).flatten() # Export the flattened array
    
    
    
    
    
    
    
# ----------------------------------------------------- #
# -------- Individual Calculations for clarity -------- #
# ----------------------------------------------------- #

def objectiveFunctionDataCons(x, N, ph, data, k):
    obj_data = np.fft.fftshift(k)*(data - tf.fft2c(x,ph))
    return obj_data*obj_data.conj() #L2 Norm



def objectiveFunctionTV(x, N, strtag, dirWeight=0, dirs=None, nmins=0,
                        dirInfo=[None,None,None,None], a=10):
    return (1/a)*np.log(np.cosh(a*tf.TV(x,N,strtag,dirWeight,dirs,nmins,dirInfo)))
    

    
def objectiveFunctionXFM(x, N, wavelet='db4', mode="per", a=10):
    if len(N) > 2:
        xfm=0
        for kk in range(N[0]):
            wvlt = tf.xfm(x[kk,:,:],wavelet=wavelet,mode=mode)
            xfm += np.sum((1/a)*np.log(np.cosh(a*wvlt[0])))
            for i in xrange(1,len(wvlt)):
                xfm += np.sum([np.sum((1/a)*np.log(np.cosh(a*wvlt[i][j]))) for j in xrange(3)]) 
    else:
        wvlt = tf.xfm(x,wavelet=wavelet,mode=mode)
        xfm = np.sum((1/a)*np.log(np.cosh(a*wvlt[0])))
        for i in xrange(1,len(wvlt)):
            xfm += np.sum([np.sum((1/a)*np.log(np.cosh(a*wvlt[i][j]))) for j in xrange(3)])
    
    return xfm
            