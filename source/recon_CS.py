#!/usr/bin/env python -tt
#
#
# recon_CS.py
#
#
# We start with the data from the scanner. The inputs are:
#       - inFile (String) -- Location of the data
#                         -- Direct to a folder where all the data is
#       - 
#

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
#from scipy import optimize as opt
import optimize as opt
EPS = np.finfo(float).eps

def derivative_fun(x, N, lam1, lam2, data, k, strtag, ph, dirWeight=0.1, dirs=None, dirInfo=None, 
                   nmins=0, wavelet="db1", mode="per", a=1.0):
    '''
    This is the function that we're going to be optimizing via the scipy optimization pack. This is the function that represents Compressed Sensing
    '''
    #import pdb; pdb.set_trace()
    disp = 0
    gObj = grads.gObj(x,N,ph,data,k) # Calculate the obj function
    gTV = grads.gTV(x,N,strtag,dirWeight,dirs,nmins,dirInfo=dirInfo,a=a) # Calculate the TV gradient
    gXFM = grads.gXFM(x,N,wavelet=wavelet,mode=mode,a=a)
    x.shape = (x.size,)
    #import pdb; pdb.set_trace();
    if disp:
        minval = np.min(np.hstack([gObj,lam1*gTV,lam2*gXFM]))
        maxval = np.max(np.hstack([gObj,lam1*gTV,lam2*gXFM]))
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        im1 = ax1.imshow(abs(gObj),interpolation='none', clim=(minval,maxval))
        ax1.set_title('Data Cons. Term')
        plt.colorbar(im1,ax=ax1)
        im2 = ax2.imshow(abs(lam1*gTV),interpolation='none',clim=(minval,maxval))
        ax2.set_title('lam1*TV Term')
        plt.colorbar(im2,ax=ax2)
        im3 = ax3.imshow(abs(lam2*gXFM),interpolation='none',clim=(minval,maxval))
        ax3.set_title('lam2*XFM Term')
        plt.colorbar(im3,ax=ax3)
        im4 = ax4.imshow(abs(gObj + lam1*gTV + lam2*gXFM),interpolation='none')
        ax4.set_title('Total Grad')
        plt.colorbar(im4,ax=ax4)
        #plt.show()
    
    return (gObj + lam1*gTV + lam2*gXFM).flatten() # Export the flattened array

def optfun(x, N, lam1, lam2, data, k, strtag, ph, dirWeight=0, dirs=None,
           dirInfo=[None,None,None,None], nmins=0,wavelet='db4',mode="per",a=1.0):
    '''
    This is the optimization function that we're trying to optimize. We are optimizing x here, and testing it within the funcitons that we want, as called by the functions that we've created
    '''
    #dirInfo[0] is M
    #import pdb; pdb.set_trace()
    data.shape = N
    x.shape = N
    obj_data = tf.ifft2c(data - np.fft.fftshift(k)*tf.fft2c(x,ph),ph)
    obj = np.sqrt(np.sum(obj_data*obj_data.conj())) #L2 Norm
    #tv = np.sum(abs(tf.TV(x,N,strtag,dirWeight,dirs,nmins,M))) #L1 Norm
    tv = np.sum((1/a)*np.log(np.cosh(a*tf.TV(x,N,strtag,dirWeight,dirs,nmins,dirInfo))))
    #xfm cost calc
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
    
    x.shape = (x.size,) # Not the most efficient way to do this, but we need the shape to reset.
    data.shape = (data.size,)
    #output
    #print('obj: %.2f' % abs(obj))
    #print('tv: %.2f' % abs(lam1*tv))
    #print('xfm: %.2f' % abs(lam2*xfm))
    return abs(obj + lam1*tv + lam2*xfm)

def phase_Calculation(data,is_kspace = 0,is_fftshifted = 0):
    
    if is_kspace:
        data = tf.ifft2c(data)
        if is_fftshifted:
            data = np.ifftshift(data)

    filtdata = sp.ndimage.uniform_filter(data,size=5)
    return exp(1.j*np.angle(filtdata)) 

    
def gDir_lookupTable(inds):
    '''
    THe lookup table takes the indicies in, and creates a lookup table based on where a value occurs within the inds matrix. It makes all of the values in the row of the counter -1 because that is where the subtraction is happening, and +1 everywhere else.
    '''
    rows,cols = inds.shape
    lookupTable = np.zeros([rows,rows,cols])
        
    for i in xrange(rows):
        lt = np.zeros([rows,cols])
        lt[i,:] = -1
        x,y = np.where(inds==i)
        for j in xrange(x.size):
            lt[x[j],y[j]] = 1
        lookupTable[i,:,:] = lt
    
    return lookupTable
    