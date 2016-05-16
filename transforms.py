# !/usr/bin/env python -tt
#
#
#   transforms.py
#
#
#
#   Code to be a wrapper for all of the transforms that are done in order to clean up the central codes

from __future__ import division
import numpy as np
import numpy.fft as fft
from rwt import dwt, idwt
from rwt.wavelets import daubcqf
import direction as d

def fft2c(data_to_fft,axes=(-2,-1)):
    FFTdata = 1/np.sqrt(data_to_fft.size)*fft.fft2(data_to_fft,axes=axes)
    return FFTdata

def ifft2c(data_to_ifft,axes=(-2,-1)):
    IFFTdata = np.sqrt(data_to_ifft.size)*fft.ifft2(data_to_ifft,axes=axes)
    return IFFTdata

def xfm(data_to_xfm,scaling_factor = 4,L = 2):
    h = daubcqf(scaling_factor)[0]
    XFMdata = dwt(data_to_xfm,h,L)
    return XFMdata

def ixfm(data_to_ixfm,scaling_factor = 4,L = 2):
    h = daubcqf(scaling_factor)[0]
    IXFMdata = idwt(data_to_ixfm,h,L)
    return IXFMdata

def TV(data,strtag,dirs = None,nmins = 3):
    
    '''
    A finite differences sampling operation done on datasets to spply some 
    smoothing techniques.
    
    Note that the output comes back such that the stacking dimension is dimension 0
    '''
    #axisvals = []
    #for i in xrange(len(strtag)):
        #if strtag[i].lower() == 'spatial':
            #nstacks += 1
            #axisvals.append(cnt)
            #cnt += 1
        #elif strtag[i].lower() == 'diff':
            #nstacks += nmins
            #axisvals.append(0)
    
    res = np.zeros(np.hstack([nstacks, data.shape]))
    
    cnt = 0
    
    for i in xrange(len(strtag)):
        if strtag[i].lower() == 'spatial':
            #res[cnt,:,:] = np.roll(data,1,axis = axisvals[i]) - data
            res[i,:,:] = np.roll(data,1,axis = i) - data
            cnt += 1
        elif strtag[i].lower() == 'diff':
            #res[cnt:cnt+nmins,:,:] = TVDir(data)
            res[i,:,:] = d.least_Squares_Fitting(data,strtag,dirs,nmins)
            #cnt += nmins
    
    return res


#def iDx(data,shp):
	#res = data[np.hstack([0,range(shp[0]-1)]),:] - data
	#res[0,:] = -data[0,:]
	#res[-1,:] = data[-2,:]
	#return res

#def iDy(data,shp):
	#res = data[:,np.hstack([0,range(shp[1]-1)])] - data
	#res[:,0] = -data[:,0]
	#res[:,-1] = data[:,-2]
	#return res

#def iDz(data,shp):
	#res = data[:,:,np.hstack([0,range(shp[2]-1)])] - data
	#res[:,:,0] = -data[:,:,0]
	#res[:,:,-1] = data[:,:,-2]
	#return res

#def iTV(data):
	#'''
	#Inverse of the finite differences sampling operation done. Attempting to build back
	#the data after it's been TV'd
	
	#Note that the input must be put in such that the stacking dimension is dimension 0
	#'''
	
	#shp = data.shape
	
	#res = iDx(data[0,:,:],shp[1:])+ iDy(data[1,:,:],shp[1:])
	
	#if len(shp) == 4:
		#res = res + iDz(data[2,:,:,:],shp[1:])
	
	#return res

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h