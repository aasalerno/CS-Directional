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
    XFMdata = dwt(data_to_xfm,h,L)[0]
    return XFMdata

def ixfm(data_to_ixfm,scaling_factor = 4,L = 2):
    h = daubcqf(scaling_factor)[0]
    IXFMdata = idwt(data_to_ixfm,h,L)[0]
    return IXFMdata

def TV(im,N,strtag,dirWeight = 1,dirs = None,nmins = 0,M=None):
    
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
    
    
    res = np.zeros(np.hstack([len(strtag), im.shape]),dtype=complex)
    
    cnt = 0
    for i in xrange(len(strtag)):
        if strtag[i] == 'spatial':
            #res[cnt,:,:] = np.roll(data,1,axis = axisvals[i]) - data
            res[i,:,:] = np.roll(im,-1,axis = i) - im
            #cnt += 1
        elif strtag[i] == 'diff':
            #res[cnt:cnt+nmins,:,:] = TVDir(data)
            res[i,:,:] = dirWeight*d.least_Squares_Fitting(im,N,strtag,dirs,nmins,M)
            #cnt += nmins
    
    return res

def matlab_style_gauss2D(im,shape=(3,3),sigmaX = 0):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    import cv2
    
    filtdata = cv2.GaussianBlur(im,shape,sigmaX = sigmaX)
    ph = np.conj(filtdata)/(abs(filtdata)+EPS)
    
    return ph