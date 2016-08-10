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
import pywt
#from rwt.wavelets import daubcqf
import direction as d

EPS = np.finfo(float).eps

def fft2c(data_to_fft,ph,axes=(-2,-1)):
    FFTdata = 1/np.sqrt(data_to_fft.size)*fft.fft2(data_to_fft*ph,axes=axes);
    return FFTdata

def ifft2c(data_to_ifft,ph,axes=(-2,-1)):
    IFFTdata = np.sqrt(data_to_ifft.size)*fft.ifft2(data_to_ifft,axes=axes)*ph;
    return IFFTdata

def xfm(data_to_xfm,wavelet = 'db1'): #Usually scaling_factor = 4, but for the haar, it's 2
    XFMdata = pywt.wavedec2(data_to_xfm,wavelet,'sym')
    return XFMdata

def ixfm(data_to_ixfm,wavelet = 'db1'):
    IXFMdata = pywt.waverec2(data_to_ixfm,wavelet,'sym')
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
    
    filtdata = cv2.GaussianBlur(im.real*1,shape,sigmaX) + cv2.GaussianBlur(im.imag*1,shape,sigmaX)*1j;
    ph = np.conj(filtdata)/(abs(filtdata)+EPS)
    
    return ph

def toMatrix(x):
    ''' Go from [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)] to a 2D image'''
    
    ax = []
    for i in xrange(len(x)):
        ax.append(xfmData[i][0].shape[0])
        
    N = sum(ax)
    res = np.zeros([N,N])
            
    res[0:ax[0],0:ax[0]] = x[0]
    
    for i in xrange(1,len(ax)):
        # Now we need to push the correct cH, cV and cD to the right spots
        #
        res[0:sum(ax[0:i]),sum(ax[0:i]):sum(ax[0:i+1])] = x[i][0] # cH
        res[sum(ax[0:i]):sum(ax[0:i+1]),0:sum(ax[0:i])] = x[i][1] # cV
        res[sum(ax[0:i]):sum(ax[0:i+1]),sum(ax[0:i]):sum(ax[0:i+1])] = x[i][2] # cD
    
    return res, ax
        

def fromMatrix(res,ax):
    
    ''' Go from a 2D image to [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]'''
    
    
    x = []
    
    x.append(res[0:ax[0],0:ax[0]])
    
    for i in xrange(1,len(ax)):
        x.append([])
        x[i].append(res[0:sum(ax[0:i]),sum(ax[0:i]):sum(ax[0:i+1])]) # cH
        x[i].append(res[sum(ax[0:i]):sum(ax[0:i+1]),0:sum(ax[0:i])]) # cV
        x[i].append(res[sum(ax[0:i]):sum(ax[0:i+1]),sum(ax[0:i]):sum(ax[0:i+1])]) # cD
    
    return x
    
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