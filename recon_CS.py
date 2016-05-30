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
import os.path
import transforms as tf
import scipy.ndimage.filters
import grads
import sampling as samp
import direction as d
from scipy import optimize as opt

EPS = np.finfo(float).eps

def derivative_fun(x,N,lam1,lam2,data,k,strtag,dirWeight = 0,dirs = None,M = None,nmins = 0, scaling_factor = 4,L = 2):
    '''
    This is the function that we're going to be optimizing via the scipy optimization pack. This is the function that represents Compressed Sensing
    '''
    gObj = grads.gObj(x,N,data,k) # Calculate the obj function
    gTV = grads.gTV(x,N,strtag,dirWeight,dirs,nmins,M) # Calculate the TV gradient
    gXFM = grads.gXFM(x,N) # Calculate the wavelet gradient
    x.shape = (x.size,)
    return (gObj + lam1*gTV + lam2*gXFM) # Export the flattened array

def optfun(x,N,lam1,lam2,data,k,strtag,dirWeight = 0,dirs = None,M = None,nmins = 0,scaling_factor = 4,L = 2):
    '''
    This is the optimization function that we're trying to optimize. We are optimizing x here, and testing it within the funcitons that we want, as called by the functions that we've created
    '''
    data.shape = N
    x.shape = N
    obj_data = tf.ifft2c(data - k*tf.fft2c(x))
    obj = np.sum(obj_data*obj_data.conj()) #L2 Norm
    tv = np.sum(abs(tf.TV(x,N,strtag,dirWeight,dirs,nmins,M))) #L1 Norm
    xfm = np.sum(abs(tf.xfm(x,scaling_factor,L))) #L1 Norm
    return abs(obj + lam1*tv + lam2*xfm)

def phase_Calculation(data,is_kspace = 0,is_fftshifted = 0):
    
    if is_kspace:
        data = tf.ifft2c(data)
        if is_fftshifted:
            data = np.ifftshift(data)
        
    F = tf.matlab_style_gauss2D(shape=(5,5),sigma=2)
    filtdata = sp.ndimage.uniform_filter(data,size=5)
    return filtdata.conj()/(abs(filtdata)+EPS)

def recon_CS(filename = '/home/asalerno/Documents/pyDirectionCompSense/data/SheppLogan256.npy',#'DTI_Phantom-SNR1000.npy',
             strtag = ['spatial','spatial'],
             TVWeight = 0.01,
             XFMWeight = 0.01,
             dirWeight = 0,
             #DirType = 2,
             ItnLim = 150,
             epsilon = 0.02,
             l1smooth = 1e-15,
             xfmNorm = 1,
             scaling_factor = 4,
             L = 2,
             method = 'BFGS',
             dirFile = None,
             nmins = None):# = 4):
    np.random.seed(2000)
    im = np.load(filename)
    # im = im + 0.1*(np.random.normal(size=[256,256]) + 1j*np.random.normal(size = [256,256])) # For the simplest case right now


    N = np.array(im.shape) #image Size
    tupleN = tuple(N)
    pctg = 0.25 # undersampling factor
    P = 5 # Variable density polymonial degree
    #TVWeight = 0.01 # Weight for TV penalty
    #xfmWeight = 0.01 # Weight for Transform L1 penalty
    #Itnlim = 8 # Number of iterations
    
    pdf = samp.genPDF(N,P,pctg,radius = 0.1,cyl=[0])
    k = samp.genSampling(pdf,10,60)[0].astype(int)
    
    # Diffusion information that we need
    if dirFile:
        dirs = np.loadtxt(dirFile)
        M = d.calc_Mid_Matrix(dirs,nmins=4)
    else:
        dirs = None
        M = None
    
    data = np.fft.ifftshift(k)*tf.fft2c(im)
    #ph = phase_Calculation(im,is_kspace = False)
    #data = np.fft.ifftshift(np.fft.fftshift(data)*ph.conj());
    
    # "Data from the scanner"
    im_scan = tf.ifft2c(data)
    
    # Primary first guess. What we're using for now. Density corrected
    im_dc = tf.ifft2c(data/np.fft.ifftshift(pdf)) 
    
    # Optimization
    im_result = opt.minimize(optfun, im_dc.flatten(), args = (N,TVWeight,XFMWeight,data,k,strtag,dirWeight,dirs,M,nmins,scaling_factor,L),method=method,jac=derivative_fun,options={'maxiter':ItnLim,'gtol':epsilon})
    
    