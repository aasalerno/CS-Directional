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

EPS = np.finfo(float).eps

def phase_Calculation(data,is_kspace = 0,is_fftshifted = 0):
    
    if is_kspace:
        data = tf.ifft2c(data)
        if is_fftshifted:
            data = np.ifftshift(data)
        
    F = tf.matlab_style_gauss2D(shape=(5,5),sigma=2)
    filtdata = sp.ndimage.uniform_filter(data,size=5)
    return filtdata.conj()/(abs(filtdata)+EPS)

def recon_CS(filename = '/home/asalerno/Documents/pyDirectionCompSense/data/SheppLogan256.npy',
             TVWeight = 0.01,
             XFMWeight = 0.01,
             TVPixWeight = 1,
             DirWeight = 0,
             DirType = 2,
             ItnLim = 150,
             epsilon = 0.02,
             l1smooth = 1e-15,
             xfmNorm = 1):
    
    np.random.seed(2000)
    
    im = np.load(filename)
    im = im + 0.1*(np.random.normal(size=[256,256]) + 1j*np.random.normal(size = [256,256])) # For the simplest case right now


    N = np.array([256,256]) #image Size
    pctg = 0.25 # undersampling factor
    P = 5 # Variable density polymonial degree
    TVWeight = 0.01 # Weight for TV penalty
    xfmWeight = 0.01 # Weight for Transform L1 penalty
    Itnlim = 8 # Number of iterations
    
    
    pdf = samp.genPDF(N,P,pctg,radius = 0.1,cyl=[0])
    k = samp.genSampling(pdf,10,60)[0]
    
    data = np.fft.ifftshift(k)*tf.fft2c(im)
    #ph = phase_Calculation(im,is_kspace = False)
    #data = np.fft.ifftshift(np.fft.fftshift(data)*ph.conj());
    
    # "Data from the scanner"
    im = tf.ifft2c(data)
    
    # Primary first guess. What we're using for now.
    im_dc = tf.ifft2c(data/np.fft.ifftshift(pdf)) 
    
    # Grads
    gObj = grads.gObj(im,data,k)
    gTV = grads.gTV(im)
    gXFM = grads.gXFM(im)
    
    
    for i in xrange(8):
        
    
    