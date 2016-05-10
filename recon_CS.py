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
import pyminc.volumes.factory
import numpy as np 
import scipy as sp
import sys
import rwt
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path
import transforms as tf
import scipy.ndimage.filters
#import sampling as samp

EPS = np.finfo(float).eps

def phase_Calculation(data,is_kspace = 0,is_fftshifted = 0):
    
    if is_kspace:
        data = tf.ifft2c(data)
        if is_fftshifted:
        data = np.ifftshift(data)
    
    #F = tf.matlab_style_gauss2D(shape=(5,5),sigma=2)
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
                    
    im = np.load(filename); # For the simplest case right now
    