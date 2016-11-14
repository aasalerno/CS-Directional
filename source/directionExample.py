# Imports
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'

import os.path
from sys import path as syspath
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/")
os.chdir(
    '/home/asalerno/Documents/pyDirectionCompSense/')  # Change this to the directory that you're saving the work in
import transforms as tf
import scipy.ndimage.filters
import grads
import sampling as samp
import direction as d
# from scipy import optimize as opt
import optimize as opt
import scipy.optimize as spopt
from recon_CS import *
import read_from_fid as rff
import saveFig

# Initialization variables
filename = '/home/asalerno/Documents/pyDirectionCompSense/directionData/singleSlice_30dirs.npy'
strtag = ['spatial', 'spatial']
TVWeight = 0.002
XFMWeight = 0.002
dirWeight = 0
# DirType = 2
ItnLim = 150
epsilon = 1e-6
l1smooth = 1e-15
xfmNorm = 1
wavelet = 'db4'
mode = 'per'
method = 'CG'
dirFile = '/home/asalerno/Documents/pyDirectionCompSense/GradientVectorMag.txt'
nmins = 5
dirs = np.loadtxt(dirFile)
M, dIM, Ause = d.calc_Mid_Matrix(dirs,nmins)
pctg = 0.25
radius = 0.1

im = np.load(filename)
ph = np.zeros(im.shape)*1j

for i in range(len(dirs)):
    ph[i,:,:] = tf.matlab_style_gauss2D(im[i,:,:],shape=(2,2))
    
# Generate the sampling or pull it from file
#k = samp.genSamplingDir(img_sz = [180,180], dirFile = dirFile, pctg = pctg, cyl = [0],                         radius = radius, nmins = nmins, engfile = None)
k = np.load('/home/asalerno/Documents/pyDirectionCompSense/directionData/30dirSampling_5mins.npy')

data = np.zeros(im.shape)
im_scan = np.zeros(im.shape)
im_dc = np.zeros(im.shape)

for i in range(len(dirs)):
    data[i,:,:] = np.fft.ifftshift(k[i,:,:]) * tf.fft2c(im[i,:,:], ph=ph[i,:,:])
    # ph = phase_Calculation(im,is_kspace = False)
    # data = np.fft.ifftshift(np.fft.fftshift(data)*ph.conj());


    # IMAGE from the "scanner data"
    im_scan[i,:,:] = tf.ifft2c(data[i,:,:], ph=ph[i,:,:])
    
    

    
