# Imports
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

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
filename = '/hpf/largeprojects/MICe/asalerno/DTIdata/26apr16.fid/kspace.npy'
strtag = ['diff','spatial', 'spatial']
TVWeight = 0.002
XFMWeight = 0.002
dirWeight = 0
# DirType = 2
ItnLim = 50
lineSearchItnLim = 30
epsilon = 1e-6
l1smooth = 1e-15
xfmNorm = 1
wavelet = 'db4'
mode = 'per'
method = 'CG'
dirFile = '/home/asalerno/Documents/pyDirectionCompSense/GradientVectorMag.txt'
nmins = 5
dirs = np.loadtxt(dirFile)
M, dIM, Ause, inds = d.calc_Mid_Matrix(dirs,nmins)
pctg = 0.25
radius = 0.15

data = np.load(filename)

dirChoice = 31
dataFFT = np.fft.fft(data,axis=(3))

im = np.load(filename)
im = im/np.max(abs(im))
N = im.shape
ph_scan = np.zeros(N,dtype=complex)
ph_ones = np.ones(N[-2:],dtype=complex)

    
# Generate the sampling or pull it from file
#k = samp.genSamplingDir(img_sz=[180,180], dirFile=dirFile, pctg=pctg, cyl=[1],radius=radius, nmins=nmins, engfile='/micehome/asalerno/Documents/pyDirectionCompSense/engFile30dir.npy', endSize=[256,256])
k = np.load('/home/asalerno/Documents/pyDirectionCompSense/directionData/30dirSampling_5mins.npy')

data = np.zeros(im.shape,dtype=complex)
im_scan = np.zeros(im.shape,dtype=complex)
im_dc = np.zeros(im.shape,dtype=complex)

minval = np.min(abs(im))
maxval = np.max(abs(im))

for i in range(len(dirs)):
    data[i,:,:] = np.fft.ifftshift(k[i,:,:]) * tf.fft2c(im[i,:,:], ph=ph_ones)
    
    # IMAGE from the "scanner data"
    im_scan_wph = tf.ifft2c(data[i,:,:], ph=ph_ones)
    ph_scan[i,:,:] = tf.matlab_style_gauss2D(im_scan_wph,shape=(5,5))
    ph_scan[i,:,:] = np.exp(1j*ph_scan[i,:,:])
    im_scan[i,:,:] = tf.ifft2c(data[i,:,:], ph=ph_scan[i,:,:])
    
data_dc = d.dir_dataSharing(k,data,dirs,[180,180],maxCheck=5,bymax=1)

for i in range(len(dirs)):
    im_dc[i,:,:] =  tf.ifft2c(data_dc[i,:,:], ph=ph_scan[i,:,:])