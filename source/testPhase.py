# Imports
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
#plt.rcParams['image.interpolation'] = 'none'

import os.path
from sys import path as syspath
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/")
os.chdir('/home/asalerno/Documents/pyDirectionCompSense/')  # Change this to the directory that you're saving the work in
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

np.random.seed(250)

# Initialization variables
petable = "/hpf/largeprojects/MICe/bjnieman/Bruker_cyltests/cylbruker_nTRfeath18_294_294"
strtag = ['spatial', 'spatial']
TVWeight = 0.005
XFMWeight = 0.005
dirWeight = 0
# DirType = 2
ItnLim = 150
epsilon = 1e-6
l1smooth = 1e-15
alpha_0 = 0.1
xtol = 1e-4

xfmNorm = 1
wavelet = 'db4'
mode = 'per'
method = 'CG'
dirFile = None
nmins = None
dirs = None
M = None
radius = 0.2


inputdirectory = "/hpf/largeprojects/MICe/segan/exercise_irradiation/bruker_data/running_C/P14/20160607_124310_Running_C_1_1"

sliceChoice = [100,125,150]
# Make the data go from clim=[0,1]
fullImData = rff.getDataFromFID(petable,inputdirectory,2)[0,:,:,:]
fullImData = fullImData/np.max(abs(fullImData))
N = fullImData.shape
im = np.zeros([334, 294, 294])
ph = np.zeros([334, 294, 294])

for i in range(N[2]):
    im[i,:,:] = fullImData[:,:,i]
    #plt.imshow(im.real)
    #plt.colorbar()
    #plt.title('Im_real for P14 Coil 3')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/brainData/phaseCheck/coil3/C-P14/real_sl' + str(sliceChoice[i]))
    #plt.figure()
    ph[i,:,:] = np.arctan2(im.imag[i,:,:],im.real[i,:,:])
    #plt.colorbar()
    #plt.title('Phase for P14 Coil 3')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/brainData/phaseCheck/coil3/C-P14/ph_sl' + str(sliceChoice[i]))
    #plt.imshow(abs(im))
    #plt.colorbar()
    #plt.title('abs(im) for P43 Coil 3')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/brainData/phaseCheck/coil3/C-P14/abs_sl' + str(sliceChoice[i]))

#N = fullImData.shape
#Nsort = np.argsort(N)