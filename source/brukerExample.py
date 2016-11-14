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

np.random.seed(1000)

# Initialization variables
inputdirectory="/hpf/largeprojects/MICe/segan/exercise_irradiation/bruker_data/running_C/P14/20160607_124310_Running_C_1_1"
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
radius = 0.1


# Make the data go from clim=[0,1]
#fullImData = rff.getDataFromFID(petable,inputdirectory,2)[0,:,:,:]
#fullImData = fullImData-np.min(fullImData)
#fullImData = fullImData/np.max(fullImData)
#im = fullImData[:,:,sliceChoice]
#N = fullImData.shape
#Nsort = np.argsort(N)


# Now, FFT over the two PE Axes
#fftData = np.fft.fft2(fullImData,axes=(Nsort[0],Nsort[1]))
#data_full = np.fft.fftshift(fftData[:,:,sliceChoice])


im = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/fullySampledBrain.npy')
N = np.array(im.shape)  # image Size
#tupleN = tuple(N)
pctg = 0.25  # undersampling factor
P = 2.05  # Variable density polymonial degree
ph = tf.matlab_style_gauss2D(im,shape=(5,5));
#ph = np.ones(im.shape, complex)

# Generate the PDF for the sampling case -- note that this type is only used in non-directionally biased cases.
pdf = samp.genPDF(N, P, pctg, radius=radius, cyl=[0]) 
# Set the sampling pattern -- checked and this gives the right percentage
k = samp.genSampling(pdf, 50, 2)[0].astype(int)

# Here is where we build the undersampled data
data = np.fft.ifftshift(k) * tf.fft2c(im, ph=ph)
# ph = phase_Calculation(im,is_kspace = False)
# data = np.fft.ifftshift(np.fft.fftshift(data)*ph.conj());
#filt = tf.fermifilt(N)
#data = data * filt

# IMAGE from the "scanner data"
im_scan = tf.ifft2c(data, ph=ph)
minval = np.min(im)
maxval = np.max(im)
# Primary first guess. What we're using for now. Density corrected
#im_dc = tf.ifft2c(data / np.fft.ifftshift(pdf), ph=ph).real.flatten().copy()
#for imdcs in ['zeros','ones','densCorr','imFull']:
for imdcs in ['densCorr','densCorr_Completed']:
    if imdcs == 'zeros':
        im_dc = np.zeros(data.shape)
    elif imdcs == 'ones':
        im_dc = np.ones(data.shape)
    elif imdcs == 'densCorr':
        im_dc = tf.ifft2c(data / np.fft.ifftshift(pdf), ph=ph).real.flatten().copy()
    elif imdcs == 'imFull':
        im_dc = im
    elif imdcs == 'densCorr_Completed':
        #im_dc = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/TV0.005_XFM0.005/0.25per_result_im_dc_densCorr.npy')
        im_dc = im_res
        TVWeight = 0.001
        XFMWeight = 0.001
        xtol = 1e-4

    # Optimization algortihm -- this is where everything culminates together
    a = 10.0
    args = (N, TVWeight, XFMWeight, data, k, strtag, ph, dirWeight, dirs, M, nmins, wavelet, mode, a)
    im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun, options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': alpha_0, 'c': 0.6, 'xtol': xtol, 'TVWeight': TVWeight, 'XFMWeight': XFMWeight, 'N': N})
    im_res = im_result['x'].reshape(N)
    
    plt.imshow(im_res,clim=(minval,maxval),interpolation='bilinear')
    plt.title("Reconstructed Image with %d%% Sampling and im_dc==" % (pctg*100) + imdcs)
    plt.show()
    
    
    if imdcs == 'densCorr':
        saveFig.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/replicate/seed1000_%.2fper_firstRun_im_dc_" % (pctg*100) + imdcs)
        np.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/replicate/seed1000_%.2fper_firstRun_im_dc_" % (pctg*100) + imdcs,im_res)
    elif imdcs == 'densCorr_Completed':
        saveFig.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/replicate/seed1000_%.2fper_secondRun_im_dc_TVXFM_0.001" % (pctg*100) + imdcs)
        np.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/replicate/seed1000_%.2fper_secondRun_im_dc_TVXFM_0.001" % (pctg*100) + imdcs,im_res)

#lrLoc = int(np.ceil((294-np.ceil(294/np.sqrt(1/pctg)))/2))
#im_lr = tf.fft2c(np.fft.fftshift(np.fft.fftshift(tf.ifft2c(im,np.ones(im.shape)))[lrLoc:-lrLoc,lrLoc:-lrLoc]),np.ones(im[lrLoc:-lrLoc,lrLoc:-lrLoc].shape))
#plt.imshow(abs(im_lr))
#plt.title('Low Resolution Equivalent with only %.2f%% of k-space Sampled' % pctg)
#saveFig.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/TV%.2f_XFM%.2f/%.2fper_result_im_lr" % (TVWeight, XFMWeight, pctg))
