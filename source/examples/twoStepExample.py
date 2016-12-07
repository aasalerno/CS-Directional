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

np.random.seed(124)

# Initialization variables
inputdirectory="/hpf/largeprojects/MICe/segan/exercise_irradiation/bruker_data/running_C/P14/20160607_124310_Running_C_1_1"
petable = "/hpf/largeprojects/MICe/bjnieman/Bruker_cyltests/cylbruker_nTRfeath18_294_294"
strtag = ['spatial', 'spatial']
dirWeight = 0
ItnLim = 150
epsilon = 1e-6
l1smooth = 1e-15
xfmNorm = 1
wavelet = 'db4'
mode = 'per'
method = 'CG'
dirFile = None
nmins = None
dirs = None
M = None
radius = 0.2
alpha_0 = 0.1
c = 0.6
a = 10.0 # value used for the tanh argument instead of sign

res = 75 # need to find where I can get this from the data itself...
phIter = 0
sliceChoice = 127

# Multi-step parameters
xtol1 = 1e-2
TV1 = 0.01
XFM1 = 0.01

xtol2 = 1e-3
TV2 = 0.005
XFM2 = 0.005

xtol3 = 5e-4
TV3 = 0.001
XFM3 = 0.001

xtol4 = 5e-4
TV4 = 0.002
XFM4 = 0.002

xtol5 = 1e-4
TV5 = 0.001
XFM5 = 0.001

# Make the data go from clim=[0,1]
#fullImData = rff.getDataFromFID(petable,inputdirectory,2)[0,:,:,:]
#fullImData = fullImData/np.max(abs(fullImData))
#im = fullImData[:,:,sliceChoice]
#N = fullImData.shape
#Nsort = np.argsort(N)


# Now, FFT over the two PE Axes
#fftData = np.fft.fft2(fullImData,axes=(Nsort[0],Nsort[1]))
#data_full = np.fft.fftshift(fftData[:,:,sliceChoice])


im = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/P14/data/fullySampledBrain.npy')[sliceChoice-1,:,:]
N = np.array(im.shape)  # image Size
#tupleN = tuple(N)
pctg = 0.25  # undersampling factor
#ph = tf.matlab_style_gauss2D(im,shape=(5,5));
P = 2

# Generate the PDF for the sampling case -- note that this type is only used in non-directionally biased cases.
pdf = samp.genPDF(N, P, pctg, radius=radius, cyl=np.hstack([0, N]), style='mult') 
# Set the sampling pattern -- checked and this gives the right percentage
k = samp.genSampling(pdf, 50, 2)[0].astype(int)

# Here is where we build the undersampled data
ph_ones = np.ones(im.shape, complex)
data = np.fft.ifftshift(k) * tf.fft2c(im, ph=ph_ones)
# data = np.fft.ifftshift(np.fft.fftshift(data)*ph.conj());
#filt = tf.fermifilt(N)
#data = data * filt

# IMAGE from the "scanner data"
#ph_ones = np.ones(im.shape, complex)
im_scan_wph = tf.ifft2c(data, ph=ph_ones)
ph_scan = tf.matlab_style_gauss2D(im_scan_wph,shape=(5,5))

for i in range(phIter):
    ph_scan = tf.laplacianUnwrap(ph_scan,N,[75,75])

ph_scan = np.exp(1j*ph_scan)
im_scan = tf.ifft2c(data, ph=ph_scan)

minval = np.min(abs(im))
maxval = np.max(abs(im))


# Primary first guess. What we're using for now. Density corrected
pdfDiv = pdf
pdfZeros = np.where(pdf==0)
pdfDiv[pdfZeros] = 1
im_dc = tf.ifft2c(data / np.fft.ifftshift(pdfDiv), ph=ph_scan).real.flatten().copy()
#im_dc = np.zeros(N)

# Optimization algortihm -- this is where everything culminates together
#----------------#
#     Step 1     #
#----------------#
args = (N, TV1, XFM1, data, k, strtag, ph_scan, dirWeight, dirs, M, nmins, wavelet, mode, a)
im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                        options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': alpha_0, 'c': c, 'xtol': xtol1, 'TVWeight': TV1, 'XFMWeight': XFM1, 'N': N})
im_dc = im_result['x'].reshape(N)
alpha_k = im_result['alpha_k']

#----------------#
#     Step 2     #
#----------------#
args = (N, TV2, XFM2, data, k, strtag, ph_scan, dirWeight, dirs, M, nmins, wavelet, mode, a)
im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                        options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': c*alpha_k, 'c': c, 'xtol': xtol2, 'TVWeight': TV2, 'XFMWeight': XFM2, 'N': N})
im_dc = im_result['x'].reshape(N)
alpha_k = im_result['alpha_k']

#----------------#
#     Step 3     #
#----------------#
args = (N, TV3, XFM3, data, k, strtag, ph_scan, dirWeight, dirs, M, nmins, wavelet, mode, a)
im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                        options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': c*alpha_k, 'c': c, 'xtol': xtol3, 'TVWeight': TV3, 'XFMWeight': XFM3, 'N': N})
im_dc = im_result['x'].reshape(N)
alpha_k = im_result['alpha_k']

#----------------#
#     Step 4     #
#----------------#
args = (N, TV4, XFM4, data, k, strtag, ph_scan, dirWeight, dirs, M, nmins, wavelet, mode, a)
im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                        options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': c*alpha_k, 'c': c, 'xtol': xtol4, 'TVWeight': TV4, 'XFMWeight': XFM4, 'N': N})
im_dc = im_result['x'].reshape(N)
alpha_k = im_result['alpha_k']

#----------------#
#     Step 5     #
#----------------#
args = (N, TV5, XFM5, data, k, strtag, ph_scan, dirWeight, dirs, M, nmins, wavelet, mode, a)
im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                        options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': c*alpha_k, 'c': c, 'xtol': xtol5, 'TVWeight': TV5, 'XFMWeight': XFM5, 'N': N})
im_res = im_result['x'].reshape(N)

plt.imshow(abs(im_res),clim=(minval,maxval),interpolation='bilinear')
plt.title("25% Data -- P=" + str(P) + " -- 5 Rounds")
plt.xlabel('TV=XFM= [' + str(TV1) + ', ' +  str(TV2) + ', ' + str(TV3) + ', ' + str(TV4) + ', ' + str(TV5) + ']')
#plt.show()
saveFig.save("brainData/P14/5rounds_P_" + str(P) + '_pctg_' + str(pctg) + '_sl_' + str(sliceChoice))

#np.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/P_analysis/3rounds_P_" + str(P) + '.npy',im_res)