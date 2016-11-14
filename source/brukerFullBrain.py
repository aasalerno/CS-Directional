# Imports
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
#plt.rcParams['image.interpolation'] = 'none'

import os.path
from sys import path as syspath
syspath.append('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/source/')
os.chdir(
    '/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/')  # Change this to the directory that you're saving the work in
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

    
strtag = ['spatial', 'spatial']
dirWeight = 0
# DirType = 2
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
radius = 0.1
a = 10.0

# Two-step parameters
xtol1 = 5e-3
TV1 = 0.005
XFM1 = 0.005

xtol2 = 1e-4
TV2 = 0.002
XFM2 = 0.002

xtol3 = 1e-5
TV3 = 0.001
XFM3 = 0.001
imFull = np.load('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/fullBrain.npy')
N = np.array(imFull.shape[0:2])  # image Size
#tupleN = tuple(N)
pctg = 0.25  # undersampling factor
P = 5  # Variable density polymonial degree
#ph = np.ones(im.shape, complex)
pdf = samp.genPDF(N, P, pctg, radius=radius, cyl=[0]) 
k = np.load('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/samplingScheme.npy')


im_recon = np.zeros(imFull.shape)

#slices = np.arange(imFull.shape[2])
slices = [190]

for sliceChoice in slices:
    print('Running slice ' + str(sliceChoice) + ' of ' + str(imFull.shape[-1]) + '...')
    # Initialization variables
    #inputdirectory="/hpf/largeprojects/MICe/segan/exercise_irradiation/bruker_data/running_C/P14/20160607_124310_Running_C_1_1"
    #petable = "/hpf/largeprojects/MICe/bjnieman/Bruker_cyltests/cylbruker_nTRfeath18_294_294"
    
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


    im = imFull[:,:,sliceChoice]# np.load('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/fullBrain.npy')[:,:,sliceChoice]
    ph = tf.matlab_style_gauss2D(im,shape=(5,5));

    # Generate the PDF for the sampling case -- note that this type is only used in non-directionally biased cases.
    #pdf = samp.genPDF(N, P, pctg, radius=radius, cyl=[0]) 
    # Set the sampling pattern -- checked and this gives the right percentage
    

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
    
    im_dc = tf.ifft2c(data / np.fft.ifftshift(pdf), ph=ph).real.flatten().copy()

    #----------------#
    #     Step 1     #
    #----------------#
    args = (N, TV1, XFM1, data, k, strtag, ph, dirWeight, dirs, M, nmins, wavelet, mode, a)
    im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                        options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 0, 'alpha_0': 0.1, 'c': 0.6, 'xtol': xtol1, 'TVWeight': TV1, 'XFMWeight': XFM1, 'N': N})
    im_dc = im_result['x'].reshape(N)

    #----------------#
    #     Step 2     #
    #----------------#
    args = (N, TV2, XFM2, data, k, strtag, ph, dirWeight, dirs, M, nmins, wavelet, mode, a)
    im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                        options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 0, 'alpha_0': 0.1, 'c': 0.6, 'xtol': xtol2, 'TVWeight': TV2, 'XFMWeight': XFM2, 'N': N})
    im_dc = im_result['x'].reshape(N)

    #----------------#
    #     Step 3     #
    #----------------#
    args = (N, TV3, XFM3, data, k, strtag, ph, dirWeight, dirs, M, nmins, wavelet, mode, a)
    im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                        options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 0, 'alpha_0': 0.1, 'c': 0.6, 'xtol': xtol3, 'TVWeight': TV3, 'XFMWeight': XFM3, 'N': N})
    im_res = im_result['x'].reshape(N)
    
    #np.save('hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/brainSlices/3step_25per_0.005_0.002_0.001_slice_' + str(sliceChoice) + '.npy',im_res)
    im_recon[:,:,sliceChoice] = im_res