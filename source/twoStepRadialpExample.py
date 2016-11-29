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

np.random.seed(50)
seeds = np.hstack([1000, np.round(np.random.random(10)*1000)])

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
radius = 0.1
sliceChoice = 190

# Two-step parameters
xtol1 = 1e-3
TV1 = 0.005
XFM1 = 0.005

xtol2 = 1e-4
TV2 = 0.002
XFM2 = 0.002

xtol3 = 5e-5
TV3 = 0.001
XFM3 = 0.001

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
#ph = tf.matlab_style_gauss2D(im,shape=(5,5));
ph = np.ones(im.shape, complex)
Ps = np.arange(0,5.1,.5)

seeds = np.array([1000])

for sdcnt in range(len(seeds)):
    np.random.seed(int(seeds[sdcnt]))
    for P in Ps:
        # Generate the PDF for the sampling case -- note that this type is only used in non-directionally biased cases.
        pdf = samp.genPDF(N, P, pctg, radius=radius, cyl=[0], style='mult') 
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
        #ph = tf.matlab_style_gauss2D(im_scan,shape=(5,5))
        #im_scan = tf.ifft2c(data, ph=ph)
        minval = np.min(im)
        maxval = np.max(im)
        # Primary first guess. What we're using for now. Density corrected
        #im_dc = tf.ifft2c(data / np.fft.ifftshift(pdf), ph=ph).real.flatten().copy()
        #for imdcs in ['zeros','ones','densCorr','imFull']:
        im_dc = tf.ifft2c(data / np.fft.ifftshift(pdf), ph=ph).real.flatten().copy()


        # Optimization algortihm -- this is where everything culminates together
        a = 10.0

        #----------------#
        #     Step 1     #
        #----------------#
        args = (N, TV1, XFM1, data, k, strtag, ph, dirWeight, dirs, M, nmins, wavelet, mode, a)
        im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                                options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': 0.1, 'c': 0.6, 'xtol': xtol1, 'TVWeight': TV1, 'XFMWeight': XFM1, 'N': N})
        im_dc = im_result['x'].reshape(N)

        #----------------#
        #     Step 2     #
        #----------------#
        args = (N, TV2, XFM2, data, k, strtag, ph, dirWeight, dirs, M, nmins, wavelet, mode, a)
        im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                                options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': 0.1, 'c': 0.6, 'xtol': xtol2, 'TVWeight': TV2, 'XFMWeight': XFM2, 'N': N})
        im_dc = im_result['x'].reshape(N)
        
        #----------------#
        #     Step 3     #
        #----------------#
        args = (N, TV3, XFM3, data, k, strtag, ph, dirWeight, dirs, M, nmins, wavelet, mode, a)
        im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                                options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': 0.1, 'c': 0.6, 'xtol': xtol3, 'TVWeight': TV3, 'XFMWeight': XFM3, 'N': N})
        im_res = im_result['x'].reshape(N)
        
        plt.imshow(abs(im_res),clim=(minval,maxval),interpolation='bilinear')
        plt.title("25% Data -- P=" + str(P) + " -- 3 Rounds")
        plt.xlabel('TV1=XFM1=' + str(TV1) + '   TV2=XFM2=' + str(TV2) + '   TV3=XFM3=' + str(TV3))
        saveFig.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/P_analysis/1_overr_p/seed" + str(int(seeds[sdcnt])) + '/3rounds_P_' + str(P) + '_ph_1')
        
        np.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/P_analysis/1_overr_p/seed" + str(int(seeds[sdcnt])) + "/3rounds_P_" + str(P) + '_ph_1.npy',im_res)