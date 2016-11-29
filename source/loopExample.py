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
sliceChoices = [127, 150, 180, 190, 200, 210, 220]
pctgs = [0.125, 0.25, 0.33, 0.5, 0.75, 0.9]
#pctgs = [0.25]

for sliceChoice in sliceChoices:
    for pctg in pctgs:
        # Multi-step parameters
        xtol = [1e-2, 1e-3, 5e-4, 5e-4, 5e-4]
        TV = [0.01, 0.005, 0.002, 0.001, 0.0005]
        XFM = [0.01, 0.005, 0.002, 0.001, 0.0005]
        radius = 0.2

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
        #pctg = 0.25  # undersampling factor
        #ph = tf.matlab_style_gauss2D(im,shape=(5,5));
        P = 2

        # Generate the PDF for the sampling case -- note that this type is only used in non-directionally biased cases.
        while True:
            try:
                pdf = samp.genPDF(N, P, pctg, radius=radius, cyl=np.hstack([0, N]), style='mult')
                break
            except:
                radius = 0.5*radius
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
        #ph_scan = tf.matlab_style_gauss2D(im,shape=(5,5))

        #for i in range(phIter):
            #ph_scan = tf.laplacianUnwrap(ph_scan,N,[75,75])

        ph_scan = np.exp(1j*ph_scan)
        im_scan = tf.ifft2c(data, ph=ph_scan)
        #im_scan = abs(tf.ifft2c(data,ph_ones))
        #data = tf.fft2c(im_scan,ph_ones).reshape(data.size).reshape(N)
        #ph_scan = ph_ones

        minval = np.min(abs(im))
        maxval = np.max(abs(im))


        # Primary first guess. What we're using for now. Density corrected
        pdfDiv = pdf
        pdfZeros = np.where(pdf==0)
        pdfDiv[pdfZeros] = 1
        im_dc = tf.ifft2c(data / np.fft.ifftshift(pdfDiv), ph=ph_scan).real.flatten().copy()
        #im_dc = np.zeros(N)
        #im_dc = abs(tf.ifft2c(data / np.fft.ifftshift(pdfDiv), ph=ph_scan).flatten().copy())

        im_sp = im_dc.copy().reshape(N)


        # Optimization algortihm -- this is where everything culminates together
        for i in range(len(TV)):
            args = (N, TV[i], XFM[i], data, k, strtag, ph_scan, dirWeight, dirs, M, nmins, wavelet, mode, a)
            im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                                    options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': alpha_0, 'c': c, 'xtol': xtol[i], 'TVWeight': TV[i], 'XFMWeight': XFM[i], 'N': N})
            
            if np.any(np.isnan(im_result['x'])):
                print('Some nan''s found. Dropping TV and XFM values')
            else:
                im_dc = im_result['x'].reshape(N)
                alpha_k = im_result['alpha_k']
            

        im_res = im_dc

        #plt.imshow(abs(im_res),clim=(minval,maxval),interpolation='bilinear')
        #plt.title(str(int(pctg*100)) + "% Data -- P=" + str(P) + " -- " + str(len(TV)) + " Rounds")
        #xlist = ','.join(map(str, TV)) 
        #plt.xlabel('TV=XFM= [' + xlist + ']')

        fig = plt.figure()
        ax = fig.add_subplot(2,2,1)
        plt.imshow(abs(im),clim=(minval,maxval),interpolation='bilinear')
        plt.title('Original Image')
        ax = fig.add_subplot(2,2,2)
        plt.imshow(abs(im_scan),clim=(minval,maxval),interpolation='bilinear')
        plt.title('|im| from Scanner')
        ax = fig.add_subplot(2,2,3)
        plt.imshow(abs(im_sp),clim=(minval,maxval),interpolation='bilinear')
        plt.title('|im_dc| - Starting Point')
        ax = fig.add_subplot(2,2,4)
        plt.imshow(abs(im_res),clim=(minval,maxval),interpolation='bilinear')
        plt.title('|im_res| - Final Result - Full im Phase')
        #plt.show()
        saveFig.save("brainData/P14/pctgComp/totComp_phUSCheck_5rounds_P_" + str(P) + '_pctg_' + str(pctg) + '_sl_' + str(sliceChoice))

    #np.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/P_analysis/3rounds_P_" + str(P) + '.npy',im_res)