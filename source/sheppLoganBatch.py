# Imports
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os.path
from sys import path as syspath
syspath.append("/home/asalerno/pyDirectionCompSense/source/")

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
import saveFig

plt.rcParams['image.cmap'] = 'gray'
from recon_CS import *

# Initialization variables
filename = '/home/asalerno/Documents/pyDirectionCompSense/data/SheppLogan256.npy'
strtag = ['spatial', 'spatial']
TVWeight = 0.01
XFMWeight = 0.01
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

np.random.seed(2000)

# im = np.zeros([8,8]);
# im[3:5,3:5] = 1;

im = np.load(filename)

N = np.array(im.shape)  # image Size
#tupleN = tuple(N)

for pctg in [0.25, 0.33, 0.40, 0.50]:
    for TVWeight in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
        for XFMWeight in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
            P = 5  # Variable density polymonial degree
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
            for imdcs in ['zeros','ones','densCorr','imFull']:
                if imdcs == 'zeros':
                    im_dc = np.zeros(data.shape)
                elif imdcs == 'ones':
                    im_dc = np.ones(data.shape)
                elif imdcs == 'densCorr':
                    im_dc = tf.ifft2c(data / np.fft.ifftshift(pdf), ph=ph).real.flatten().copy()
                elif imdcs == 'imFull':
                    im_dc = im
                
                # Optimization algortihm -- this is where everything culminates together
                a = 10.0
                args = (N, TVWeight, XFMWeight, data, k, strtag, ph, dirWeight, dirs, M, nmins, wavelet, mode, a)
                im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                                        options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': 0.1, 'c': 0.6, 'xtol': 5e-3, 'TVWeight': TVWeight, 'XFMWeight': XFMWeight, 'N': N})
                im_res = im_result['x'].reshape(N)
                plt.imshow(abs(im_res),clim=(minval,maxval))
                plt.title("Reconstructed Image with %.2f%% Sampling and im_dc==" % pctg + imdcs)
                saveFig.save("sheppLoganData/TV%s_XFM%s/%.2fper_result_im_dc_" % (float('%.1g' % TVWeight), float('%.1g' % XFMWeight) , pctg) + imdcs)
                np.save("sheppLoganData/TV%s_XFM%s/%.2fper_result_im_dc_" % (float('%.1g' % TVWeight), float('%.1g' % XFMWeight) , pctg) + imdcs + ".npy" ,im_res)

    lrLoc = int(np.ceil((N[0]-np.ceil(N[0]/np.sqrt(1/pctg)))/2))
    im_lr = tf.fft2c(np.fft.fftshift(np.fft.fftshift(tf.ifft2c(im,np.ones(im.shape)))[lrLoc:-lrLoc,lrLoc:-lrLoc]),np.ones(im[lrLoc:-lrLoc,lrLoc:-lrLoc].shape))
    plt.imshow(abs(im_lr))
    plt.title('Low Resolution Equivalent with only %.2f%% of k-space Sampled' % pctg)
    saveFig.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/TV%.2f_XFM%.2f/%.2fper_result_im_lr" % (TVWeight, XFMWeight, pctg))
    np.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/TV%.2f_XFM%.2f/%.2fper_result_im_lr" % (TVWeight, XFMWeight, pctg) + ".npy", im_lr)