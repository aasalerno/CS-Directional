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

sd = 534
np.random.seed(sd)
pctg = 0.25
sliceChoice = 127

# Initialization variables
inputdirectory="/hpf/largeprojects/MICe/asalerno/Bruker_cyltests/ASundersampled_2016-12-02/U_seed"+ str(sd) +"/"
petable = "brainData/seedTest/petable_" + str(int(pctg*100)) +"pctg_seed" + str(sd) + "_294_294.txt"
idFull = "/hpf/largeprojects/MICe/asalerno/Bruker_cyltests/ASundersampled_2016-12-02/fullacq"
peFull = "/hpf/largeprojects/MICe/bjnieman/Bruker_cyltests/cylbruker_nTRfeath18_294_294"
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
dirInfo = [None]*4
radius = 0.2
alpha_0 = 0.1
c = 0.6
a = 10.0 # value used for the tanh argument instead of sign

phIter = 0

# Multi-step parameters
xtol = [1e-2, 1e-3, 5e-4, 5e-4, 5e-4]
#TV = [0.01, 0.005, 0.002, 0.001, 0.0005]
TV = [0]*5
#XFM = [0.01, 0.005, 0.002, 0.001, 0.0005]
XFM = [0]*5
radius = 0.2


#full imdata
#fullImData = rff.getDataFromFID(peFull,idFull,3)[0,:,:,:]
#fullImData = fullImData/np.max(abs(fullImData))
#Nf = fullImData.shape
#fullImData = np.rollaxis(fullImData,axis=np.argmax(Nf))
#Nf = fullImData.shape
fullImData = np.load('BrukerPhantom/phantom_full_294_294.npy')
imf = fullImData[sliceChoice,:,:]


#read in the US data
#usImData = rff.getDataFromFID(petable,inputdirectory,3)[0,:,:,:]
#usImData = usImData/np.max(abs(usImData))
#N = usImData.shape
#usImData = np.rollaxis(usImData,axis=np.argmax(N))
#N = usImData.shape
usImData = np.load('BrukerPhantom/phantom_25per_seed' + str(sd) + '_294_294.npy')
im = np.ascontiguousarray(usImData[sliceChoice,:,:])
N = im.shape

#Now, FFT over the two PE Axes
fftData = np.fft.fft2(usImData,axes=(-2,-1))
P = 2
k = samp.readPEtable(petable)


# Generate the PDF for the sampling case -- note that this type is only used in non-directionally biased cases.
while True:
    try:
        pdf = samp.genPDF(N, P, pctg, radius=radius, cyl=np.hstack([1, N[-2:]]), style='mult')
        break
    except:
        radius = 0.5*radius
# Set the sampling pattern -- checked and this gives the right percentage

if len(N) == 2:
    N = np.hstack([1, N])
    k = k.reshape(N)

# Here is where we build the undersampled data
ph_ones = np.ones(im.shape, complex)
data = tf.fft2c(im, ph=ph_ones)
data_full = tf.fft2c(imf, ph=ph_ones)

# IMAGE from the "scanner data"
im_scan_wph = tf.ifft2c(data, ph=ph_ones)
ph_scan = np.exp(1j*tf.matlab_style_gauss2D(im_scan_wph,shape=(5,5)))
im_scan = tf.ifft2c(data, ph=ph_scan)

ph_full = np.exp(1j*tf.matlab_style_gauss2D(imf,shape=(5,5)))
im_full = tf.ifft2c(data_full, ph=ph_full)
#im_scan = abs(tf.ifft2c(data,ph_ones))
#data = tf.fft2c(im_scan,ph_ones).reshape(data.size).reshape(N)
#ph_scan = ph_ones

minval = np.min(abs(im))
maxval = np.max(abs(im))

# Primary first guess. What we're using for now. Density corrected
pdfDiv = pdf.copy()
pdfZeros = np.where(pdf<0.01)
pdfDiv[pdfZeros] = 1
im_dc = tf.ifft2c(data / np.fft.ifftshift(pdfDiv), ph=ph_scan).real.flatten().copy()
#im_dc = np.zeros(N).flatten()
#im_dc = abs(tf.ifft2c(data / np.fft.ifftshift(pdfDiv), ph=ph_scan).flatten().copy())

im_sp = im_dc.copy().reshape(N)
data = np.ascontiguousarray(data)


# Optimization algortihm -- this is where everything culminates together
for i in range(len(TV)):
    args = (N, TV[i], XFM[i], data, k, strtag, ph_scan, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
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

#fig = plt.figure()
#ax = fig.add_subplot(2,2,1)
#plt.imshow(abs(im),clim=(minval,maxval),interpolation='bilinear')
#plt.title('Original Image')
#ax = fig.add_subplot(2,2,2)
#plt.imshow(abs(im_scan),clim=(minval,maxval),interpolation='bilinear')
#plt.title('|im| from Scanner')
#ax = fig.add_subplot(2,2,3)
#plt.imshow(abs(im_sp),clim=(minval,maxval),interpolation='bilinear')
#plt.title('|im_dc| - Starting Point')
#ax = fig.add_subplot(2,2,4)
#plt.imshow(abs(im_res),clim=(minval,maxval),interpolation='bilinear')
#plt.title('|im_res| - Final Result - Full im Phase')
#plt.show()
#saveFig.save("brainData/P14/pctgComp/totComp_phUSCheck_5rounds_P_" + str(P) + '_pctg_' + str(pctg) + '_sl_' + str(sliceChoice))

#np.save("brainData/seedTest/" + str(int(pctg*100)) + "pctg_seed" + str(sd) + '.npy',im_res)
