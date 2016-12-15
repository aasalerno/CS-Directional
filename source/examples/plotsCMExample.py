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

np.random.seed(534)

inputdirectory="/hpf/largeprojects/MICe/segan/exercise_irradiation/bruker_data/running_C/P14/20160607_124310_Running_C_1_1"
petable = "/hpf/largeprojects/MICe/bjnieman/Bruker_cyltests/cylbruker_nTRfeath18_294_294"
strtag = ['spatial', 'spatial']
dirWeight = 0
ItnLim = 30
lineSearchItnLim = 30
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
#sliceChoices = [127, 150, 180, 190, 200, 210, 220]
sliceChoice = 334-190
#pctgs = [0.125, 0.25, 0.33, 0.5, 0.75, 0.9]
pctgs = [0.25]#,0.33,0.5] 
xtol = [1e-2, 1e-3, 5e-4, 5e-4, 5e-4]
TV = [0.01, 0.005, 0.002, 0.001]
XFM = [0.01, 0.005, 0.002, 0.001]
radius = 0.2

im = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/P14/data/fullySampledBrain.npy')#[sliceChoice-1,:,:]
N = np.array(im.shape)  # image Size
#tupleN = tuple(N)
#pctg = 0.25  # undersampling factor
#ph = tf.matlab_style_gauss2D(im,shape=(5,5));
P = 2

for pctg in pctgs:
    print(pctg)
    # Generate the PDF for the sampling case -- note that this type is only used in non-directionally biased cases.
    while True:
        try:
            print(radius)
            pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=np.hstack([1, N[-2:]]), style='mult')
            break
        except:
            radius = 0.5*radius
    # Set the sampling pattern -- checked and this gives the right percentage
    k = samp.genSampling(pdf, 50, 2)[0].astype(int)
    if len(N) == 2:
        N = np.hstack([1, N])
        k = k.reshape(N)
    elif len(N) == 3:
        k = k.reshape(np.hstack([1,N[-2:]])).repeat(N[0],0)

    # Here is where we build the undersampled data
    ph_ones = np.ones(N[-2:], complex)
    ph_scan = np.zeros(N, complex)
    data = np.zeros(N,complex)
    im_dc = np.zeros(N,complex)
    im_scan = np.zeros(N,complex)
    print('Looping through data')
    for i in range(N[0]):
        print(i)
        data[i,:,:] = np.fft.ifftshift(k[i,:,:]) * tf.fft2c(im[i,:,:], ph=ph_ones)

        # IMAGE from the "scanner data"
        im_scan_wph = tf.ifft2c(data[i,:,:], ph=ph_ones)
        ph_scan[i,:,:] = tf.matlab_style_gauss2D(im_scan_wph,shape=(5,5))
        ph_scan[i,:,:] = np.exp(1j*ph_scan[i,:,:])
        im_scan[i,:,:] = tf.ifft2c(data[i,:,:], ph=ph_scan[i,:,:])
        #im_lr = samp.loRes(im,pctg)

        pdfDiv = pdf
        pdfZeros = np.where(pdf==0)
        pdfDiv[pdfZeros] = 1
        im_dc[i,:,:] = tf.ifft2c(data[i,:,:] / np.fft.ifftshift(pdfDiv), ph=ph_scan[i,:,:]).real.copy()
    
    im_dc = im_dc.flatten()
    minval = np.min(abs(im))
    maxval = np.max(abs(im))
    
    im_sp = im_dc.copy().reshape(N)
    data = np.ascontiguousarray(data)
    
    print('Starting the loop')
    for i in range(len(TV)):
        args = (N, TV[i], XFM[i], data, k, strtag, ph_scan, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
        im_result = opt.minimize(optfun, im_dc, args=args, method=method,
                                jac=derivative_fun, 
                                options={'maxiter': ItnLim, 'lineSearchItnLim': lineSearchItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': alpha_0, 'c': c, 'xtol': xtol[i], 'TVWeight': TV[i], 'XFMWeight': XFM[i], 'N': N})
        
        if np.any(np.isnan(im_result['x'])):
            print('Some nan''s found. Dropping TV and XFM values')
        elif im_result['status'] != 0:
            print('TV and XFM values too high -- no solution found. Dropping...')
        else:
            im_dc = im_result['x'].reshape(N)
            #alpha_k = im_result['alpha_k']
        

    im_res = im_dc
    
    #plt.imshow(im_sp.real.reshape(N[-2:]),clim=(minval,maxval))
    #plt.title('Starting Point -- ' + str(int(pctg*100)) + '%')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/CMExamples/sl190' + str(int(pctg*100)) + 'p_SP')
    #plt.imshow(im_res.real.reshape(N[-2:]),clim=(minval,maxval))
    #plt.title('Result -- ' + str(int(pctg*100)) + '%')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/CMExamples/sl190' + str(int(pctg*100)) + 'p_Result')
    #plt.imshow(im_lr.real.reshape(N[-2:]),clim=(minval,maxval))
    #plt.title('Lo Res -- ' + str(int(pctg*100)) + '%')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/CMExamples/sl190' + str(int(pctg*100)) + 'p_LR')
    
    
#plt.imshow(im_full.real.reshape(N[-2:]),clim=(minval,maxval))
#plt.title('Fully Sampled')
#saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/CMExamples/FullySampled')