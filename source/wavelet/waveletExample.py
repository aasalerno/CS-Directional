# Imports
from __future__ import division
import numpy as np
import scipy as sp
import os.path
import matplotlib.pyplot as plt
from sys import path as syspath
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/")
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/")
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/wavelet")
os.chdir('/home/asalerno/Documents/pyDirectionCompSense/')
import transforms as tf
import scipy.ndimage.filters
import gradWavelet as grads
import sampling as samp
import direction as d
import optimize as opt
import scipy.optimize as spopt
import wavelet_DC_TV_XFM_f_df as funs
import read_from_fid as rff
import saveFig

f = funs.objectiveFunction
df = funs.derivativeFunction
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(534)

inputdirectory="/hpf/largeprojects/MICe/segan/exercise_irradiation/bruker_data/running_C/P14/20160607_124310_Running_C_1_1"
petable = "/hpf/largeprojects/MICe/bjnieman/Bruker_cyltests/cylbruker_nTRfeath18_294_294"

strtag = ['','spatial', 'spatial']
dirWeight = 0
ItnLim = 30
lineSearchItnLim = 30
wavelet = 'db4'
mode = 'per'
method = 'CG'
kern = np.array([[[ 0.,  0.,  0.],
                  [ 0.,  0.,  0.],
                  [ 0.,  0.,  0.]],
                  
                  [[ 0.,  0.,  0.],
                  [ 0., -1.,  0.],
                  [ 0.,  1.,  0.]],
                  
                  [[ 0.,  0.,  0.],
                  [ 0., -1.,  1.],
                  [ 0.,  0.,  0.]]])

dirFile = None
nmins = None
dirs = None
M = None
dirInfo = [None]*4
radius = 0.2
pft=False
alpha_0 = 0.1
c = 0.6
a = 10.0 # value used for the tanh argument instead of sign

phIter = 0
sliceChoice = 150
pctg = 0.25 
xtol = [1e-2, 1e-3, 5e-4, 5e-4]
TV = [0.01, 0.005, 0.002, 0.001]
XFM = [0.01,.005, 0.002, 0.001]
radius = 0.2

im = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/P14/data/fullySampledBrain.npy')[sliceChoice,:,:]
#im = (np.random.randn(294**2)+1.j*np.random.randn(294**2)).reshape([294,294])
im = im/np.max(abs(im))
N = np.array(im.shape)  # image Size
P = 2

pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=np.hstack([1, N[-2:]]), style='mult', pft=pft,ext=0.5)
if pft:
    print('Partial Fourier sampling method used')
k = samp.genSampling(pdf, 50, 2)[0].astype(int)
if len(N) == 2:
    N = np.hstack([1, N])
    k = k.reshape(N)
    im = im.reshape(N)
elif len(N) == 3:
    k = k.reshape(np.hstack([1,N[-2:]])).repeat(N[0],0)

ph_ones = np.ones(N[-2:], complex)
ph_scan = np.zeros(N, complex)
data = np.zeros(N,complex)
im_scan = np.zeros(N,complex)
for i in range(N[0]):
    k[i,:,:] = np.fft.fftshift(k[i,:,:])
    data[i,:,:] = k[i,:,:]*tf.fft2c(im[i,:,:], ph=ph_ones)

    # IMAGE from the "scanner data"
    im_scan_wph = tf.ifft2c(data[i,:,:], ph=ph_ones)
    ph_scan[i,:,:] = tf.matlab_style_gauss2D(im_scan_wph,shape=(5,5))
    ph_scan[i,:,:] = np.exp(1j*ph_scan[i,:,:])
    im_scan[i,:,:] = tf.ifft2c(data[i,:,:], ph=ph_scan[i,:,:])
    
    
    #im_lr = samp.loRes(im,pctg)


# ------------------------------------------------------------------ #
# A quick way to look at the PSF of the sampling pattern that we use #
delta = np.zeros(N[-2:])
delta[int(N[-2]/2),int(N[-1]/2)] = 1
psf = tf.ifft2c(tf.fft2c(delta,ph_ones)*k,ph_ones)
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
# -- Currently broken - Need to figure out what's happening here. -- #
# ------------------------------------------------------------------ #
if pft:
    for i in xrange(N[0]):
        dataHold = np.fft.fftshift(data[i,:,:])
        kHold = np.fft.fftshift(k[i,:,:])
        loc = 98
        for ix in xrange(N[-2]):
            for iy in xrange(loc,N[-1]):
                dataHold[-ix,-iy] = dataHold[ix,iy].conj()
                kHold[-ix,-iy] = kHold[ix,iy]
    # ------------------------------------------------------------------ #



pdfDiv = pdf.copy()
pdfZeros = np.where(pdf==0)
pdfDiv[pdfZeros] = 1
#im_scan_imag = im_scan.imag
#im_scan = im_scan

N_im = N
hld, dims, dimOpt, dimLenOpt = tf.wt(im_scan[0].real,wavelet,mode)
N = np.hstack([N_im[0], hld.shape])

w_scan = np.zeros(N)
w_full = np.zeros(N)
im_dc = np.zeros(N_im)
w_dc = np.zeros(N)

for i in xrange(N[0]):
    w_scan[i,:,:] = tf.wt(im_scan.real[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)[0]
    w_full[i,:,:] = tf.wt(abs(im[i,:,:]),wavelet,mode,dims,dimOpt,dimLenOpt)[0]

    im_dc[i,:,:] = tf.ifft2c(data[i,:,:] / np.fft.ifftshift(pdfDiv), ph=ph_scan[i,:,:]).real.copy()
    w_dc[i,:,:] = tf.wt(im_dc,wavelet,mode,dims,dimOpt,dimLenOpt)[0]
    k[i,:,:] = np.fft.fftshift(k[i,:,:])

w_dc = w_dc.flatten()
im_sp = im_dc.copy().reshape(N_im)
minval = np.min(abs(im))
maxval = np.max(abs(im))
data = np.ascontiguousarray(data)


imdcs = [im_dc,np.zeros(N_im),np.ones(N_im),np.random.randn(np.prod(N_im)).reshape(N_im)]
mets = ['Density Corrected','Zeros','Ones','Random']
wdcs = []


for i in range(len(imdcs)):
    wdcs.append(tf.wt(imdcs[i][0].real,wavelet,mode,dims,dimOpt,dimLenOpt)[0].reshape(N))

ims = []
#print('Starting the CS Algorithm')
for kk in range(len(wdcs)):
    w_dc = wdcs[kk]
    print(mets[kk])
    for i in range(len(TV)):
        args = (N, N_im, dims, dimOpt, dimLenOpt, TV[i], XFM[i], data, np.fft.fftshift(k), strtag, ph_scan, kern, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
        w_result = opt.minimize(f, w_dc, args=args, method=method, jac=df, 
                                    options={'maxiter': ItnLim, 'lineSearchItnLim': lineSearchItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': alpha_0, 'c': c, 'xtol': xtol[i], 'TVWeight': TV[i], 'XFMWeight': XFM[i], 'N': N})
        if np.any(np.isnan(w_result['x'])):
            print('Some nan''s found. Dropping TV and XFM values')
        elif w_result['status'] != 0:
            print('TV and XFM values too high -- no solution found. Dropping...')
        else:
            w_dc = w_result['x']
            
    w_res = w_dc.reshape(N)
    im_res = np.zeros(N_im)
    for i in xrange(N[0]):
        im_res[i,:,:] = tf.iwt(w_res[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)
    ims.append(im_res)
    
