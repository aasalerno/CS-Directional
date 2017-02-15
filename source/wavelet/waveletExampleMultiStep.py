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
import gradWaveletMS as grads
import sampling as samp
import direction as d
import optimize as opt
import scipy.optimize as spopt
import wavelet_MS_DC_TV_XFM_f_df as funs
import read_from_fid as rff
import saveFig
from scipy.interpolate import RectBivariateSpline
from unwrap2d import *
import visualization as vis

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

pctg = 0.25
phIter = 0
sliceChoice = 150
xtol = [1e-2, 1e-2, 1e-3, 5e-4, 5e-4]
TV = [0.005, 0.005, 0.002, 0.001, 0.001]
XFM = [0.005, 0.005, 0.002, 0.001, 0.001]
radius = 0.2

im = np.load('/home/asalerno/Documents/pyDirectionCompSense/phantom/imFullCyl.npy')
im = im/np.max(abs(im))
minval = np.min(abs(im))
maxval = np.max(abs(im))
N = np.array(im.shape)  # image Size
P = 2

pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=[1, N[-2], N[-1]], style='mult', pft=pft,ext=0.5)
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
dataFull = np.zeros(N,complex)
im_scan = np.zeros(N,complex)
for i in range(N[0]):
    #k[i,:,:] = np.fft.fftshift(k[i,:,:])
    data[i,:,:] = np.fft.fftshift(k[i,:,:])*tf.fft2c(im[i,:,:], ph=ph_ones)
    dataFull[i,:,:] = np.fft.fftshift(tf.fft2c(im[i,:,:], ph=ph_ones))

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

x, y = np.meshgrid(np.linspace(-1,1,N[-1]),np.linspace(-1,1,N[-2]))
locs = (abs(x)<=radius) * (abs(y)<=radius)
minLoc = np.min(np.where(locs==True))

pctgSamp = np.zeros(minLoc+1)
for i in range(1,minLoc+1):
    kHld = k[0,i:-i,i:-i]
    pctgSamp[i] = np.sum(kHld)/kHld.size

nSteps = 4
pctgLocs = np.arange(1,nSteps+1)/(nSteps)

locSteps = np.zeros(nSteps)
locSteps[0] = minLoc

for i in range(nSteps):
    locSteps[i] = np.argmin(abs(pctgLocs[i]-pctgSamp))

# Flip it here to make sure we're starting at the right point
locSteps = locSteps[::-1].astype(int)
locSteps = np.hstack([locSteps,0])

ims = []
stps = []
szStps = []
imStp = []
tvStps = []
#data_stp = np.zeros(N_im,complex)
w_stp = []
szFull = im.size    


for j in range(nSteps+1):
    NSub = np.array([N[0], N[1]-2*locSteps[j], N[2]-2*locSteps[j]]).astype(int)
    ph_onesSub = np.ones(NSub[-2:], complex)
    ph_scanSub = np.zeros(NSub, complex)
    dataSub = np.zeros(NSub,complex)
    im_scanSub = np.zeros(NSub,complex)
    im_FullSub = np.zeros(NSub,complex)
    kSub = np.zeros(NSub)
    for i in range(N[0]):
        if locSteps[j]==0:
            kSub[i,:,:] = k[i,:,:]
            dataSub[i,:,:] = np.fft.fftshift(kSub[i,:,:]*dataFull[i,:,:])
            im_FullSub = tf.ifft2c(np.fft.fftshift(dataFull[i,:,:]),ph=ph_onesSub,sz=szFull)
        else:
            kSub[i,:,:] = k[i,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]]
            dataSub[i,:,:] = np.fft.fftshift(kSub[i,:,:]*dataFull[i,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]])
            im_FullSub = tf.ifft2c(np.fft.fftshift(dataFull[i,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]]),ph=ph_onesSub,sz=szFull)
            
        im_scan_wphSub = tf.ifft2c(dataSub[i,:,:], ph=ph_onesSub, sz=szFull)
        ph_scanSub[i,:,:] = tf.matlab_style_gauss2D(im_scan_wphSub,shape=(5,5))
        ph_scanSub[i,:,:] = np.exp(1j*ph_scanSub[i,:,:])
        im_scanSub[i,:,:] = tf.ifft2c(dataSub[i,:,:], ph=ph_scanSub[i,:,:], sz=szFull)
    
    if j == 0:
        kMasked = kSub[0].copy()
    else:
        padMask = tf.zpad(np.ones(kMasked.shape),NSub[-2:])
        kMasked = (1-padMask)*kSub[0] + padMask*tf.zpad(kMasked,NSub[-2:])
    
    
    # Now we need to construct the starting point
    if locSteps[j]==0:
        pdfDiv = pdf.copy()
    else:
        pdfDiv = pdf[locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
    pdfZeros = np.where(pdfDiv < 1e-4)
    pdfDiv[pdfZeros] = 1
    
    N_imSub = NSub
    hldSub, dimsSub, dimOptSub, dimLenOptSub = tf.wt(im_scanSub[0].real,wavelet,mode)
    NSub = np.hstack([N_imSub[0], hldSub.shape])
    
    w_scanSub = np.zeros(NSub)
    im_dcSub = np.zeros(N_imSub)
    w_dcSub = np.zeros(NSub)
    
    for i in xrange(N[0]):
        w_scanSub[i,:,:] = tf.wt(im_scanSub.real[i,:,:],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0]

        im_dcSub[i,:,:] = tf.ifft2c(dataSub[i,:,:] / np.fft.ifftshift(pdfDiv), ph=ph_scanSub[i,:,:], sz=szFull).real.copy()
        w_dcSub[i,:,:] = tf.wt(im_dcSub,wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0]
        kSub[i,:,:] = np.fft.fftshift(kSub[i,:,:])
        
    w_dcSub = w_dcSub.flatten()
    im_spSub = im_dcSub.copy().reshape(N_imSub)
    dataSub = np.ascontiguousarray(dataSub)
    
    mets = ['Density Corrected']#,'Zeros','Ones','Random']
    wdcs = []
    
    if (j!=0) and (locSteps[j]!=0):
        kpad = tf.zpad(kStp[0],np.array(kSub.shape[-2:]).astype(int))
        data_dc = np.fft.fftshift(tf.zpad(np.ones(kStp.shape[-2:])*dataStp[0], np.array(kSub.shape[-2:]).astype(int)) + (1-kpad)*np.fft.fftshift(dataSub))
        im_dcSub = tf.ifft2c(data_dc[i,:,:] / np.fft.ifftshift(pdfDiv), ph=ph_scanSub[i,:,:]).real.copy().reshape(N_imSub)
    imdcs = [im_dcSub] #,np.zeros(N_im),np.ones(N_im),np.random.randn(np.prod(N_im)).reshape(N_im)]
    for i in range(len(imdcs)):
        wdcs.append(tf.wt(imdcs[i][0].real,wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0].reshape(NSub))
        
    
    
    for kk in range(len(wdcs)):
        w_dc = wdcs[kk]
        print(mets[kk])
        args = (NSub, N_imSub, szFull, dimsSub, dimOptSub, dimLenOptSub, TV[j], XFM[j], dataSub, kSub, strtag, ph_scanSub, kern, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
        w_result = opt.minimize(f, w_dc, args=args, method=method, jac=df, 
                                    options={'maxiter': ItnLim, 'lineSearchItnLim': lineSearchItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': alpha_0, 'c': c, 'xtol': xtol[j], 'TVWeight': TV[j], 'XFMWeight': XFM[j], 'N': NSub})
        #if np.any(np.isnan(w_result['x'])):
            #print('Some nan''s found. Dropping TV and XFM values')
        #elif w_result['status'] != 0:
            #print('TV and XFM values too high -- no solution found. Dropping...')
        #else:
            #w_dc = w_result['x']
            ##import pdb; pdb.set_trace()
            #stps.append(w_dc)
            #tvStps.append(TV[i])
            #w_stp.append(w_dc.reshape(NSub))
            #imStp.append(tf.iwt(w_stp[-1][0],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub))
            ##plt.imshow(imStp[-1]); plt.colorbar(); plt.show()
            ##w_dc = w_stp[k].flatten()
            #stps.append(w_dc)
            #wdcHold = w_dc.reshape(NSub)
            #kMasked = np.floor(1-kMasked)*pctgSamp[locSteps[j]] + kMasked
        #if j == len(TV):
            #print('No solution found on final run. Saving last spot.')
            #w_dc = w_result['x']
            ##import pdb; pdb.set_trace()
            #stps.append(w_dc)
            #tvStps.append(TV[i])
            #w_stp.append(w_dc.reshape(NSub))
            #imStp.append(tf.iwt(w_stp[-1][0],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub))
            ##plt.imshow(imStp[-1]); plt.colorbar(); plt.show()
            ##w_dc = w_stp[k].flatten()
            #stps.append(w_dc)
            #wdcHold = w_dc.reshape(NSub)
            ##dataStp = np.fft.fftshift(tf.fft2c(imStp[-1],ph_scanSub))
            ##kStp = np.fft.fftshift(kSub).copy()
            #kMasked = np.floor(1-kMasked)*pctgSamp[locSteps[j]] + kMasked
        w_dc = w_result['x']
        #import pdb; pdb.set_trace()
        stps.append(w_dc)
        tvStps.append(TV[i])
        w_stp.append(w_dc.reshape(NSub))
        imStp.append(tf.iwt(w_stp[-1][0],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub))
        #plt.imshow(imStp[-1]); plt.colorbar(); plt.show()
        #w_dc = w_stp[k].flatten()
        stps.append(w_dc)
        wdcHold = w_dc.reshape(NSub)
        kMasked = np.floor(1-kMasked)*pctgSamp[locSteps[j]] + kMasked    
        dataStp = np.fft.fftshift(tf.fft2c(imStp[-1],ph_scanSub))
        kStp = np.fft.fftshift(kSub).copy()