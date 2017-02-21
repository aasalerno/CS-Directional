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
from scipy.ndimage.filters import gaussian_filter
from time import gmtime, strftime
from pyminc.volumes.factory import *

strftime("%Y-%m-%d %H:%M:%S", gmtime())

f = funs.objectiveFunction
df = funs.derivativeFunction
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(534)

inputdirectory="/hpf/largeprojects/MICe/segan/exercise_irradiation/bruker_data/running_C/P14/20160607_124310_Running_C_1_1"
petable = "/hpf/largeprojects/MICe/bjnieman/Bruker_cyltests/cylbruker_nTRfeath18_294_294"

strtag = ['spatial','spatial', 'spatial']
dirWeight = 0
ItnLim = 30
lineSearchItnLim = 30
wavelet = 'db4'
mode = 'per'
method = 'CG'
#kern = np.array([[[ 0.,  0.,  0.],
                  #[ 0.,  0.,  0.],
                  #[ 0.,  0.,  0.]],
                  
                  #[[ 0.,  0.,  0.],
                  #[ 0., -1.,  0.],
                  #[ 0.,  1.,  0.]],
                  
                  #[[ 0.,  0.,  0.],
                  #[ 0., -1.,  1.],
                  #[ 0.,  0.,  0.]]])


kern = np.zeros([3,3,3,3])
for i in range(kern.shape[0]):
    kern[i,1,1,1] = -1

kern[0,2,1,1] = 1
kern[1,1,2,1] = 1
kern[2,1,1,2] = 1
                  
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

pctg = 0.50
phIter = 0
sliceChoice = 150
xtol = [0.1, 1e-2, 1e-3, 5e-4, 5e-4]
TV = [0.005]#, 0.005, 0.002, 0.001, 0.001]
XFM = [0.005]#, 0.005, 0.002, 0.001, 0.001]
radius = 0.2

im = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/P14/data/fullySampledBrain.npy')
#im = im[150:152,:,:]
#im = np.load('/home/asalerno/Documents/pyDirectionCompSense/phantom/imFull.npy')
im = im/np.max(abs(im))
minval = np.min(abs(im))
maxval = np.max(abs(im))
N = np.array(im.shape)  # image Size
szFull = im.size


P = 2
nSteps = 4
if len(TV) < (nSteps+1):
    for i in xrange(nSteps - len(TV) + 1):
        TV.append(TV[-1])
        XFM.append(XFM[-1])
        xtol.append(xtol[-1])


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

#if N[0] == 1:
    ## IMAGE from the "scanner data"
    #for i in range(N[0]):
        #data[i,:,:] = np.fft.fftshift(k[i,:,:])*tf.fft2c(im[i,:,:], ph=ph_ones)
        #dataFull[i,:,:] = np.fft.fftshift(tf.fft2c(im[i,:,:], ph=ph_ones))
        #im_scan_wph = tf.ifft2c(data[i,:,:], ph=ph_ones)
        #ph_scan[i,:,:] = tf.matlab_style_gauss2D(im_scan_wph,shape=(5,5))
        #ph_scan[i,:,:] = np.exp(1j*ph_scan[i,:,:])
        #im_scan = tf.ifft2c(data[i,:,:], ph=ph_scan[i,:,:])
#else:
data = k*tf.fftnc(im, ph=ph_ones)
dataFull = np.fft.fftshift(tf.fftnc(im, ph=ph_ones))
im_scan_wph = tf.ifftnc(data, ph=np.ones(data.shape))
ph_scan = np.angle(gaussian_filter(im_scan_wph.real,2) +  1.j*gaussian_filter(im_scan_wph.imag,2))
ph_scan = np.exp(1j*ph_scan)
im_scan = tf.ifft2c(data, ph=ph_scan)
    
    #im_lr = samp.loRes(im,pctg)


## ------------------------------------------------------------------ #
## A quick way to look at the PSF of the sampling pattern that we use #
#delta = np.zeros(N[-2:])
#delta[int(N[-2]/2),int(N[-1]/2)] = 1
#psf = tf.ifft2c(tf.fft2c(delta,ph_ones)*k,ph_ones)
## ------------------------------------------------------------------ #


## ------------------------------------------------------------------ #
## -- Currently broken - Need to figure out what's happening here. -- #
## ------------------------------------------------------------------ #
#if pft:
    #for i in xrange(N[0]):
        #dataHold = np.fft.fftshift(data[i,:,:])
        #kHold = np.fft.fftshift(k[i,:,:])
        #loc = 98
        #for ix in xrange(N[-2]):
            #for iy in xrange(loc,N[-1]):
                #dataHold[-ix,-iy] = dataHold[ix,iy].conj()
                #kHold[-ix,-iy] = kHold[ix,iy]
    ## ------------------------------------------------------------------ #



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

Ns = []
N_ims = []
dims = [] 
dimOpts = [] 
dimLenOpts = []
kMasks = []
ph_scans = []
dataSubs=[]



for j in range(nSteps+1):
    NSub = np.array([N[0], N[1]-2*locSteps[j], N[2]-2*locSteps[j]]).astype(int)
    ph_onesSub = np.ones(NSub, complex)
    ph_scanSub = np.zeros(NSub, complex)
    dataSub = np.zeros(NSub,complex)
    im_scanSub = np.zeros(NSub,complex)
    im_FullSub = np.zeros(NSub,complex)
    kSub = np.zeros(NSub)
    if N[0] == 1:
        i = 0
        if locSteps[j]==0:
            kSub = k[i,:,:].copy()
            dataSub[i,:,:] = np.fft.fftshift(kSub*dataFull[i,:,:])
            im_FullSub = tf.ifft2c(np.fft.fftshift(dataFull[i,:,:]),ph=ph_onesSub[i],sz=szFull/N[0])
        else:
            kSub[i,:,:] = k[i,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
            dataSub[i,:,:] = np.fft.fftshift(kSub[i,:,:]*dataFull[i,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]])
            im_FullSub = tf.ifft2c(np.fft.fftshift(dataFull[i,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]]),ph=ph_onesSub,sz=szFull/N[0])
            
        im_scan_wphSub = tf.ifft2c(dataSub[i,:,:], ph=ph_onesSub, sz=szFull/N[0])
        ph_scanSub[i,:,:] = tf.matlab_style_gauss2D(im_scan_wphSub,shape=(5,5))
        ph_scanSub[i,:,:] = np.exp(1j*ph_scanSub[i,:,:])
        im_scanSub[i,:,:] = tf.ifft2c(dataSub[i,:,:], ph=ph_scanSub[i,:,:], sz=szFull/N[0])
    else:
        if locSteps[j]==0:
            kSub = k.copy()
            dataSub = np.fft.fftshift(kSub*dataFull)
            im_FullSub = tf.ifftnc(np.fft.fftshift(dataFull),ph=ph_onesSub,sz=szFull)
        else:
            kSub = k[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
            dataSub = np.fft.fftshift(kSub*dataFull[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]])
            im_FullSub = tf.ifftnc(np.fft.fftshift(dataFull[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]]),ph=ph_onesSub,sz=szFull)
            
        im_scan_wphSub = tf.ifftnc(dataSub, ph=ph_onesSub, sz=szFull)
        ph_scanSub = np.angle(gaussian_filter(im_scan_wphSub.real,0) +  1.j*gaussian_filter(im_scan_wphSub.imag,0))
        ph_scanSub = np.exp(1j*ph_scanSub)
        im_scanSub = tf.ifftnc(dataSub, ph=ph_scanSub, sz=szFull)
    
    if j == 0:
        if len(kSub.shape) == 3:
            kMasked = kSub[0].copy()
        else:
            kMasked = kSub.copy()
    else:
        padMask = tf.zpad(np.ones(kMasked[0].shape),NSub[-2:])
        kMasked = ((1-padMask)*kSub[0] + padMask*tf.zpad(kMasked[0],NSub[-2:])).reshape(np.hstack([1,NSub[-2:]]))
    kMasks.append(kMasked)
    
    # Now we need to construct the starting point
    if locSteps[j]==0:
        pdfDiv = pdf.copy()
    else:
        pdfDiv = pdf[locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
    pdfZeros = np.where(pdfDiv < 1e-4)
    pdfDiv[pdfZeros] = 1
    pdfDiv = pdfDiv.reshape(np.hstack([1,NSub[-2:]])).repeat(NSub[0],0)
    
    N_imSub = NSub
    hldSub, dimsSub, dimOptSub, dimLenOptSub = tf.wt(im_scanSub[0].real,wavelet,mode)
    NSub = np.hstack([N_imSub[0], hldSub.shape])
    
    w_scanSub = np.zeros(NSub)
    im_dcSub = np.zeros(N_imSub)
    w_dcSub = np.zeros(NSub)
    
    im_dcSub = tf.ifftnc(dataSub / np.fft.ifftshift(pdfDiv), ph=ph_scanSub, sz=szFull).real.copy()
    for i in xrange(N[0]):
        w_scanSub[i,:,:] = tf.wt(im_scanSub.real[i,:,:],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0]

        w_dcSub[i,:,:] = tf.wt(im_dcSub[i,:,:],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0]
        if len(kSub.shape) == 3:
            kSub[i,:,:] = np.fft.fftshift(kSub[i,:,:])
        else:
            kSub = np.fft.fftshift(kSub)
        
    w_dcSub = w_dcSub.flatten()
    im_spSub = im_dcSub.copy().reshape(N_imSub)
    dataSub = np.ascontiguousarray(dataSub)
    
    mets = ['Density Corrected']#,'Zeros','Ones','Random']
    wdcs = []
    
    #if (j!=0) and (locSteps[j]!=0):
        #kpad = tf.zpad(kStp[0],np.array(kSub.shape[-2:]).astype(int))
        #data_dc = np.fft.fftshift(tf.zpad(np.ones(kStp.shape[-2:])*dataStp[0], np.array(kSub.shape[-2:]).astype(int)) + (1-kpad)*np.fft.fftshift(dataSub))
        #im_dcSub = tf.ifft2c(data_dc[i,:,:] / np.fft.ifftshift(pdfDiv), ph=ph_scanSub[i,:,:]).real.copy().reshape(N_imSub)
    imdcs = [im_dcSub] #,np.zeros(N_im),np.ones(N_im),np.random.randn(np.prod(N_im)).reshape(N_im)]
    #import pdb; pdb.set_trace()
    
    kSamp = np.fft.fftshift(kMasked).reshape(np.hstack([1,N_imSub[-2:]])).repeat(N_imSub[0],0)
    
    args = (NSub, N_imSub, szFull, dimsSub, dimOptSub, dimLenOptSub, TV[0], XFM[0], dataSub, kSamp, strtag, ph_scanSub, kern, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
    w_result = opt.fmin_tnc(f, w_scanSub.flat, fprime=df, args=args, accuracy=1e-4, disp=0)
    w_dc = w_result[0]
    stps.append(w_dc)
    tvStps.append(TV[j])
    w_stp.append(w_dc.reshape(NSub))
    imStp.append(tf.iwt(w_stp[-1][0],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub))
    #plt.imshow(imStp[-1],clim=(minval,maxval)); plt.colorbar(); plt.show()
    #w_dc = w_stp[k].flatten()
    #stps.append(w_dc)
    wdcHold = w_dc.reshape(NSub)
    dataStp = np.fft.fftshift(tf.fft2c(imStp[-1],ph_scanSub))
    kStp = np.fft.fftshift(kSub).copy()
    #kMaskRpt = kMasked.reshape(np.hstack([1,N_imSub[-2:]])).repeat(N_imSub[0],0)
    kMasked = (np.floor(1-kMasked)*pctgSamp[locSteps[j]]*1.0 + kMasked).reshape(np.hstack([1, N_imSub[-2:]]))
    #Ns.append(NSub)
    #N_ims.append(N_imSub)
    #dims.append(dimsSub)
    #dimOpts.append(dimOptSub)
    #dimLenOpts.append(dimLenOptSub)
    #ph_scans.append(ph_scanSub)
    #dataSubs.append(dataSub)
    
wHold = w_dc.copy().reshape(NSub)
imHold = np.zeros(N_imSub)

for i in xrange(N[0]):
    imHold[i,:,:] = tf.iwt(wHold[i,:,:],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)
    

np.save('tests/fullBrainTests/' + str(int(100*pctg)) + '_3_spatial_0.005_TV_im_final.npy',imHold)
np.save('tests/fullBrainTests/' + str(int(100*pctg)) + '_3_spatial_0.005_TV_im_final.npy',wHold)


outvol = volumeFromData('tests/fullBrainTests/' + str(int(100*pctg)) + '_3_spatial_0.005_TV_im_final.mnc', imHold, dimnames=['xspace','yspace','zspace'], starts=(0, 0, 0), steps=(1, 1, 1), volumeType="uint")
outvol.writeFile()
strftime("%Y-%m-%d %H:%M:%S", gmtime())


#im_stps = []
#gtv = []
#gxfm = []
#gdc = [] 
#for jj in range(len(stps)):
    #im_stps.append(tf.iwt(stps[jj].reshape(Ns[jj][-2:]),wavelet,mode,dims[jj],dimOpts[jj],dimLenOpts[jj]))
    #gtv.append(grads.gTV(im_stps[jj].reshape(N_ims[jj]),N_ims[jj],strtag, kern, 0, a=a).reshape(N_ims[jj][-2:]))
    #gxfm.append(tf.iwt(grads.gXFM(stps[jj].reshape(Ns[jj][-2:]),a=a),wavelet,mode,dims[jj],dimOpts[jj],dimLenOpts[jj]).reshape(N_ims[jj][-2:]))
    #gdc.append(grads.gDataCons(im_stps[jj], N_ims[jj], ph_scans[jj], dataSubs[jj], kMasks[jj],sz=szFull).reshape(N_ims[jj][-2:]))
        
#pctgLocs = np.hstack([pctgLocs, 1.05])
        
#for i in xrange(len(stps)):
    #plt.imshow(imStp[i]); plt.colorbar();
    #plt.title('Step with Negative Values')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/stepTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_a_' + str(int(a)) + '_setSamp_' + str(int(100*pctgLocs[i])) + '_nSteps_' + str(int(nSteps)) + '_negs')
    #vis.figSubplots([im_stps[i],gtv[i],gxfm[i],gdc[i]],titles=['Step','gTV','gXFM','gDC'],clims=[(minval,maxval),(np.min(gtv[i]),np.max(gtv[i])),(np.min(gxfm[i]),np.max(gxfm[i])),(np.min(gdc[i]),np.max(gdc[i]))])
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/stepTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_a_' + str(int(a)) + '_setSamp_' + str(int(100*pctgLocs[i])) + '_nSteps_' + str(int(nSteps)) + '_grads')
    #plt.imshow(TV[i]*gtv[i] + TV[i]*gxfm[i] + gdc[i])
    #plt.colorbar()
    #plt.title('Total Gradient')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/stepTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_a_' + str(int(a)) + '_setSamp_' + str(int(100*pctgLocs[i])) + '_nSteps_' + str(int(nSteps)) + '_gradTotal')
    #if locSteps[i] != 0:
        #plt.imshow(abs(abs(im[0,locSteps[i]:-locSteps[i],locSteps[i]:-locSteps[i]]) - imStp[i]));
    #else:
        #plt.imshow(abs(abs(im[0] - imStp[i])));
    #plt.colorbar();
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/stepTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_a_' + str(int(a)) + '_setSamp_' + str(int(100*pctgLocs[i])) + '_nSteps_' + str(int(nSteps)) + 'imDiff')
    
    