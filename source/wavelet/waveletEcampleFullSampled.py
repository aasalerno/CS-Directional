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

phIter = 0
sliceChoice = 150
pctg = 1
xtol = [1e-2, 1e-3, 5e-4]
TV = [0.02, 0.01, 0.005]
XFM = [0.02, 0.01, 0.005]
#TV = [0.01, 0.005, 0.002, 0.001]
#XFM = [0.01,.005, 0.002, 0.001]
radius = 0.2

im = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/P14/data/fullySampledBrain.npy')[sliceChoice,:,:]
#im = (np.random.randn(294**2)+1.j*np.random.randn(294**2)).reshape([294,294])
im = im/np.max(abs(im))
N = np.array(im.shape)  # image Size
P = 2

k = np.ones(N)
if len(N) == 2:
    N = np.hstack([1, N])
    k = k.reshape(N)
    im = im.reshape(N)
elif len(N) == 3:
    k = k.reshape(np.hstack([1,N[-2:]])).repeat(N[0],0)

im_scan = abs(im).reshape(N)
im_dc = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/P14/data/im_dc.npy')


ph_ones = np.ones(N[-2:], complex)
ph_scan = np.exp(1.j*np.angle(im))
data = np.zeros(N,complex)
for i in range(N[0]):
    k[i,:,:] = np.fft.fftshift(k[i,:,:])
    data[i,:,:] = k[i,:,:]*tf.fft2c(im[i,:,:], ph=ph_ones)

N_im = N
hld, dims, dimOpt, dimLenOpt = tf.wt(im_scan[0].real,wavelet,mode)
N = np.hstack([N_im[0], hld.shape])

w_scan = np.zeros(N)
w_full = np.zeros(N)
w_dc = np.zeros(N)

for i in xrange(N[0]):
    w_scan[i,:,:] = tf.wt(im_scan.real[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)[0]
    w_full[i,:,:] = tf.wt(abs(im[i,:,:]),wavelet,mode,dims,dimOpt,dimLenOpt)[0]
    w_dc[i,:,:] = tf.wt(im_dc[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)[0]

w_dc = w_dc.flatten()
im_sp = im_dc.copy().reshape(N_im)
minval = np.min(abs(im))
maxval = np.max(abs(im))
data = np.ascontiguousarray(data)


imdcs = [im_dc]
mets = ['Density Corrected']
wdcs = []


for i in range(len(imdcs)):
    wdcs.append(tf.wt(imdcs[i][0].real,wavelet,mode,dims,dimOpt,dimLenOpt)[0].reshape(N))

ims = []
stps = []
tvStps = []
#print('Starting the CS Algorithm')
for kk in range(len(wdcs)):
    w_dc = wdcs[kk]
    print(mets[kk])
    for i in range(len(TV)):
        args = (N, N_im, dims, dimOpt, dimLenOpt, TV[i], XFM[i], data, k, strtag, ph_scan, kern, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
        w_result = opt.minimize(f, w_dc, args=args, method=method, jac=df, 
                                    options={'maxiter': ItnLim, 'lineSearchItnLim': lineSearchItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': alpha_0, 'c': c, 'xtol': xtol[i], 'TVWeight': TV[i], 'XFMWeight': XFM[i], 'N': N})
        if np.any(np.isnan(w_result['x'])):
            print('Some nan''s found. Dropping TV and XFM values')
        elif w_result['status'] != 0:
            print('TV and XFM values too high -- no solution found. Dropping...')
        else:
            w_dc = w_result['x']
            stps.append(w_dc)
            tvStps.append(TV[i])
            
            
    w_res = w_dc.reshape(N)
    im_res = np.zeros(N_im)
    for i in xrange(N[0]):
        im_res[i,:,:] = tf.iwt(w_res[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)
    ims.append(im_res)
    
im_stps = np.zeros([len(stps), N_im[-2], N_im[-1]])
gtv = np.zeros([len(stps), N_im[-2], N_im[-1]])
gxfm = np.zeros([len(stps), N_im[-2], N_im[-1]])
gdc = np.zeros([len(stps), N_im[-2], N_im[-1]])
for jj in range(len(stps)):
    im_stps[jj,:,:] = tf.iwt(stps[jj].reshape(N[-2:]),wavelet,mode,dims,dimOpt,dimLenOpt)
    gtv[jj,:,:] = grads.gTV(im_stps[jj,:,:].reshape(N_im),N_im,strtag, kern, 0, a=a)
    gxfm[jj,:,:] = tf.iwt(grads.gXFM(stps[jj].reshape(N[-2:]),a=a),wavelet,mode,dims,dimOpt,dimLenOpt)
    gdc[jj,:,:] = grads.gDataCons(im_stps[jj,:,:], N_im, ph_scan, data, k)
    

    
for i in xrange(len(stps)):
    vis.figSubplots([im_stps[i],gtv[i],gxfm[i],gdc[i]],titles=['Step','gTV','gXFM','gDC'])
    saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/gradTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_grads')
    plt.imshow(TV[i]*gtv[i] + TV[i]*gxfm[i] + gdc[i])
    plt.colorbar()
    plt.title('Total Gradient')
    saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/gradTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_gradTotal')
