#samplingTest.py

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
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/waveletDirection")
os.chdir('/home/asalerno/Documents/pyDirectionCompSense/')
import transforms as tf
import scipy.ndimage.filters
import gradWaveletMS as grads
import sampling as samp
import direction as d
import optimize as opt
import scipy.optimize as spopt
import functionsWaveletDirection as funs
import read_from_fid as rff
import saveFig
from scipy.interpolate import RectBivariateSpline
from unwrap2d import *
import visualization as vis
from scipy.ndimage.filters import gaussian_filter
from time import localtime, strftime
from pyminc.volumes.factory import *

strftime("%Y-%m-%d %H:%M:%S", localtime())

f = funs.objectiveFunction
df = funs.derivativeFunction
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(534)
ItnLim = 30
lineSearchItnLim = 30
wavelet = 'db4'
mode = 'per'
method = 'CG'

dirFile = '/home/asalerno/Documents/pyDirectionCompSense/GradientVectorMag.txt'
#engfile = '/home/asalerno/Documents/pyDirectionCompSense/phantom/engFile30dir_5mins.npy'
engfile = None
nmins = 5
dirs = np.loadtxt(dirFile)
nmins = 5
dirInfo = d.calc_Mid_Matrix(dirs,nmins)


radius = 0.2
pft=False
alpha_0 = 0.1
c = 0.6
a = 10.0 # value used for the tanh argument instead of sign

pctg = 0.25
sliceChoice = 150
xtol = [1e-3]
TV = [0.005] #, 0.005, 0.002, 0.001, 0.001]
XFM = [0.005] #, 0.005, 0.002, 0.001, 0.001]
dirWeight = 1


#im = np.load('/home/asalerno/Documents/pyDirectionCompSense/phantom/diffusionPhantomSNR1000.npy')
im = np.load('/home/asalerno/Documents/pyDirectionCompSense/directionData/singleSlice_30dirs.npy')
im=im/np.max(abs(im))
minval = np.min(abs(im))
maxval = np.max(abs(im))
N = np.array(im.shape)  # image Size
szFull = im.size


P = 2
nSteps = 4

print('k for Directions')
kDir = samp.genSamplingDir(img_sz=N, dirFile=dirFile, pctg=pctg, cyl=[1], radius=radius,            
                nmins=nmins, engfile=engfile)

pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=[1, N[-2], N[-1]], style='mult', pft=pft,ext=0.5)
k = np.zeros(N)

print('k for non')
for i in xrange(N[0]):                             
    np.random.seed(int(i*abs(np.random.randn(1)*np.random.randn(1))))
    print(i)
    k[i,:,:] = samp.genSampling(pdf, 50, 2)[0].astype(int)


ph_ones = np.ones(N[-2:], complex)
dataFull = np.zeros(N,complex)

ph_scan = np.zeros(N, complex)
data = np.zeros(N,complex)
im_scan = np.zeros(N, complex)

ph_scanDir = np.zeros(N, complex)
dataDir = np.zeros(N,complex)
im_scanDir = np.zeros(N, complex)

print('Data Production')
for i in range(N[0]):
    data[i,:,:] = np.fft.fftshift(k[i,:,:])*tf.fft2c(im[i,:,:], ph=ph_ones)
    dataDir[i,:,:] = np.fft.fftshift(kDir[i,:,:])*tf.fft2c(im[i,:,:], ph=ph_ones)
    dataFull[i,:,:] = np.fft.fftshift(tf.fft2c(im[i,:,:], ph=ph_ones))
    im_scan_wph = tf.ifft2c(data[i,:,:], ph=ph_ones)
    im_scan_wphDir = tf.ifft2c(dataDir[i,:,:], ph=ph_ones)
    ph_scan[i,:,:] = tf.matlab_style_gauss2D(im_scan_wph,shape=(5,5))
    ph_scanDir[i,:,:] = tf.matlab_style_gauss2D(im_scan_wphDir,shape=(5,5))
    ph_scan[i,:,:] = np.exp(1j*ph_scan[i,:,:])
    ph_scanDir[i,:,:] = np.exp(1j*ph_scanDir[i,:,:])
    im_scan[i,:,:] = tf.ifft2c(data[i,:,:], ph=ph_scan[i,:,:])
    im_scanDir[i,:,:] = tf.ifft2c(dataDir[i,:,:], ph=ph_scanDir[i,:,:])


print('Mix the Data')
dataDirComb = d.dir_dataSharing(kDir,dataDir,dirs,[256,256],maxCheck=5,bymax=1)
dataComb = d.dir_dataSharing(k,data,dirs,[256,256],maxCheck=5,bymax=1)

ph_scanComb = np.zeros(N, complex)
im_scanComb = np.zeros(N, complex)
ph_scanDirComb = np.zeros(N, complex)
im_scanDirComb = np.zeros(N, complex)

print('Create Mixed Images')
for i in range(N[0]):
    im_scan_wphComb = tf.ifft2c(dataComb[i,:,:], ph=ph_ones)
    ph_scanComb[i,:,:] = tf.matlab_style_gauss2D(im_scan_wphComb,shape=(5,5))
    ph_scanComb[i,:,:] = np.exp(1j*ph_scanComb[i,:,:])
    im_scanComb[i,:,:] = tf.ifft2c(dataComb[i,:,:], ph=ph_scanComb[i,:,:])
    im_scan_wphDirComb = tf.ifft2c(dataDirComb[i,:,:], ph=ph_ones)
    ph_scanDirComb[i,:,:] = tf.matlab_style_gauss2D(im_scan_wphDirComb,shape=(5,5))
    ph_scanDirComb[i,:,:] = np.exp(1j*ph_scanDirComb[i,:,:])
    im_scanDirComb[i,:,:] = tf.ifft2c(dataDirComb[i,:,:], ph=ph_scanDirComb[i,:,:])
    

print('Plot it')
clims = [[minval,maxval]]*5
for i in range(3):
    clims.append([0,1])
vis.figSubplots([abs(im[0]),
                 im_scan[0].real,
                 im_scanComb[0].real,
                 im_scanDir[0].real,
                 im_scanDirComb[0].real,
                 k[0],
                 kDir[0],
                 np.zeros([256,256])],
                 clims=clims,
                 titles=['Original','Var Dens','Var Dens Comb','Uni Samp','Uni Samp Comb','Var Dens k','Uni Dens k',''])
                 
#plt.show()
saveFig.save('tests/directionTests/samplingMethodsBrain')