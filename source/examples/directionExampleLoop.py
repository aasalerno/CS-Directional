# Imports
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import os.path
from sys import path as syspath
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/")
os.chdir(
    '/home/asalerno/Documents/pyDirectionCompSense/')  # Change this to the directory that you're saving the work in
from varian_read_dti import getDTIDataFromFID 
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
import visualization as vis

np.random.seed(627)

xtol = [1e-2, 1e-3, 5e-4, 5e-4, 5e-4]
TV = [0.01, 0.005, 0.002, 0.001, 0.0005]
XFM = [0.01, 0.005, 0.002, 0.001, 0.0005]
dirWeight = 0.1
radius = 0.15

# Data information
useNPY = True
inputfilename = '/hpf/largeprojects/MICe/asalerno/DTIdata/26apr16.fid/kspace_slices/kspace_ro_slice_165.npy'
inputdirectory='/hpf/largeprojects/MICe/jacob/fid/26apr16.fid'
petable='/projects/souris/jacob/fid/table_test/JE_Table_Angdist_nf60_ni17'
imouse = 5
sampfilename = '/home/asalerno/Documents/pyDirectionCompSense/directionData/30dirSampling_25per.npy'
engfile='/micehome/asalerno/Documents/pyDirectionCompSense/engFile30dir.npy'
#sampfilename = None
#engfile = None
strtag = ['diff','spatial', 'spatial']
# DirType = 2
ItnLim = 150
epsilon = 1e-6
l1smooth = 1e-15
xfmNorm = 1
wavelet = 'db4'
mode = 'per'
method = 'CG'
dirFile = '/home/asalerno/Documents/pyDirectionCompSense/GradientVectorMag.txt'
nmins = 5
dirs = np.loadtxt(dirFile)
ndirs = dirs.shape[0]
dirInfo = d.calc_Mid_Matrix(dirs,nmins)
pctg = 0.25
radius = 0.15
alpha_0 = 0.1
c = 0.6
a = 10.0 # value used for the tanh argument instead of sign

if useNPY:
    data = np.load(inputfilename)
else:
    data = getDTIDataFromFID(inputdirectory=inputdirectory,petable=petable,imouse=imouse)
N = data.shape

# Now look at the sampling pattern. Recreate it if required.
if sampfilename:
    k = np.load(sampfilename)
else:
    k = samp.genSamplingDir(img_sz=N[-2:], dirFile=dirFile, pctg=pctg, cyl=[1],radius=radius, nmins=nmins, engfile=engfile)

data_b0 = data[dirs.shape[0]:,:,:]
data_b1_full = data[:dirs.shape[0],:,:]
data_b1 = data_b1_full*np.fft.fftshift(k,axes=(-2,-1))
N = data_b1.shape

######################################################
# Remember that the b0 will ALWAYS BE FULLY SAMPLED

# Try to find the phase of the fully sampled b0s as well, so have a ph_ones
ph_ones = np.ones(N[-2:])

ph_b0 = np.ones(data_b0.shape, dtype='complex')
im_b0_wph = np.zeros(data_b0.shape, dtype='complex')
im_b0_scan = np.zeros(data_b0.shape, dtype='complex')

for i in range(data_b0.shape[0]):
    im_b0_wph[i,:,:] = tf.ifft2c(data_b0[i,:,:],ph=ph_ones)
    ph_b0[i,:,:] = np.exp(1j*tf.matlab_style_gauss2D(im_b0_wph[i,:,:],shape=(5,5)))
    im_b0_scan[i,:,:] = tf.ifft2c(data_b0[i,:,:],ph_b0[i,:,:])

im_b0_avg = np.mean(im_b0_scan,axis=(0))
minval = np.min(abs(im_b0_avg))
maxval = np.max(abs(im_b0_avg))

###############################################################################
# Now for both the undersampled cases and fully sampled cases for the actual
ph_b1 = np.ones(data_b1.shape, dtype='complex')
im_b1_wph = np.zeros(data_b1.shape, dtype='complex')
im_b1_scan = np.zeros(data_b1.shape, dtype='complex')

ph_b1_full = np.ones(data_b1.shape, dtype='complex')
im_b1_wph_full = np.zeros(data_b1.shape, dtype='complex')
im_b1_full = np.zeros(data_b1.shape, dtype='complex')

for i in range(data_b1.shape[0]):
    # Fully sampled case
    im_b1_wph_full[i,:,:] = tf.ifft2c(data_b1_full[i,:,:],ph=ph_ones)
    ph_b1_full[i,:,:] = np.exp(1j*tf.matlab_style_gauss2D(im_b1_wph_full[i,:,:],shape=(5,5)))
    im_b1_full[i,:,:] = tf.ifft2c(data_b1_full[i,:,:],ph=ph_b1_full[i,:,:])
    # US Case
    im_b1_wph[i,:,:] = tf.ifft2c(data_b1[i,:,:],ph=ph_ones)
    ph_b1[i,:,:] = np.exp(1j*tf.matlab_style_gauss2D(im_b1_wph[i,:,:],shape=(5,5)))
    im_b1_scan[i,:,:] = tf.ifft2c(data_b1[i,:,:],ph_b1[i,:,:])

    
###############################################################################
# Now we add the k-space maps together in order to utilize the redundancies ot our advantage
data_dc = np.fft.fftshift(d.dir_dataSharing(k, np.fft.fftshift(data_b1, axes=(-2,-1)), dirs=dirs,
                          origDataSize=N[-2:]), axes=(-2,-1))
im_dc = np.zeros(data_dc.shape, dtype='complex')
# And make the images
for i in range(ndirs):
    im_dc[i,:,:] = tf.ifft2c(data_dc[i,:,:],ph_b1[i,:,:])
    

# A way to quickly plot the different direction cases
if 0:
    for i in range(ndirs):
        ims = [im_b0_avg, im_b1_full[i,:,:], im_b1_scan[i,:,:], im_dc[i,:,:]]
        titles = ['b0 Average', 'im_full dir' + str(i), 'im_scan dir' + str(i), 'im_dc dir' + str(i)]
        vis.figSubplots(ims,clim=(minval,maxval),titles=titles)
        saveFig.save('/micehome/asalerno/Documents/pyDirectionCompSense/brainData/DTI/26apr16/33per/imdc/im_b1_comps_dir'+str(i))
    

im_sp = im_dc.copy().reshape(N)

# Optimization algortihm -- this is where everything culminates together
#M, dIM, Ause, inds = dirInfo

for i in range(len(TV)):
    args = (N, TV[i], XFM[i], data_b1, k, strtag, ph_b1, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
    im_result = opt.minimize(optfun, im_dc, args=args, method=method, jac=derivative_fun,
                            options={'maxiter': ItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': alpha_0, 'c': c, 'xtol': xtol[i], 'TVWeight': TV[i], 'XFMWeight': XFM[i], 'N': N})
        
    if np.any(np.isnan(im_result['x'])):
        print('Some nan''s found. Dropping TV and XFM values')
    else:
        im_dc = im_result['x'].reshape(N)
        alpha_k = im_result['alpha_k']
        

im_res = im_dc

