#Imports
from __future__ import division
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import os.path
os.chdir('/home/asalerno/Documents/pyDirectionCompSense/') # Change this to the directory that you're saving the work in
import transforms as tf
import scipy.ndimage.filters
import grads
import sampling as samp
import direction as d
#from scipy import optimize as opt
import optimize as opt
import scipy.optimize as spopt
plt.rcParams['image.cmap'] = 'gray'
from recon_CS import *

# Initialization variables
filename = '/home/asalerno/Documents/pyDirectionCompSense/data/SheppLogan256.npy'
strtag = ['spatial','spatial']
TVWeight = 0.01
XFMWeight = 0.01
dirWeight = 0
#DirType = 2
ItnLim = 150
epsilon = 1e-6
l1smooth = 1e-15
xfmNorm = 1
wavelet = 'db1'
mode = 'per'
method = 'CG'
dirFile = None
nmins = None
dirs = None
M = None

np.random.seed(2000)

#im = np.zeros([8,8]);
#im[3:5,3:5] = 1;

im=np.load(filename)

N = np.array(im.shape) #image Size
tupleN = tuple(N)
pctg = 0.25 # undersampling factor
P = 5 # Variable density polymonial degree
#ph = tf.matlab_style_gauss2D(im,shape=(5,5));
ph=np.ones(im.shape,complex)


# Generate the PDF for the sampling case -- note that this type is only used in non-directionally biased cases.
pdf = samp.genPDF(N,P,pctg,radius = 0.3,cyl=[0]) # Currently not working properly for the cylindrical case -- can fix at home
# Set the sampling pattern -- checked and this gives the right percentage
k = samp.genSampling(pdf,50,2)[0].astype(int)

# Here is where we build the undersampled data
data = np.fft.ifftshift(k)*tf.fft2c(im,ph=ph)
#ph = phase_Calculation(im,is_kspace = False)
#data = np.fft.ifftshift(np.fft.fftshift(data)*ph.conj());

# IMAGE from the "scanner data"
im_scan = tf.ifft2c(data,ph=ph)

# Primary first guess. What we're using for now. Density corrected
im_dc = tf.ifft2c(data/np.fft.ifftshift(pdf),ph=ph).real.flatten().copy()

# Optimization algortihm -- this is where everything culminates together
a=10.0
args = (N,TVWeight,XFMWeight,data,k,strtag,ph,dirWeight,dirs,M,nmins,wavelet,mode,a)
im_result = spopt.minimize(optfun, im_dc, args = args,method=method,jac=derivative_fun,options={'maxiter':ItnLim,'gtol':epsilon,'disp':1})
im_res = im_result['x'].reshape(N);

plt.imshow(im,interpolation='nearest')
plt.figure(2); plt.imshow(np.reshape(im_dc,(N)),interpolation='nearest')
plt.figure(3); plt.imshow(im_res,interpolation='nearest')
plt.show()