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
wavelet = 'db4'
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

imvals = np.concatenate([im_dc.reshape([N[0],N[1],1]),0*im_dc.reshape([N[0],N[1],1]),im.reshape([N[0],N[1],1])],axis=2)
x0sty = ['im_dc','zeros','im']
sty = ['TVOnly','XFMOnly-nonTanh','TVandXFM']
#----------------------------------------------------------
# Here is where the important stuff starts!
#----------------------------------------------------------
XFMWeight = 0.
TVWeight = 0.
for i in xrange(3):
    if i == 0:
        XFMWeight = 0.01
        TVWeight = 0.
    elif i == 1:
        XFMWeight = 0.
        TVWeight = 0.01
    elif i == 2:
        XFMWeight = 0.01
        TVWeight = 0.01
    
    for j in xrange(3):
        args = (N,TVWeight,XFMWeight,data,k,strtag,ph,dirWeight,dirs,M,nmins,wavelet,mode,a)
        im_result = opt.minimize(optfun, imvals[:,:,j].flatten(), args = args, method=method, jac=derivative_fun, options={'maxiter':ItnLim,'gtol':0.01,'disp':1,'alpha_0':0.5,'c':0.6,'xtol':5e-3})
        im_res = im_result['x'].reshape(N);
        
        np.save('simpleBoxData/' + sty[i] + '/SL-' + x0sty[j] + '-result.npy',im_res)
        
        plt.figure(2); plt.imshow(np.reshape(im_vals[:,:,j],(N)),interpolation='nearest',clim=(0,1))
        plt.title('Original Image - ' + x0sty[j])
        plt.savefig('simpleBoxData/' + sty[i] + '/SL-x0-' + x0sty[j] + '.png')
        plt.figure(3); plt.imshow(im_res,interpolation='nearest',clim=(0,1))
        plt.title(x0sty[j] + ' as x0 Final Result')
        plt.savefig('simpleBoxData/' + sty[i] + '/' + 'SL-final-' + x0sty[j] + '.png')