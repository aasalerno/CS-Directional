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

def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper


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

im = np.zeros([8,8]);
im[3:5,3:5] = 1;

#im=np.load(filename)

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


# Get things set to test alpha values
f = optfun
fprime = derivative_fun
x0 = np.asarray(im_dc).flatten()

func_calls, f = wrap_function(f, args)
grad_calls, myfprime = wrap_function(fprime, args) # Wraps the derivative function
gfk = myfprime(x0)
k = 0
xk = x0
old_fval = f(xk)
pk = -gfk # Here is where the -1 is applied -- thus I shouldn't apply it in mine
newargs = args

gradient = True

gval = [gfk]
gc = [0]
fc = [0]

def phi(s):
    fc[0] += 1
    return f(xk + s*pk, args)

def derphi(s):
    gval[0] = fprime(xk + s*pk, *newargs)
    if gradient:
        gc[0] += 1
    else:
        fc[0] += len(xk) + 1
    return np.dot(gval[0], pk)

derphi0 = np.dot(gfk, pk)

def alpha_check(s):                                                                           
    return optfun(xk+s*pk,N,TVWeight,XFMWeight,data,k,strtag,ph,dirWeight,dirs,M,nmins,wavelet,mode,a)
    

s = np.logspace(-8,1,1000)
y = np.zeros(len(s))

for i in xrange(len(s)):
    y[i] = alpha_check(s[i])

plt.plot(s,y,'.')
plt.title('Objective Function Values')
plt.xlabel('alpha')
plt.ylabel('phi(alpha)')
plt.show()