#!/usr/bin/env python -tt
#
#
# recon_CS.py
#
#
# We start with the data from the scanner. The inputs are:
#       - inFile (String) -- Location of the data
#                         -- Direct to a folder where all the data is
#       - 
#

from __future__ import division
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import os.path
import transforms as tf
import scipy.ndimage.filters
import grads
import sampling as samp
import direction as d
#from scipy import optimize as opt
import optimize as opt
plt.rcParams['image.cmap'] = 'gray'

EPS = np.finfo(float).eps

def derivative_fun(x,N,lam1,lam2,data,k,strtag,ph,dirWeight = 0,dirs = None,M = None,nmins = 0, scaling_factor = 4,L = 2):
    '''
    This is the function that we're going to be optimizing via the scipy optimization pack. This is the function that represents Compressed Sensing
    '''
    gObj = grads.gObj(x,N,ph,data,k) # Calculate the obj function
    gTV = grads.gTV(x,N,strtag,dirWeight,dirs,nmins,M) # Calculate the TV gradient
    gXFM = tf.ixfm(grads.gXFM(tf.xfm(x),N)) # Calculate the wavelet gradient
    x.shape = (x.size,)
    import pdb; pdb.set_trace();
    return (gObj + lam1*gTV + lam2*gXFM).flatten() # Export the flattened array

def optfun(x,N,lam1,lam2,data,k,strtag,ph,dirWeight = 0,dirs = None,M = None,nmins = 0,scaling_factor = 4,L = 2):
    '''
    This is the optimization function that we're trying to optimize. We are optimizing x here, and testing it within the funcitons that we want, as called by the functions that we've created
    '''
    data.shape = N
    x.shape = N
    obj_data = tf.ifft2c(data - np.fft.fftshift(k)*tf.fft2c(x,ph),ph)
    obj = np.sqrt(np.sum(obj_data*obj_data.conj())) #L2 Norm
    tv = np.sum(abs(tf.TV(x,N,strtag,dirWeight,dirs,nmins,M))) #L1 Norm
    xfm = np.sum(abs(tf.xfm(x,scaling_factor,L))) #L1 Norm
    x.shape = (x.size,) # Not the most efficient way to do this, but we need the shape to reset.
    data.shape = (data.size,)
    print('obj: %.2f' % abs(obj))
    print('tv: %.2f' % abs(lam1*tv))
    print('xfm: %.2f' % abs(lam2*xfm))
    return abs(obj + lam1*tv + lam2*xfm)

def phase_Calculation(data,is_kspace = 0,is_fftshifted = 0):
    
    if is_kspace:
        data = tf.ifft2c(data)
        if is_fftshifted:
            data = np.ifftshift(data)
        
    F = tf.matlab_style_gauss2D(shape=(5,5),sigma=2)
    filtdata = sp.ndimage.uniform_filter(data,size=5)
    return filtdata.conj()/(abs(filtdata)+EPS)

    
def gDir_lookupTable(inds):
    '''
    THe lookup table takes the indicies in, and creates a lookup table based on where a value occurs within the inds matrix. It makes all of the values in the row of the counter -1 because that is where the subtraction is happening, and +1 everywhere else.
    '''
    rows,cols = inds.shape
    lookupTable = np.zeros([rows,rows,cols])
        
    for i in xrange(rows):
        lt = np.zeros([rows,cols])
        lt[i,:] = -1
        x,y = np.where(inds==i)
        for j in xrange(x.size):
            lt[x[j],y[j]] = 1
        lookupTable[i,:,:] = lt
    
    return lookupTable
    
    
def recon_CS(filename = 
             '/home/asalerno/Documents/pyDirectionCompSense/data/SheppLogan256.npy', #'DTI_Phantom-SNR1000.npy',
             strtag = ['spatial','spatial'],
             TVWeight = 0.01,
             XFMWeight = 0.01,
             dirWeight = 0,
             #DirType = 2,
             ItnLim = 150,
             epsilon = 0.1,
             l1smooth = 1e-15,
             xfmNorm = 1,
             scaling_factor = 4,
             L = 2,
             method = 'CG',
             dirFile = None,
             nmins = None): # = 4):
             
    # Here we set the seed of the random number generator in order to ensure that
    # we get the same map when we do our undersampling
    np.random.seed(2000)
    
    # Load in the image
    im = np.load(filename)
    # im = im + 0.1*(np.random.normal(size=[256,256]) + 1j*np.random.normal(size = [256,256])) # For the simplest case right now
    
    # Set the string tags for the dimensions to be all lower case for comparison in TV operator
    for i in range(len(strtag)):
        strtag[i] = strtag[i].lower()

    # Set up some variables for future use.
    N = np.array(im.shape) #image Size
    tupleN = tuple(N)
    pctg = 0.25 # undersampling factor
    P = 5 # Variable density polymonial degree
    ph = tf.matlab_style_gauss2D(im,shape=(5,5));
    
    # Generate the PDF for the sampling case -- note that this type is only used in non-directionally biased cases.
    pdf = samp.genPDF(N,P,pctg,radius = 0.1,cyl=[0]) # Currently not working properly for the cylindrical case -- can fix at home
    # Set the sampling pattern -- checked and this gives the right percentage
    k = samp.genSampling(pdf,10,60)[0].astype(int)
    
    # Diffusion information that we need
    if dirFile:
        dirs = np.loadtxt(dirFile)
        dirInfo = d.calc_Mid_Matrix(dirs,nmins=4)
    else:
        dirs = None
        dirInfo = None
    
    # Here is where we build the undersampled data
    data = np.fft.ifftshift(k)*tf.fft2c(im,ph=ph)
    #ph = phase_Calculation(im,is_kspace = False)
    #data = np.fft.ifftshift(np.fft.fftshift(data)*ph.conj());
    
    # IMAGE from the "scanner data"
    im_scan = tf.ifft2c(data,ph=ph)
    
    # Primary first guess. What we're using for now. Density corrected
    im_dc = tf.ifft2c(data/np.fft.ifftshift(pdf),ph=ph).flatten().copy()
    
    # Optimization algortihm -- this is where everything culminates together
    args = (N,TVWeight,XFMWeight,data,k,strtag,ph,dirWeight,dirs,dirInfo,nmins,scaling_factor,L)
    im_result = opt.minimize(optfun, im_dc, args = args,method=method,jac=derivative_fun,options={'maxiter':ItnLim,'gtol':epsilon,'disp':1})
    im_res = im_result['x'].reshape(256,256);
    
    # im_result gives us a lot of information, what we really need is ['x'] reshaped to the required image size --
    #import pdb; pdb.set_trace();
    # return {'im_result': im_result, 'im_dc': im_dc, 'im_scan': im_scan}
    return im_result
    
if __name__ == "__main__":
    print(recon_CS())