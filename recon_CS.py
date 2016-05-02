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
import pyminc.volumes.factory
import numpy as np 
import sys
import glob
import matplotlib.pyplot as plt

EPS = np.finfo(float).eps

def indata(inFolder):
    '''
    This code reads in data from mnc files in order to be worked on via the code. 
    Reads the data in and outputs it as a list
    '''
    us_data = [] 

    # get the names of the input files from the specified folder
    filenames = glob.glob(inFolder + '/*.mnc')

    # Put the data in a large dataset to be worked with
    for files in filenames:
        cnt = 0
        us_data.append(pyminc.volumes.factory.volumeFromFile(files))
        cnt += 1 
    return us_data


    
def zpad(res_sz,orig_data):
    res_sz = np.array(res_sz)
    orig_sz = np.array(orig_data.shape)
    padval = np.ceil((res_sz-orig_sz)/2)
    res = np.pad(orig_data,([padval[0],padval[0]],[padval[1],padval[1]]),mode='constant')
    return res
    
def genPDF(img_sz,
        p,
        pctg,
        l_norm = 2,
        radius = 0,
        cyl = [0],
        disp = 0):
    """
    Generates a Probability Density Function for Pseudo-undersampling (and potentially for use with the scanner after the fact. 
    
    This uses a variable density undersampling, allowing for more (or less) data in the centre
    
    Input:
        [int list] img_sz - The size of the input dataset
        [int]         p - polynomial power (1/r^p)
        [float]    pctg - Sampling factor -- how much data we want to collect
        [int]    l_norm - L1 or L2 distance measure
        [float]  radius - fully sampled centre radius
        [int list]  cyl - Is it cylindrical data acquisition? 
        [bool]     disp - Do you want it displayed or not?
    
    Output:
        2D[float] pdf - The probability density function
        
    Based on Lustig's work from 2007
    """
    
    minval = 0.0
    maxval = 1.0
    val = 0.5
    
    # Check if we're doing cylindrical data, if so, change img_sz and note that we need to zero pad
    if cyl[0] == 1:
        img_sz_hold = [cyl[1], cyl[2]]
        cir = True;
        
        if img_sz_hold != img_sz:
            zpad_mat = True;
        else:
            zpad_mat = False;
            
    else:
        img_sz_hold = img_sz
        zpad_mat = False;
    
    # If the size vector is only one value, add on another value
    if len(img_sz_hold) == 1:
        img_sz_hold = [img_sz_hold, 1]
    
    
    # How many of the datapoints do we have to look at?
    sx = img_sz_hold[0]
    sy = img_sz_hold[1]
    PCTG = int(np.floor(pctg*sx*sy))
    
    if sum(img_sz_hold == 1) == 0: #2D case
        [x,y] = np.meshgrid(np.linspace(-1,1,sy),np.linspace(-1,1,sx))
        if l_norm == 1:
            r = abs(np.array([x,y])).max(0)
        elif l_norm == 2:
            r = np.sqrt(x**2 + y**2)
            # Check if the data is cyl acquisition -- if so, make sure outside datapoints don't get chosen by setting r = 1
            if cyl[0]:
                r[np.where(r > 1)] = 1
            else:
                r = r/r.max()
    else: #1D
        r = abs(np.linspace(-1,1,max([sx,sy])))
        
    idx = np.where(r < radius)
    pdf = pdf = (1-r)**p
    pdf[idx] = 1
    
    if np.floor(np.sum(pdf))>PCTG:
        raise NameError('Polynomial too low. Would need to undersample DC. Increase P')
    
    # Bisect the data to get the proper PDF values to allow for the optimal sampling pattern generation
    while(1):
        val = minval/2 + maxval/2;
        pdf = (1-r)**p + val
        pdf[np.where(pdf > 1)] = 1
        pdf[idx] = 1
        N = np.floor(np.sum(pdf));
        if N > PCTG:
            maxval = val
        elif N < PCTG:
            minval = val;
        else:
            break;
    
    if zpad_mat:
        pdf = zpad(img_sz,pdf)
    
    if disp:
        plt.figure
        plt.subplot(211) 
        plt.imshow(pdf)
        plt.subplot(212)
        if np.sum(img_sz_hold == 1) == 0:
            plt.plot(pdf[len(pdf)/2:,])
        else:
            plt.subplot(212)
            plt.plot(pdf)
        
        plt.show()
        
    return pdf
    
def genSampling(pdf, n_iter, tol):
    '''
    Quick Monte-Carlo Algorithm to generate a sampling pattern, to try and have minimal peak interference. Number of samples is np.sum(pdf) +/- tol.
    
    Inputs:
    [np.array]  pdf - Probability density function to choose from
    [float]  n_iter - number of attempts
    [int]       tol - Deviation from the desired number of samples
    
    Outputs:
    [bool array]  mask - sampling pattern
    [float]  actpctg - actual undersampling factor
    
    This code is ported from Michael Lustig 2007
    '''
    
    pdf[np.where(pdf > 1)] = 1
    K = np.sum(pdf)
    
    minIntr = 1e99;
    minIntrVec = np.zeros(pdf.shape)
    stat = []
    
    for n in xrange(n_iter):
        tmp = np.zeros(pdf.shape)
        while abs(np.sum(tmp - K)) > tol:
            tmp = np.matlib.rand(pdf.shape) < pdf
            
        TMP = np.fft.ifft2(tmp/(pdf+EPS))
        if max(abs(TMP[1:])) < minIntr:
            minIntr = max(abs(TMP[1:]))
            minIntrVec = tmp;
            
        stat.append(max(abs(TMP[1:])))
        
    actpctg = np.sum(minIntrVec)/minIntrVec.size
    return minIntrVec, actpctg

