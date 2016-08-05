#!/usr/bin/env python -tt
#
#
# sampling.py
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
import scipy as sp
import sys
import glob
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import matplotlib as mpl
import os.path

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

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    
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
        img_sz_hold = cyl[1:]
        cir = True
        if np.all(img_sz_hold == img_sz):
            zpad_mat = False
        else:
            zpad_mat = True
    else:
        img_sz_hold = img_sz
        zpad_mat = False;
        outcyl = None;

    # If the size vector is only one value, add on another value
    if len(img_sz_hold) == 1:
        img_sz_hold = [img_sz_hold, 1]


    # How many of the datapoints do we have to look at?
    sx = img_sz_hold[0]
    sy = img_sz_hold[1]
    PCTG = int(np.floor(pctg*sx*sy))

    if np.sum(np.array(img_sz_hold == 1,dtype='int')) == 0: #2D case
        [x,y] = np.meshgrid(np.linspace(-1,1,sy),np.linspace(-1,1,sx))
        if l_norm == 1:
            r = abs(np.array([x,y])).max(0)
        elif l_norm == 2:
            r = np.sqrt(x**2 + y**2)
            # Check if the data is cyl acquisition -- if so, make sure outside datapoints don't get chosen by setting r = 1
            if cyl[0]:
                outcyl = np.where(r > 1)
                r[outcyl] = 1
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
        if outcyl:
            pdf[outcyl] = 0
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
        plt.imshow(pdf)
        
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
        while abs(np.sum(tmp) - K) > tol:
            tmp = np.random.random(pdf.shape) < pdf
            
        TMP = np.fft.ifft2(tmp/(pdf+EPS))
        if np.max(abs(TMP[1:])) < minIntr:
            minIntr = np.max(abs(TMP[1:]))
            minIntrVec = tmp;
            
        stat.append(np.max(abs(TMP[1:])))
        
    actpctg = np.sum(minIntrVec)/minIntrVec.size
    return minIntrVec, actpctg

def genSamplingDir(img_sz = [256,256],
                dirFile = 'GradientVectorMag.txt',
                pctg = 0.25,
                cyl = [0],
                radius = 0.1,
                nmins = 5,
                endSize = [256,256],
                engfile = None):
    
    import itertools
    dirs = np.genfromtxt(dirFile, delimiter = '\t')
    n = int(dirs.shape[0])
    r = np.zeros([n,n])

    for i in xrange(n):
        if dirs[i,2] < 0:
            dirs[i,:] = -dirs[i,:]
            
    for i in xrange(n):
        for j in xrange(n):
            r[i,j] = np.sqrt(np.sum((dirs[i,:] - dirs[j,:])**2))
            r[i,j] = min(np.sqrt(np.sum((-dirs[i,:] - dirs[j,:])**2)),r[i,j])

    invR = 1/(r+EPS)

    # Find all of the possible combinations of directions
    k = int(np.floor(n*pctg))
    combs = np.array(list(itertools.combinations(range(1,n+1),k)))
    vecs = np.array(list(itertools.combinations(range(1,k+1),2)))
    engStart = np.zeros([combs.shape[0]])

    # Run the "Potential energy" for each of the combinations
    if 'engFile' not in locals():
        for i in xrange(combs.shape[0]):
            for j in xrange(vecs.shape[0]):
                engStart[i] = engStart[i] + invR[combs[i,vecs[j,0]-1]-1,combs[i,vecs[j,1]-1]-1]
    else:
        engStart = np.load(engFile)
        # npy file
    
    # Build the best cases of trying to get the vectors together
    ind = engStart.argsort()
    eng = engStart[ind]
    vecsind = combs[ind,]
    locs = np.zeros([n,nmins])
    vecsMin = np.zeros([k,n*nmins])
    
    # Look for the locations where the indicies exist first (that is, they are the smallest)
    # and input those as the vectors we want to use
    for i in range(n):
        locs[i,] = np.array(np.where(vecsind == i+1))[0,0:nmins]
        vecsMin[...,nmins*i:nmins*(i+1)] = vecsind[locs[i,].astype(int),...].T-1
    
    vecsMin = unique_rows(vecsMin.T).astype(int)
    amts = np.zeros(n)
    
    for i in xrange(n):
        amts[i] = vecsMin[vecsMin == i].size
        
    srt = amts.argsort()
    cts = amts[srt]
    
    qEng = np.percentile(eng,20)
    
    # if theres a huge difference, tack more of the lower counts on, but make sure that we aren't hitting too high energy sets
    
    vecsUnique,vecs_idx = np.unique(vecsind,return_index = True)
    
    while cts[-1]/cts[0] >= 1.1:
        srt_hold = np.reshape(srt.copy(),(1,len(srt)))[0,:k]+1
        srt_hold.sort()
        # We need to add one here as index and direction number differ by a value of one
        indx = np.where(np.all(srt_hold == vecsind,axis = 1)) 
        
        if eng[indx] < qEng:
            vecsMin = np.vstack([vecsMin,srt_hold])
        else:
            while eng[indx] >= qEng:
                arr = np.zeros(k)
                cnt = 0
                while not np.all(arr == 0):
                    
                    st = np.ceil(n*np.random.random(1))
                    
                    if not np.any(arr == st-1):
                        arr[cnt] = st-1;
                        
                    
                arr_hold = np.reshape(arr(),(1,len(srt)))[0,:k]+1
                arr_hold.sort()
                indx = np.where(np.all(arr_hold == vecsind,axis = 1))
            
            vecsMin = np.vstack([vecsMin,arr_hold])
            for i in xrange(30):
                amts[i] = vecsMin[vecsMin == i].size
            srt = amts.argsort()
            cts = amts[srt]
            
    for i in xrange(30):
        amts[i] = vecsMin[vecsMin == i].size
    
    # Now we have to finally get the sampling pattern from this!
    
    [x,y] = np.meshgrid(np.linspace(-1,1,img_sz[1]),np.linspace(-1,1,img_sz[0]))
    r = np.sqrt(x**2 + y**2)
    
    if not cyl:
        r = r/np.max(abs(r))
        
    [rows,cols] = np.where(r <= 1) and np.where(r > radius)
    [rx,ry] = np.where(r <= radius)
    
    samp = np.zeros(hstack([img_sz,n]))
    nSets = np.hstack([vecsMin.size, 1])
    
    for i in xrange(rows):
        val = np.ceil(nSets*np.random.random(1))
        choice = vecsMin[val,]
        samp[rows[i],cols[i],choice] = 1
        
    for i in xrange(rx.size):
        samp[rx[i],ry[i],:] = 1
        
    if endSize.shape != img_sz:
        samp_final = np.zeros(np.hstack([endSize,n]))
        
        for i in xrange(n):
            samp_final[...,...,i] = np.resize(zpad(samp[...,...,i].flat,endSize),np.hstack([endSize,1]))
        
        samp = samp_final
    
    return samp
    