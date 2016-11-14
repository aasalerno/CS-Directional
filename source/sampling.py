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
import scipy.ndimage as ndimage
import sys
import glob
#import matplotlib.pyplot as plt
#plt.rcParams['image.cmap'] = 'gray'
#import matplotlib.pyplot as plt
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
    
def zpad(orig_data,res_sz):
    res_sz = np.array(res_sz)
    orig_sz = np.array(orig_data.shape)
    padval = np.ceil((res_sz-orig_sz)/2)
    res = np.pad(orig_data,([int(padval[0]),int(padval[0])],[int(padval[1]),int(padval[1])]),mode='constant')
    return res
    
def genPDF(img_sz,
        p,
        pctg,
        l_norm = 2,
        radius = 0,
        cyl = [0],
        disp = 0,
        style='add'):
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
        if np.logical_and(img_sz_hold[0] == img_sz[0], img_sz_hold[1] == img_sz[1]):
            zpad_mat = False
        else:
            zpad_mat = True
    else:
        cir = False
        img_sz_hold = img_sz
        zpad_mat = False
        outcyl = None

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
            if cir:
                outcyl = np.where(r > 1)
                r[outcyl] = 1
            else:
                r = r/r.max()
    else: #1D
        r = abs(np.linspace(-1,1,max([sx,sy])))
        
    idx = np.where(r < radius)
    pdf = (1-r)**p
    pdf[idx] = 1

    if len(idx[0]) > PCTG/3:
        raise NameError('Radius is too big! Rerun with smaller central radius.')

    # Bisect the data to get the proper PDF values to allow for the optimal sampling pattern generation
    if p==0:
        val = PCTG - len(idx[0])
        pdf = PCTG/(pdf.size-len(idx))*np.ones(pdf.shape)
        pdf[idx] = 1
    else:
        if style=='mult':
            #maxPx = sy/2
            #maxPy = sx/2
            alpha = 10
            maxPx = 10
            maxPy = 10
            c = 0.90
            while alpha>1:
                maxPx = c*maxPx
                maxPy = c*maxPy
                [px,py] = np.meshgrid(np.linspace(-maxPx,maxPx,sy),np.linspace(-maxPy,maxPy,sx))
                rpx = np.sqrt(px**2+py**2)
                r0 = rpx[idx[0][0],idx[1][0]]
                rpx = rpx - r0 + 1
                rpx[idx] = 1
                pdf = 1/(rpx**p)
                val = PCTG - len(idx[0])
                sumval = np.sum(pdf) - len(idx[0])
                alpha = val/sumval
                pdf = alpha*pdf
                pdf[idx] = 1
        else:
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
        if (img_sz[0] < img_sz_hold[0]) or (img_sz[1] < img_sz_hold[1]):
            pdf = zpad(pdf,img_sz)
        else:
            xdiff = int((img_sz[0] - img_sz_hold[0])/2)
            ydiff = int((img_sz[1] - img_sz_hold[1])/2)
            pdf = pdf[xdiff:-xdiff,ydiff:-ydiff]

    pdf = ndimage.filters.gaussian_filter(pdf,3)
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
                dirFile = '/home/asalerno/Documents/pyDirectionCompSense/GradientVectorMag.txt',
                pctg = 0.25,
                cyl = [0],
                radius = 0.1,
                nmins = 5,
                endSize = [256,256],
                engfile = None):
    
    import itertools
    # load the directions
    print('Loading Directions...')
    dirs = np.loadtxt(dirFile) 
    n = int(dirs.shape[0])
    r = np.zeros([n,n])

    # Push everything onto one half sphere
    #    for i in xrange(n):
    #        if dirs[i,2] < 0:
    #            dirs[i,:] = -dirs[i,:]

    print('Calculating Distances...')
    # Calculate the distance. Do it for both halves of the sphere
    for i in xrange(n):
        for j in xrange(n):
            r[i,j] = min(np.sqrt(np.sum((-dirs[i,:] - dirs[j,:])**2)),np.sqrt(np.sum((dirs[i,:] - dirs[j,:])**2)))

    invR = 1/(r+EPS)

    print('Finding all possible direction combinations...')
    # Find all of the possible combinations of directions
    k = int(np.floor(n*pctg)) # How many "directions" will have a point in k space
    combs = np.array(list(itertools.combinations(range(0,n),k))) # All the different vector combinations
    vecs = np.array(list(itertools.combinations(range(0,k),2))) # All the different combos that need to be checked for the energy
    engStart = np.zeros([combs.shape[0]]) # Initialization for speed of the energy
    
    print('Running PE Electrostatics system...')
    # Run the "Potential energy" for each of the combinations
    if 'engFile' not in locals():
        for i in xrange(combs.shape[0]):
            for j in xrange(vecs.shape[0]):
                engStart[i] = engStart[i] + invR[combs[i,vecs[j,0]],combs[i,vecs[j,1]]]
    else:
        engStart = np.load(engFile)
        # npy file
    
    print('Producing "best cases..."')
    # Build the best cases of trying to get the vectors together
    ind = engStart.argsort() # Sort from lowest energy (farthest apart) to highest
    eng = engStart[ind] # Again, sort
    vecsInd = combs[ind,] # This tells us the vectors that we're going to be using for our mins
    locs = np.zeros([n,nmins]) # This gives us the mins for our individual vectors
    vecsMin = np.zeros([k,n*nmins]) 

    # Look for the locations where the indicies exist first (that is, they are the smallest)
    # and input those as the vectors we want to use
    for i in range(n):
        locs[i,] = np.array(np.where(vecsInd == i))[0,0:nmins]
        vecsMin[:,nmins*i:nmins*(i+1)] = vecsInd[locs[i,].astype(int),:].T

    # Only keep those rows that are unique
    vecsMin = unique_rows(vecsMin.T).astype(int)
    amts = np.zeros(n)

    # Count how often each direction gets pulled
    for i in xrange(n):
        amts[i] = vecsMin[vecsMin == i].size
    srt = amts.argsort()
    cts = amts[srt]
    
    print('Check lowest 20%')
    # Make sure we only look at the lowest 20% of the energies
    qEng = np.percentile(eng,20)

    # if theres a huge difference, tack more of the lower counts on, but make sure that we aren't hitting too high energy sets

    #vecsUnique,vecs_idx = np.unique(vecsInd,return_index = True)
    
    while cts[-1]/cts[0] >= 1.25:
        #import pdb; pdb.set_trace()
        srt_hold = srt.copy().reshape(1,len(srt))[0,:k]
        srt_hold.sort()
        # We need to add one here as index and direction number differ by a value of one
        indx = np.where(np.all(srt_hold == vecsInd,axis = 1)) 
        #import pdb; pdb.set_trace();
        if eng[indx] < qEng:
            vecsMin = np.vstack([vecsMin,srt_hold])
        else:
            while eng[indx] >= qEng: # Take this if the bottom ones are too big!
                #import pdb; pdb.set_trace();
                arr = np.zeros(k) # Create a holder array
                cnt = 0 # Create an iterator
                while np.any(arr == 0):
                    st = np.ceil(n*np.random.random(1)) # A holder for the value
                    if not np.any(arr == st): # Make sure that value doesn't already exist in the holder array
                        arr[cnt] = st; # If it doesn't, add it
                        cnt += 1 # Move to the next location
                arr_hold = arr.copy().reshape((1,len(arr)))[0,:k]
                arr_hold.sort() # Sort it out. Making sure it didn't end up too long
                indx = np.where(np.all(arr_hold == vecsInd,axis = 1)) # find the index
                if eng[indx] < qEng: # Make sure we oly add it if the indx is low enough
                    vecsMin = np.vstack([vecsMin,arr_hold])
            
        for i in xrange(30):
            amts[i] = vecsMin[vecsMin == i].size
        srt = amts.argsort()
        cts = amts[srt]
    
    # Now we have to finally get the sampling pattern from this!
    
    print('Obtaining sampling pattern...')
    [x,y] = np.meshgrid(np.linspace(-1,1,img_sz[1]),np.linspace(-1,1,img_sz[0]))
    r = np.sqrt(x**2 + y**2)
    
    # If not cylindrical, we need to have vals < 1
    if not cyl[0]:
        print('Not cylindrical, so we need r<=1')
        r = r/np.max(abs(r))
        
    [rows,cols] = np.where((r <= 1).astype(int)*(r > radius).astype(int) == 1)
    [rx,ry] = np.where(r <= radius)
    
    # Create our sampling mask
    samp = np.zeros(np.hstack([n,img_sz]))
    nSets = np.hstack([vecsMin.shape, 1])
    
    # Start making random choices for the values that require them
    
    for i in xrange(len(rows)):
        val = np.floor(nSets[0]*np.random.random(1)).astype(int)
        choice = vecsMin[val,].astype(int)
        samp[choice,rows[i],cols[i]] = 1
        
    for i in xrange(len(rx)):
        samp[:,rx[i],ry[i]] = 1
        
    if endSize != img_sz:
        print('Zero padding...')
        samp_final = np.zeros(np.hstack([n,endSize]))
        
        for i in xrange(n):
            samp_hold = zpad(samp[i,:,:].reshape(img_sz),endSize)
            samp_final[i,:,:] = samp_hold.reshape(np.hstack([1,endSize]))
        
        samp = samp_final
    
    return samp

def radialHistogram(k,rmax=np.sqrt(2),bins=50):
    
    maxxy = (rmax**2)/2
    [x,y] = np.meshgrid(np.linspace(-maxxy,maxxy,k.shape[0]), np.linspace(-maxxy,maxxy,k.shape[1]))
    r = np.sqrt(x**2+y**2)
    r *= k
    cnts = plt.hist(r.flat,bins=bins)
    ymax = np.sort(cnts[0])[-2]*1.1
    plt.xlim(0,rmax)
    plt.ylim(0,ymax)
    plt.title('Radial Histogram')
    plt.xlabel('Radius')
    plt.ylabel('Counts')
    plt.show()
    
    