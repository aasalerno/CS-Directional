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
import grads
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

def runCSAlgorithm(fromfid=False,
                   filename='/home/asalerno/Documents/pyDirectionCompSense/brainData/P14/data/fullySampledBrain.npy',
                   sliceChoice=150,
                   strtag = ['','spatial', 'spatial'],
                   xtol = [1e-2, 1e-3, 5e-4, 5e-4],
                   TV = [0.01,0.005, 0.002, 0.001],
                   XFM = [0.01,.005, 0.002, 0.001],
                   dirWeight=0,
                   pctg=0.25,
                   radius=0.2,
                   P=2,
                   wavelet='db4',
                   mode='per',
                   method='CG',
                   ItnLim=30,
                   lineSearchItnLim=30,
                   alpha_0=0.6,
                   c=0.6,
                   a=10.0,
                   dirFile = None,
                   nmins = None,
                   dirs = None,
                   M = None,
                   dirInfo = [None]*4,
                   saveNpy=False,
                   saveNpyFile=None,
                   saveImsPng=False,
                   saveImsPngFile=None,
                   saveImDiffPng=False,
                   saveImDiffPngFile=None):
    
    if fromfid==True:
        inputdirectory=filename[0]
        petable=filename[1]
        fullImData = rff.getDataFromFID(petable,inputdirectory,2)[0,:,:,:]
        fullImData = fullImData/np.max(abs(fullImData))
        im = fullImData[:,:,sliceChoice]
    else:
        im = np.load(filename)[sliceChoice,:,:]
        
    N = np.array(im.shape)  # image Size

    pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=np.hstack([1, N[-2:]]), style='mult')
    k = samp.genSampling(pdf, 50, 2)[0].astype(int)
    if len(N) == 2:
        N = np.hstack([1, N])
        k = k.reshape(N)
        im = im.reshape(N)
    elif (len(N) == 3) and ('dir' not in strtag):
        k = k.reshape(np.hstack([1,N[-2:]])).repeat(N[0],0)

    ph_ones = np.ones(N[-2:], complex)
    ph_scan = np.zeros(N, complex)
    data = np.zeros(N,complex)
    im_dc = np.zeros(N,complex)
    im_scan = np.zeros(N,complex)
    for i in range(N[0]):
        data[i,:,:] = np.fft.ifftshift(k[i,:,:]) * tf.fft2c(im[i,:,:], ph=ph_ones)

        # IMAGE from the "scanner data"
        im_scan_wph = tf.ifft2c(data[i,:,:], ph=ph_ones)
        ph_scan[i,:,:] = tf.matlab_style_gauss2D(im_scan_wph,shape=(5,5))
        ph_scan[i,:,:] = np.exp(1j*ph_scan[i,:,:])
        im_scan[i,:,:] = tf.ifft2c(data[i,:,:], ph=ph_scan[i,:,:])
        #im_lr = samp.loRes(im,pctg)

    pdfDiv = pdf
    pdfZeros = np.where(pdf==0)
    pdfDiv[pdfZeros] = 1
    im_scan_imag = im_scan.imag
    im_scan = im_scan.real

    N_im = N
    hld, dims, dimOpt, dimLenOpt = tf.wt(im_scan[0],wavelet,mode)
    N = np.hstack([N_im[0], hld.shape])

    w_scan = np.zeros(N)
    w_full = np.zeros(N)
    im_dc = np.zeros(N_im)
    w_dc = np.zeros(N)

    for i in range(N[0]):
        w_scan[i,:,:] = tf.wt(im_scan[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)[0]
        w_full[i,:,:] = tf.wt(abs(im[i,:,:]),wavelet,mode,dims,dimOpt,dimLenOpt)[0]

        im_dc[i,:,:] = tf.ifft2c(data[i,:,:] / np.fft.ifftshift(pdfDiv), ph=ph_scan[i,:,:]).real.copy()
        w_dc[i,:,:] = tf.wt(im_dc,wavelet,mode,dims,dimOpt,dimLenOpt)[0]

    w_dc = w_dc.flatten()
    im_sp = im_dc.copy().reshape(N_im)
    minval = np.min(abs(im))
    maxval = np.max(abs(im))
    data = np.ascontiguousarray(data)

    imdcs = [im_dc,np.zeros(N_im),np.ones(N_im),np.random.randn(np.prod(N_im)).reshape(N_im)]
    mets = ['Density Corrected','Zeros','Ones','Random']
    wdcs = []
    for i in range(len(imdcs)):
        wdcs.append(tf.wt(imdcs[i][0],wavelet,mode,dims,dimOpt,dimLenOpt)[0].reshape(N))

    ims = []
    #print('Starting the CS Algorithm')
    for kk in range(len(wdcs)):
        w_dc = wdcs[kk]
        print(mets[kk])
        for i in range(len(TV)):
            args = (N, N_im, dims, dimOpt, dimLenOpt, TV[i], XFM[i], data, k, strtag, ph_scan, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
            w_result = opt.minimize(f, w_dc, args=args, method=method, jac=df, 
                                        options={'maxiter': ItnLim, 'lineSearchItnLim': lineSearchItnLim, 'gtol': 0.01, 'disp': 1, 'alpha_0': alpha_0, 'c': c, 'xtol': xtol[i], 'TVWeight': TV[i], 'XFMWeight': XFM[i], 'N': N})
            if np.any(np.isnan(w_result['x'])):
                print('Some nan''s found. Dropping TV and XFM values')
            elif w_result['status'] != 0:
                print('TV and XFM values too high -- no solution found. Dropping...')
            else:
                w_dc = w_result['x']
                
        w_res = w_dc.reshape(N)
        im_res = np.zeros(N_im)
        for i in xrange(N[0]):
            im_res[i,:,:] = tf.iwt(w_res[i,:,:],wavelet,mode,dims,dimOpt,dimLenOpt)
        ims.append(im_res)
    
    if saveNpy:
        if saveNpyFile is None:
            np.save('./holdSave_im_res_' + str(int(pctg*100)) + 'p_all_SP',ims)
        else:
            np.save(saveNpyFile,ims)
    
    if saveImsPng:
        vis.figSubplots(ims,titles=mets,clims=(minval,maxval),colorbar=True)
        if saveImsPngFile is None:
            saveFig.save('./holdSave_ims_' + str(int(pctg*100)) + 'p_all_SP')
        else:
            saveFig.save(saveImsPngFile)
    
    if saveImDiffPng:
        imdiffs, clims = vis.imDiff(ims)
        diffMets = ['DC-Zeros','DC-Ones','DC-Random','Zeros-Ones','Zeros-Random','Ones-Random']
        vis.figSubplots(imdiffs,titles=diffMets,clims=clims,colorbar=True)
        if saveImDiffPngFile is None:
            saveFig.save('./holdSave_im_diffs_' + str(int(pctg*100)) + 'p_all_SP')
        else:
            saveFig.save(saveImDiffPngFile)
            
