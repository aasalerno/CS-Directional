# Imports
from __future__ import division
import numpy as np
import scipy as sp
import os.path
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from sys import path as syspath
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/")
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/")
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/wavelet")
os.chdir('/home/asalerno/Documents/pyDirectionCompSense/')
import transforms as tf
import scipy.ndimage.filters
import gradWaveletMS as grads
import sampling as samp
import direction as d
import optimize as opt
import scipy.optimize as spopt
#import wavelet_MS_DC_TV_XFM_f_df as funs
import optimizationFunctions as funs
f = funs.objectiveFunction
df = funs.derivativeFunction
import read_from_fid as rff
import saveFig
from scipy.interpolate import RectBivariateSpline
from unwrap2d import *
import visualization as vis
from scipy.ndimage.filters import gaussian_filter
from time import localtime, strftime
#from pyminc.volumes.factory import *
np.random.seed(534)

#strftime("%Y-%m-%d %H:%M:%S", gmtime())


# Start with some files that we need -- note that we wouldn't use both a "filename" and inputdirectory
inputdirectory="/hpf/largeprojects/MICe/segan/exercise_irradiation/bruker_data/running_C/P14/20160607_124310_Running_C_1_1"
petable = "/hpf/largeprojects/MICe/bjnieman/Bruker_cyltests/cylbruker_nTRfeath18_294_294"
filename = '/home/asalerno/Documents/pyDirectionCompSense/brainData/P14/data/fullySampledBrain.npy'
#filename = '/home/asalerno/Documents/pyDirectionCompSense/phantom/imFull.npy'
dirFile = '/home/asalerno/Documents/pyDirectionCompSense/GradientVectorMag.txt'
#sliceChoice = np.arange(0,334)

# Initialization that we need for functions
strtag = ['','spatial', 'spatial']
dirWeight = 0
ItnLim = 30
lineSearchItnLim = 30
wavelet = 'db4'
mode = 'per'
method = 'CG'
kern = np.zeros([3,3,3,3])
for i in range(kern.shape[0]):
    if strtag[i] == 'spatial':
        kern[i,1,1,1] = -1

if strtag[0] == 'spatial':
    kern[0,2,1,1] = 1
if strtag[1] == 'spatial':
    kern[1,1,2,1] = 1
if strtag[2] == 'spatial':
    kern[2,1,1,2] = 1
                  
# Constants for the objective functions and the optimizer
alpha_0 = 0.1
c = 0.6
a = 10.0 # value used for the tanh argument instead of sign
xtol = [0.1, 1e-2, 1e-3, 5e-4, 5e-4]
TV = [0.001]#, 0.005, 0.002, 0.001, 0.001]
XFM = [0.001]#, 0.005, 0.002, 0.001, 0.001]
lam_trust = 0.5
phIter = 0

# Constants for your method of sampling
#--------------------------------------------------------------
# Note that for the future, this wouldn't be done as this would 
# be done in order to build our PE table 
#--------------------------------------------------------------
radius = 0.2
pft=False
pctg = 0.25
P = 2

# If we have directional data, then we need to pull it in!
if dirFile:
    nmins = 5
    dirs = np.loadtxt(dirFile)
    dirInfo = d.calc_Mid_Matrix(dirs,nmins)
    
# Load our data and get some of the information that we need!
im = np.load(filename)
#im = im[sliceChoice,:,:]
#im = im/np.max(abs(im))
minval = np.min(abs(im))
maxval = np.max(abs(im))

# SPECIAL CASES!!!!
#--------------------------------------------------------------
# Here is where we build our special cases -- for example, if
# we have data that we want to repeat a bunch of times
#--------------------------------------------------------------
#im = np.tile(im,(30,1,1))

#--------------------------------------------------------------

# Lets now check size, and we need our full size as well
N = np.array(im.shape)  # image Size
szFull = im.size

nSteps = 4
# This is in use if we want to alternate our values for TV etc
#if len(TV) < (nSteps+1):
    #for i in xrange(nSteps - len(TV) + 1):
        #TV.append(TV[-1])
        #XFM.append(XFM[-1])
        #xtol.append(xtol[-1])



# Here is now where we build our pdf and things like that
# SPECIAL CASES CAN HAVE EFFECTS HERE!!!!
#--------------------------------------------------------------
# If our special cases have an effect here, PLEASE put the
# original code within these two bars.

#pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=[1, N[-2], N[-1]], style='mult', pft=pft,ext=0.5)
#if pft:
    #print('Partial Fourier sampling method used')

#k = samp.genSampling(pdf, 50, 2)[0].astype(int)
#d.dirPDFSamp(N,P=2,pctg=0.25,radius=0.2,dirs=dirs,cyl=True,taper=0.25)

#--------------------------------------------------------------
#k = np.zeros([int(np.ceil(N[0]/dirs.shape[0])*dirs.shape[0]), 294, 294])
k = np.zeros([dirs.shape[0], 294, 294])
pdf = samp.genPDF(N[-2:], P, pctg, radius=radius, cyl=[1, N[-2], N[-1]], style='mult', pft=pft,ext=0.5)
k = d.dirPDFSamp([int(dirs.shape[0]), 294, 294], P=2, pctg=0.25, radius=0.2, dirs=dirs, cyl=True, taper=0.25)

k = [k[0] for _ in range(im.shape[0])]
k = np.stack(k,axis=0)
# SPECIAL CASE END

# Since our functions are built to work in 3D datasets, here we
# make sure that N and things are all in 3D
if len(N) == 2:
    N = np.hstack([1, N])
    k = k.reshape(N)
    im = im.reshape(N)
elif len(N) == 3:
    if k.ndim == 2:
        k = k.reshape(np.hstack([1,N[-2:]])).repeat(N[0],0)

    
# Now we initialize to build up "what we would get from the
# scanner" -- as well as our phase corrections
#ph_scan = np.zeros(N, complex)
#data = np.zeros(N,complex)
#dataFull = np.zeros(N,complex)

# We need to try to make this be as efficient and accurate as 
# possible. The beauty of this, is if we are using data that is
# anatomical, we can use the RO direction as well
# NOTE: Something that we can do later is make this estimation of
# phase inclue the RO direction, and then do a split later. This is 
# post-processing, but pre-CS
k = np.fft.fftshift(k, axes=(-2,-1))
     
ph_ones = np.ones(N, complex)
dataFull = tf.fft2c(im, ph=ph_ones,axes=(-2,-1))
data = k*dataFull
k = np.fft.fftshift(k, axes=(-2,-1))
#im_scan_wph = tf.ifft2c(data,ph=ph_ones)
#ph_scan = np.angle(gaussian_filter(im_scan_wph.real,0) +  1.j*gaussian_filter(im_scan_wph.imag,0))
#ph_scan = np.exp(1j*ph_scan)
#im_scan = tf.ifft2c(data,ph=ph_scan,sz=szFull)


# Now, we can use the PDF (for right now) to make our starting point
# NOTE: This won't be a viable method for data that we undersample
#       because we wont have a PDF -- or if we have uniformly undersampled
#       data, we need to come up with a method to have a good SP
pdfDiv = pdf.copy()
pdfZeros = np.where(pdf==0)
pdfDiv[pdfZeros] = 1


# Here, we look at the number of "steps" we want to do and step 
# up from there. The "steps" are chose based on the percentage that 
# we can sample and is based on the number of steps we can take.
x, y = np.meshgrid(np.linspace(-1,1,N[-1]),np.linspace(-1,1,N[-2]))
locs = (abs(x)<=radius) * (abs(y)<=radius)
minLoc = np.min(np.where(locs==True))
pctgSamp = np.zeros(minLoc+1)
for i in range(1,minLoc+1):
    kHld = k[0,i:-i,i:-i]
    pctgSamp[i] = np.sum(kHld)/kHld.size
pctgLocs = np.arange(1,nSteps+1)/(nSteps)
locSteps = np.zeros(nSteps)
locSteps[0] = minLoc

# Find the points where the values are as close as possible
for i in range(nSteps):
    locSteps[i] = np.argmin(abs(pctgLocs[i]-pctgSamp))
# Flip it here to make sure we're starting at the right point
locSteps = locSteps[::-1].astype(int)
locSteps = np.hstack([locSteps,0])


# We initialize a bunch of lists here for debugging
ims = []
stps = []
szStps = []
imStp = []
tvStps = []
w_stp = []
Ns = []
N_ims = []
dims = [] 
dimOpts = [] 
dimLenOpts = []
kMasks = []
ph_scans = []
dataSubs=[]


# ------------------------------------------------------ #
# -------------- MULTI-STEP OPTIMIZATIONS -------------- #
# ------------------------------------------------------ #
# Here we step through the different steps as per our n 
# that we chose at the beginning. 
    for j in range(nSteps+1):
        # we need to now step through and make sure that we 
        # take care of all the proper step sizes
        NSub = np.array([N[0], N[1]-2*locSteps[j], N[2]-2*locSteps[j]]).astype(int)
        ph_onesSub = np.ones(NSub, complex)
        ph_scanSub = np.zeros(NSub, complex)
        dataSub = np.zeros(NSub,complex)
        im_scanSub = np.zeros(NSub,complex)
        im_FullSub = np.zeros(NSub,complex)
        kSub = np.zeros(NSub)
        if locSteps[j]==0:
            kSub = k.copy()
            dataSub = np.fft.fftshift(kSub*dataFull,axes=(-2,-1))
            im_FullSub = tf.ifft2c(np.fft.fftshift(dataFull,axes=(-2,-1)),ph=ph_onesSub,sz=szFull)
        else:
            kSub = k[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
            dataSub = np.fft.fftshift(kSub*dataFull[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]],axes=(-2,-1))
            im_FullSub = tf.ifft2c(np.fft.fftshift(dataFull[:,locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]],axes=(-2,-1)),ph=ph_onesSub,sz=szFull)
                
        im_scan_wphSub = tf.ifft2c(dataSub, ph=ph_onesSub, sz=szFull)
        ph_scanSub = np.angle(gaussian_filter(im_scan_wphSub.real,0) +  1.j*gaussian_filter(im_scan_wphSub.imag,0))
        #ph_scanSub[i,:,:] = tf.matlab_style_gauss2D(im_scan_wphSub,shape=(5,5))
        ph_scanSub = np.exp(1j*ph_scanSub)
        im_scanSub = tf.ifft2c(dataSub, ph=ph_scanSub, sz=szFull)
        
        if j == 0:
            kMasked = kSub.copy()
        else:
            kHld = np.zeros(NSub)
            for msk in range(N[0]):
                padMask = tf.zpad(np.ones(kMasked[msk].shape),NSub[-2:])
                kHld[msk] = ((1-padMask)*kSub[msk] + padMask*tf.zpad(kMasked[msk],NSub[-2:])).reshape(np.hstack([1,NSub[-2:]]))
            kMasked = kHld
        #kMasks.append(kMasked)
        
        # Now we need to construct the starting point
        if locSteps[j]==0:
            pdfDiv = pdf.copy()
        else:
            pdfDiv = pdf[locSteps[j]:-locSteps[j],locSteps[j]:-locSteps[j]].copy()
        pdfZeros = np.where(pdfDiv < 1e-4)
        pdfDiv[pdfZeros] = 1
        pdfDiv = pdfDiv.reshape(np.hstack([1,NSub[-2:]])).repeat(NSub[0],0)
        
        N_imSub = NSub
        hldSub, dimsSub, dimOptSub, dimLenOptSub = tf.wt(im_scanSub[0].real,wavelet,mode)
        NSub = np.hstack([N_imSub[0], hldSub.shape])
        
        w_scanSub = np.zeros(NSub)
        im_dcSub = np.zeros(N_imSub)
        w_dcSub = np.zeros(NSub)
        
        if j == 0:
            data_dc = np.zeros(N_imSub,complex)
        else:
            data_dc_hld = np.zeros(N_imSub,complex)
            for i in range(N_imSub[0]):
                data_dc_hld[i] = tf.zpad(np.fft.fftshift(data_dc[i],axes=(-2,-1)),N_imSub[-2:])*(1-kSub)
            data_dc = np.fft.fftshift(data_dc_hld,axes=(-2,-1))
            
        dataSub += data_dc
        im_dcSub = tf.ifft2c(dataSub / np.fft.ifftshift(pdfDiv,axes=(-2,-1)), ph=ph_scanSub, axes=(-2,-1)).real.copy()
        for i in xrange(N[0]):
            w_scanSub[i,:,:] = tf.wt(im_scanSub.real[i],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0]
            w_dcSub[i,:,:] = tf.wt(im_dcSub[i],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)[0]
            
        kSub = np.fft.fftshift(kSub,axes=(-2,-1))
        w_dcSub = w_dcSub.flatten()
        im_spSub = im_dcSub.copy().reshape(N_imSub)
        dataSub = np.ascontiguousarray(dataSub)
        
        mets = ['Density Corrected']#,'Zeros','Ones','Random']
        wdcs = []
        
        #if (j!=0) and (locSteps[j]!=0):
            #kpad = tf.zpad(kStp[0],np.array(kSub.shape[-2:]).astype(int))
            #data_dc = np.fft.fftshift(tf.zpad(np.ones(kStp.shape[-2:])*dataStp[0], np.array(kSub.shape[-2:]).astype(int)) + (1-kpad)*np.fft.fftshift(dataSub))
            #im_dcSub = tf.ifft2c(data_dc[i,:,:] / np.fft.ifftshift(pdfDiv), ph=ph_scanSub[i,:,:]).real.copy().reshape(N_imSub)
        imdcs = [im_dcSub] #,np.zeros(N_im),np.ones(N_im),np.random.randn(np.prod(N_im)).reshape(N_im)]
        #import pdb; pdb.set_trace()
        
        kSamp = np.fft.fftshift(kMasked,axes=(-2,-1))
        
        args = (NSub, N_imSub, szFull, dimsSub, dimOptSub, dimLenOptSub, TV, XFM, dataSub, kSamp, strtag, ph_scanSub, kern, dirWeight, dirs, dirInfo, nmins, wavelet, mode, a)
        w_result = opt.fmin_tnc(f, w_scanSub.flat, fprime=df, args=args, accuracy=1e-4, disp=0)
        w_dc = w_result[0].reshape(NSub)
        #stps.append(w_dc)
        #w_stp.append(w_dc.reshape(NSub))
        #im_hld = np.zeros(N_imSub)
        #for i in range(NSub[0]):
            #im_hld[i] = tf.iwt(w_stp[-1][i],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)
        #imStp.append(im_hld)
        #plt.imshow(imStp[-1],clim=(minval,maxval)); plt.colorbar(); plt.show()
        #w_dc = w_stp[k].flatten()
        #stps.append(w_dc)
        #wdcHold = w_dc.reshape(NSub)
        #dataStp = np.fft.fftshift(tf.fft2c(imStp[-1],ph_scanSub),axes=(-2,-1))
        #kStp = np.fft.fftshift(kSub,axes=(-2,-1)).copy()
        #kMaskRpt = kMasked.reshape(np.hstack([1,N_imSub[-2:]])).repeat(N_imSub[0],0)
        im_hld = np.zeros(N_imSub)
        for i in range(NSub[0]):
            im_hld[i] = tf.iwt(w_dc[i],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)
        data_dc = tf.fft2c(im_hld, ph=ph_scanSub, axes=(-2,-1))
        kMasked = (np.floor(1-kMasked)*pctgSamp[locSteps[j]]*lam_trust + kMasked)
        #kMasked = (np.floor(1-kMasked)*pctgSamp[locSteps[j]]*1.0 + kMasked).reshape(np.hstack([1, N_imSub[-2:]]))
    
wHold = w_dc.copy().reshape(NSub)
imHold = np.zeros(N_imSub)

for i in xrange(N[0]):
    imHold[i,:,:] = tf.iwt(wHold[i,:,:],wavelet,mode,dimsSub,dimOptSub,dimLenOptSub)
    

np.save('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/tests/fullBrainTests/' + str(int(100*pctg)) + '_3_spatial_TV_im_final' + str(int(nSteps)) + '_comb_ks.npy',imHold)
np.save('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/tests/fullBrainTests/' + str(int(100*pctg)) + '_3_spatial_TV_im_final' + str(int(nSteps)) + '_comb_ks.npy',wHold)

#outvol = volumeFromData('/hpf/largeprojects/MICe/asalerno/pyDirectionCompSense/tests/fullBrainTests/' + str(int(100*pctg)) + '_3_spatial_TV_im_final_' + str(int(nSteps)) + '_comb_ks.mnc', imHold, dimnames=['xspace','yspace','zspace'], starts=(0, 0, 0), steps=(1, 1, 1), volumeType="uint")

np.save('tests/fullBrainTests/' + str(int(100*pctg)) + '_3_spatial_' + str(TV[0]) + '_TV_im_final_comb_ks.npy',imHold)
np.save('tests/fullBrainTests/' + str(int(100*pctg)) + '_3_spatial_' + str(TV[0]) + '_TV_im_final_comb_ks.npy',wHold)


outvol = volumeFromData('tests/fullBrainTests/' + str(int(100*pctg)) + '_3_spatial_' + str(TV[0]) + '_TV_im_final_comb_ks.mnc', imHold, dimnames=['xspace','yspace','zspace'], starts=(0, 0, 0), steps=(1, 1, 1), volumeType="uint")

outvol.writeFile()
strftime("%Y-%m-%d %H:%M:%S", localtime())


#im_stps = []
#gtv = []
#gxfm = []
#gdc = [] 
#for jj in range(len(stps)):
    #im_stps.append(tf.iwt(stps[jj].reshape(Ns[jj][-2:]),wavelet,mode,dims[jj],dimOpts[jj],dimLenOpts[jj]))
    #gtv.append(grads.gTV(im_stps[jj].reshape(N_ims[jj]),N_ims[jj],strtag, kern, 0, a=a).reshape(N_ims[jj][-2:]))
    #gxfm.append(tf.iwt(grads.gXFM(stps[jj].reshape(Ns[jj][-2:]),a=a),wavelet,mode,dims[jj],dimOpts[jj],dimLenOpts[jj]).reshape(N_ims[jj][-2:]))
    #gdc.append(grads.gDataCons(im_stps[jj], N_ims[jj], ph_scans[jj], dataSubs[jj], kMasks[jj],sz=szFull).reshape(N_ims[jj][-2:]))
        
#pctgLocs = np.hstack([pctgLocs, 1.05])
        
#for i in xrange(len(stps)):
    #plt.imshow(imStp[i]); plt.colorbar();
    #plt.title('Step with Negative Values')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/stepTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_a_' + str(int(a)) + '_setSamp_' + str(int(100*pctgLocs[i])) + '_nSteps_' + str(int(nSteps)) + '_negs')
    #vis.figSubplots([im_stps[i],gtv[i],gxfm[i],gdc[i]],titles=['Step','gTV','gXFM','gDC'],clims=[(minval,maxval),(np.min(gtv[i]),np.max(gtv[i])),(np.min(gxfm[i]),np.max(gxfm[i])),(np.min(gdc[i]),np.max(gdc[i]))])
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/stepTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_a_' + str(int(a)) + '_setSamp_' + str(int(100*pctgLocs[i])) + '_nSteps_' + str(int(nSteps)) + '_grads')
    #plt.imshow(TV[i]*gtv[i] + TV[i]*gxfm[i] + gdc[i])
    #plt.colorbar()
    #plt.title('Total Gradient')
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/stepTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_a_' + str(int(a)) + '_setSamp_' + str(int(100*pctgLocs[i])) + '_nSteps_' + str(int(nSteps)) + '_gradTotal')
    #if locSteps[i] != 0:
        #plt.imshow(abs(abs(im[0,locSteps[i]:-locSteps[i],locSteps[i]:-locSteps[i]]) - imStp[i]));
    #else:
        #plt.imshow(abs(abs(im[0] - imStp[i])));
    #plt.colorbar();
    #saveFig.save('/home/asalerno/Documents/pyDirectionCompSense/stepTests/'+ str(int(100*pctg)) + '_TV_XFM_'+ str(TV[i]) +'_a_' + str(int(a)) + '_setSamp_' + str(int(100*pctgLocs[i])) + '_nSteps_' + str(int(nSteps)) + 'imDiff')
    
    