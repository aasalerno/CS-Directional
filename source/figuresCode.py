import os
from sys import path as syspath
os.chdir(
    '/home/asalerno/Documents/pyDirectionCompSense/')  # Change this to the directory that you're saving the work in
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/")
import numpy as np
#import errorcalc as err
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'

import saveFig
import transforms as tf


def rmse(xk,x):
    return np.sqrt(np.sum((xk-x)**2)/len(xk))
    
def zpad(orig_data,res_sz):
    res_sz = np.array(res_sz)
    orig_sz = np.array(orig_data.shape)
    padval = (np.ceil((res_sz-orig_sz)/2))
    res = np.pad(orig_data,([int(padval[0]),int(padval[0])],[int(padval[1]),int(padval[1])]),mode='constant')
    return res

im = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/fullySampledBrain.npy')
imgsData = np.zeros([5,6,6,4,294,294])
rmse_data = np.zeros([5,6,6,4])
pc=-1
TV=-1
XFM=-1
imdc=-1



for pctg in [0.20, 0.25, 0.33, 0.40, 0.50]:
    pc += 1
    for TVWeight in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
        TV+=1
        for XFMWeight in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
            XFM+=1
            for imdcs in ['zeros','ones','densCorr','imFull']:
                imdc += 1
                imgsData[pc,TV,XFM,imdc,:,:] = np.load("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/TV%s_XFM%s/%.2fper_result_im_dc_" % (float('%.1g' % TVWeight), float('%.1g' % XFMWeight) , pctg) + imdcs + ".npy")
                rmse_data[pc,TV,XFM,imdc] = rmse(imgsData[pc,TV,XFM,imdc,:,:],im)
            imdc=-1
        XFM=-1
    TV=-1

pctg = [0.20, 0.25, 0.33, 0.40, 0.50]
TVWeight = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
XFMWeight = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
imdcs = ['zeros','ones','densCorr','imFull']

N = len(XFMWeight)

ind = np.arange(N)+.5
width =0.15

# TV and XFM Comparison
# imdcs = densCorr -- 2
# pctg = 0.33 -- 2
fig, ax = plt.subplots()
rects1 = ax.bar(ind - 3*width, rmse_data[2,0,:,2], width, color='r', alpha=.1)
rects2 = ax.bar(ind - 2*width, rmse_data[2,1,:,2], width, color='r', alpha=.3)
rects3 = ax.bar(ind - 1*width, rmse_data[2,2,:,2], width, color='m', alpha=.5)
rects4 = ax.bar(ind + 0*width, rmse_data[2,3,:,2], width, color='m', alpha=.7)
rects5 = ax.bar(ind + 1*width, rmse_data[2,4,:,2], width, color='b', alpha=.8)
rects6 = ax.bar(ind + 2*width, rmse_data[2,5,:,2], width, color='b', alpha=1)

ax.set_ylabel('RMSE')
ax.set_title('RMSE for Different TV and XFM Combinations')
ax.set_xticks(ind)
ax.set_xticklabels(('0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05'))
ax.set_xlabel('XFMWeights')
ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]),('0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05'),title='TVWeights',loc='upper left'); #loc='center left', bbox_to_anchor=(1, 0.5))


## im_dc and pctg Comparison
## TV = 0.005 -- 3
## XFM = 0.005 -- 3
#N = len(imdcs)

#ind = np.arange(N)+.5
#width =0.15

#fig, ax = plt.subplots()
#rects1 = ax.bar(ind - 3*width, rmse_data[0,3,3,:], width, color='b', alpha=.2)
#rects2 = ax.bar(ind - 2*width, rmse_data[1,3,3,:], width, color='b', alpha=.4)
#rects3 = ax.bar(ind - 1*width, rmse_data[2,3,3,:], width, color='b', alpha=.6)
#rects4 = ax.bar(ind + 0*width, rmse_data[3,3,3,:], width, color='b', alpha=.8)
#rects5 = ax.bar(ind + 1*width, rmse_data[4,3,3,:], width, color='b', alpha=1)

#ax.set_ylabel('RMSE')
#ax.set_title('RMSE for Different Pctg and Starting Point Combinations')
#ax.set_xticks(ind)
#ax.set_xticklabels(('zeros','ones','densCorr','imFull'))
#ax.set_xlabel('Percent Sampled')
#ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]),('20%', '25%', '33%', '40%', '50%'), title='im_dc',loc='upper right'); #loc='center left', bbox_to_anchor=(1, 0.5))


# im_dc and pctg Comparison
# TV = 0.005 -- 3
# XFM = 0.005 -- 3
rmse_lr = np.zeros(len(pctg))
for i in range(len(pctg)):
lrLoc = int(np.ceil((N[0]-np.ceil(N[0]/np.sqrt(1/pctg)))/2))
data_lr_us = np.fft.fftshift(tf.ifft2c(im,np.ones(im.shape)))[lrLoc:-lrLoc,lrLoc:-lrLoc]
data_lr_rs = zpad(data_lr_us,im.shape)
im_lr_rs = abs(tf.fft2c(np.fft.fftshift(data_lr_rs),np.ones(im.shape)))
rmse_lr[i] = err.rmse(abs(im_lr_rs),im)

N = len(pctg)

ind = np.arange(N)+.5
width =0.15

fig, ax = plt.subplots()
rects0 = ax.bar(ind - 4*width, rmse_lr, width, color='b', alpha=0.1)
rects1 = ax.bar(ind - 3*width, rmse_data[:,3,3,0], width, color='b', alpha=.25)
rects2 = ax.bar(ind - 2*width, rmse_data[:,3,3,1], width, color='b', alpha=.5)
rects3 = ax.bar(ind - 1*width, rmse_data[:,3,3,2], width, color='b', alpha=.75)
rects4 = ax.bar(ind + 0*width, rmse_data[:,3,3,3], width, color='b', alpha=1)

ax.set_ylabel('RMSE')
ax.set_title('RMSE for Different Pctg and Starting Point Combinations')
ax.set_xticks(ind)
ax.set_xticklabels(('20%', '25%', '33%', '40%', '50%'))
ax.set_xlabel('Percent Sampled')
ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]),('zeros','ones','densCorr','imFull'), title='im_dc',loc='upper right'); #loc='center left', bbox_to_anchor=(1, 0.5))



# Subplots for each XFM and TV solution and each pctg
pc=-1
TV=-1
XFM=-1
N = im.shape
minval = np.min(im)
maxval = np.max(im)

for pctg in [0.20, 0.25, 0.33, 0.40, 0.50]:
    pc += 1
    lrLoc = int(np.ceil((N[0]-np.ceil(N[0]/np.sqrt(1/pctg)))/2))
    im_lr = tf.fft2c(np.fft.fftshift(np.fft.fftshift(tf.ifft2c(im,np.ones(im.shape)))[lrLoc:-lrLoc,lrLoc:-lrLoc]),np.ones(im[lrLoc:-lrLoc,lrLoc:-lrLoc].shape))
    for TVWeight in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
        TV+=1
        for XFMWeight in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
            XFM+=1
            fig = plt.figure(figsize=(15,10))
            ax1 = fig.add_subplot(2,3,1)
            plt.imshow(im, clim=(minval,maxval))
            ax1.set_title('Original Image')
            ax2 = fig.add_subplot(2,3,2)
            plt.imshow(imgsData[pc,TV,XFM,0,:,:],clim=[minval,maxval])
            ax2.set_title('Recon %.2f%% samp imdc=zeros' % (pctg*100))
            ax3 = fig.add_subplot(2,3,3)
            plt.imshow(imgsData[pc,TV,XFM,1,:,:],clim=(minval,maxval))
            ax3.set_title('Recon %.2f%% samp imdc=ones' % (pctg*100))
            ax4 = fig.add_subplot(2,3,4)
            plt.imshow(abs(im_lr))
            ax4.set_title('Low Res with %.2f%% of Data' % (pctg*100))
            ax5 = fig.add_subplot(2,3,5)
            plt.imshow(imgsData[pc,TV,XFM,2,:,:],clim=(minval,maxval))
            ax5.set_title('Recon %.2f%% samp imdc=densCorr' % (pctg*100))
            ax6 = fig.add_subplot(2,3,6)
            plt.imshow(imgsData[pc,TV,XFM,3,:,:],clim=(minval,maxval))
            ax6.set_title('Recon %.2f%% samp imdc=imFull' % (pctg*100))
            
            saveFig.save("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/TV%s_XFM%s/%.2fper_im_lr_edit_comparison" % (float('%.1g' % TVWeight), float('%.1g' % XFMWeight), pctg))
        XFM=-1
    TV=-1

    
# Rerunning data comparison
pctg = 0.25
im = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/fullySampledBrain.npy')
N = im.shape
minval = np.min(im)
maxval = np.max(im)

lrLoc = int(np.ceil((N[0]-np.ceil(N[0]/np.sqrt(1/pctg)))/2))
im_lr = tf.fft2c(np.fft.fftshift(np.fft.fftshift(tf.ifft2c(im,np.ones(im.shape)))[lrLoc:-lrLoc,lrLoc:-lrLoc]),np.ones(im[lrLoc:-lrLoc,lrLoc:-lrLoc].shape))


fig = plt.figure(figsize=(15,10))

# Original Image
ax1 = fig.add_subplot(2,3,1)
plt.imshow(im, clim=(minval,maxval)) 
ax1.set_title('Original Image')

# im_full as starting point
ax2 = fig.add_subplot(2,3,2)
im_datfull = np.load("brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/TV0.005_XFM0.005/0.25per_result_im_dc_imFull.npy") 
plt.imshow(abs(im_datfull), clim=(minval,maxval)) 
ax2.set_title('One Run Recon im_dc=im')

# Lo Resolution
ax3 = fig.add_subplot(2,3,3)
plt.imshow(abs(im_lr))
ax3.set_title('LoRes')

# im_dc after one run -- TV=XFM=0.005
ax4 = fig.add_subplot(2,3,4)
im_dc = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/TV0.005_XFM0.005/0.25per_result_im_dc_densCorr.npy')
plt.imshow(abs(im_dc), clim=(minval,maxval))
ax4.set_title('1 Recon im_dc=DC XFM=TV=0.005')

# im_dc after two runs -- TV=XFM=0.002
ax5 = fig.add_subplot(2,3,5)
im_dc2 = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/postComps/rerun1_TV_0.02_XFM_0.02.npy')
plt.imshow(abs(im_dc2), clim=(minval,maxval))
ax5.set_title('2 Recon im_dc=DC XFM=TV=0.002')


# im_dc after three runs -- TV=XFM=0.001
ax6 = fig.add_subplot(2,3,6)
im_dc3 = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/postComps/rerun1_TV_0.001_XFM_0.001.npy')
plt.imshow(abs(im_dc3), clim=(minval,maxval))
ax6.set_title('3 Recon im_dc=DC XFM=TV=0.001')