# Imports
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import os.path
from sys import path as syspath
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/")

# Change this to the directory that you're saving the work in
os.chdir('/home/asalerno/Documents/pyDirectionCompSense/')  
import transforms as tf
import scipy.ndimage.filters
import grads
import sampling as samp
import direction as d
# from scipy import optimize as opt
import optimize as opt
import scipy.optimize as spopt
from recon_CS import *
import read_from_fid as rff
import saveFig
radius = 0.1

im = np.load('/home/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/fullySampledBrain.npy')
N = np.array(im.shape)  # image Size
#tupleN = tuple(N)
pctg = 0.25  # undersampling factor
ph = tf.matlab_style_gauss2D(im,shape=(5,5));
#ph = np.ones(im.shape, complex)
Ps = np.arange(0,5.1,.5)

kmult = np.zeros(np.hstack([len(Ps), N]))
kadd = np.zeros(kmult.shape)
pdfadd = np.zeros(kmult.shape)
pdfmult = np.zeros(kmult.shape)

cnt = -1
for P in Ps:
    cnt += 1
    print(P)
    if P>2:
        pdfadd[cnt,:,:] = samp.genPDF(N, P, pctg, radius=radius, cyl=[0])
        kadd[cnt,:,:] = samp.genSampling(pdfadd[cnt,:,:], 50, 2)[0].astype(int)
        
    pdfmult[cnt,:,:] = samp.genPDF(N, P, pctg, radius=radius, cyl=[0], style = 'mult')
    kmult[cnt,:,:] = samp.genSampling(pdfmult[cnt,:,:], 50, 2)[0].astype(int)

cnt = -1
rads = np.linspace(-np.sqrt(2),np.sqrt(2),N[0])
plt.rcParams['lines.linewidth'] = 4
cols = ('b','k','r','g','m','c','y','0.5','0.75',(0.25,0.25,0.75),(1,0.25,0.5))
fig=plt.figure()
strPs = map(str,Ps)
for P in Ps:
    print(P)
    cnt += 1
    plt.plot(rads,pdfmult[cnt,147,:],color=cols[cnt])
    #samp.pltSlice(pdfmult[cnt,:,:],sl=147,rads=rads,col=cols[cnt])
    plt.legend(strPs)
    plt.xlim(-np.sqrt(2),np.sqrt(2))
    plt.ylim(0,1.1)
    plt.xlabel('Radius')
    plt.ylabel('PDF')
    plt.title('PDF using 1/|r|^p')
    
saveFig.save('/micehome/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/pdf/r_-p_method_compared')

cnt = -1
pleg=[]
for P in Ps:
    print(P)
    cnt += 1
    if P>2:
        pleg.append(str(P))
        plt.plot(rads,pdfadd[cnt,147,:],color=cols[cnt])
        #samp.pltSlice(pdfmult[cnt,:,:],sl=147,rads=rads,col=cols[cnt])
        plt.legend(strPs)
        plt.xlim(-np.sqrt(2),np.sqrt(2))
        plt.ylim(0,1.1)
        plt.xlabel('Radius')
        plt.ylabel('PDF')
        plt.title('PDF (1-r)^p + const')
    
saveFig.save('/micehome/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/pdf/1-r_p_method_compared')

cnt=-1
for P in Ps:
    print(P)
    cnt += 1
    if P>2:
        pleg.append(str(P))
        plt.plot(rads,pdfmult[cnt,147,:],color='b')
        plt.plot(rads,pdfadd[cnt,147,:],color='r')
        #samp.pltSlice(pdfmult[cnt,:,:],sl=147,rads=rads,col=cols[cnt])
        plt.legend(('1/r^p','(1-r)^p + const'))
        plt.xlim(-np.sqrt(2),np.sqrt(2))
        plt.ylim(0,1.1)
        plt.xlabel('Radius')
        plt.ylabel('PDF')
        plt.title('PDF Compared - P = ' + str(P))
        saveFig.save('/micehome/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/pdf/add_mult_comp_P_' + str(P))
        

cnt=-1
for P in Ps:
    print(P)
    cnt += 1
    radialHistogram(kmult[cnt,:,:],disp=0)
    plt.title('Radial Histogram Mult - P = ' + str(P))
    saveFig.save('/micehome/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/pdf/radialHist_mult_P_' + str(P))
    
    if P>2:
        radialHistogram(kadd[cnt,:,:],disp=0)
        plt.title('Radial Histogram Add - P = ' + str(P))
        saveFig.save('/micehome/asalerno/Documents/pyDirectionCompSense/brainData/exercise_irradiation/bruker_data/running_C/P14/rad_0.1/pdf/radialHist_add_P_' + str(P))