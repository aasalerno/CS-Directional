from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import os.path
from sys import path as syspath
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/")
os.chdir(
    '/home/asalerno/Documents/pyDirectionCompSense/')  # Change this to the directory that you're saving the work in
import transforms as tf
import sampling as samp


def calcPhase(im, k, w1=10, w2=8, w3=4, eps = np.pi/18, sig = 0):
    '''
    This function is to try to iteratively calculate the phase as per Tisdall and Atkins 2005 (https://www.cs.sfu.ca/~stella/papers/2005/spie.pdf)
    
    With X(p) being our spatial domain value at pixel p:
        - X(p) = image we have
        - s(p) = signal proper
        - n_r(p) = real noise -- Gaussian
        - n_i(p) = imaginary noise -- Gaussian
        
    Assume: X(p) = s(p) exp[i φ(p)] + n_r(p) + i n_i(p)
    
    We want to calculate φ^(p) which is an estimate of φ(p), thrn multiply it in. If φ(p) == φ^(p), then:
    
    X(p) exp[-i φ^(p)] = s(p) + (n_r(p) + i n_i(p)) exp[-i φ^(p)]
        Because exp[-i φ^(p)] is just a rotation, the rotation of noise, just makes different noise, so (n_r(p) + i n_i (p)) exp[-i φ^(p)] == n_r`(p) + i n_i`(p)
    
    So our new measurement is:
        X(p) exp[-i φ^(p)] = s(p) + (n_r`(p) + i n_i`(p)) 
        
    '''
    if sig==0:
        sig = np.var(im[:50,:50])
    
    ph_ones = np.ones(im.shape)
    data = np.fft.ifftshift(k) * tf.fft2c(im, ph=ph_ones)
    im_scan_ph = tf.ifft2c(data, ph=ph_ones)
    ph = tf.matlab_style_gauss2D(im_scan_ph,shape=(5,5))
    im_scan = tf.ifft2c(data, ph=ph)

    N = im.shape
    window1 = int(np.ceil(w1/2))
    window2 = int(np.ceil(w2/2))
    window3 = int(np.ceil(w3/2))

    
    im_wrap = np.pad(im_scan,window1,'wrap')

    #ph = np.zeros(im.shape,complex)
    #ph_new = np.zeros(im.shape,complex)
    ph_new = ph.copy()
    wgts = np.ones(im.size)    
        
    '''
    
    We then do this over three steps. Step 1 is:
    
        1. Apply the given phase correction, φ^ to the recorded image, I to get our current best guess image, I`.
    
        2. Calculate the mean of the imaginary component of all the pixels in I` in a window of width w1 around p.
        
        3. If the mean imaginary component is greater than ||p||, the magnitude of p, set p’s new phase estimate, φ^`(p), to be π/2 to correct as much as possible.
        
        4. If the mean imaginary component is less than −||p||, set φ^`(p)=−π/2 to correct as much as possible.
        
        5. Set φ^`(p) so p’s phase is on (−π/2,π/2) and its imaginary component cancels the mean component of all the other pixels in the window.
    
    
    '''
    
    ph = ph_new.copy()
    
    for x in range(N[0]):
        for y in range(N[1]):
            mn = np.mean(im_wrap[x:x+1+w1,y:y+1+w1].imag)
            if mn > abs(im_scan[x,y]):
                ph_new[x,y] = +1j
            elif mn < -abs(im_scan[x,y]):
                ph_new[x,y] = -1j
            else:
                ph_new[x,y] = abs(ph_new[x,y].real) - mn*1j
                # The abs() is required here to ensure that the phase is on (-π/2,π/2)
    
    
    ''' 
    
    Step 2 requires us to look at those times where we shifted positives to negatives, and try to flip it back when necessary.
    
    This then follows three more substeps:
        
        1. Calculate the mean of the distances, wrapped onto the range [−π,π), from φ^(p) to each other phase estimate pixel in a window of with w2 centered on p.
        
        2. Calculate the mean of the distances, wrapped onto the range [−π,π), from φ^(p) + π to each other phase estimate pixel in a window of with w2 centered on p.
        
        3. If the second mean distance is smaller than the first, mark p as flipped.
        
    '''
    
    # need to map phases from [-pi,pi)
    #ph_wrap_angles_piShift = (np.angle(np.pad(ph_new,window2,'wrap')) + np.pi) % (2*np.pi)
    ph_wrap_angles = np.arctan2(ph_new.imag, ph_new.real)
    cnt = 0
    
    for x in range(N[0]):
        for y in range(N[1]):
            diffs = np.sum(np.diff(ph_wrap_angles[x:x+1+w2,y:y+1+w2],axis=0)) + \
                    np.sum(np.diff(ph_wrap_angles[x:x+1+w2,y:y+1+w2],axis=1)) 
            ph_wrap_hold = np.exp(1j*ph_wrap_angles[x,y]+np.pi)
            ph_wrap_angles[x,y] = np.arctan2(ph_wrap_hold.imag,ph_wrap_hold.real)
            diffs_piShift = np.sum(np.diff(ph_wrap_angles[x:x+1+w2,y:y+1+w2],axis=0)) + \
                            np.sum(np.diff(ph_wrap_angles[x:x+1+w2,y:y+1+w2],axis=1)) 
            
            if diffs_piShift < diffs:
                #print('Smaller')
                cnt+=1
                ph_new[x,y] = np.exp(1j*ph_wrap_angles[x,y])
            
            ph_wrap_hold = np.exp(1j*ph_wrap_angles[x,y]-np.pi)
            ph_wrap_angles[x,y] = np.arctan2(ph_wrap_hold.imag,ph_wrap_hold.real)
        
    ph_new = np.exp(1j*ph_wrap_angles)
    
    
            