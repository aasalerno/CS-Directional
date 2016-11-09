import numpy as np

def rmse(xk,x):
    return np.sqrt(np.sum((xk-x)**2)/len(xk))

    
import matplotlib.pyplot as plt
import os.path
os.chdir('/home/asalerno/Documents/pyDirectionCompSense/') # Change this to the directory that you're saving the work in
filename = '/home/asalerno/Documents/pyDirectionCompSense/data/SheppLogan256.npy'

np.random.seed(2000)

#im = np.zeros([8,8]);
#im[3:5,3:5] = 1;

im=np.load(filename)

x0sty = ['im_dc','zeros','im']
sty = ['TVOnly','XFMOnly-nonTanh','TVandXFM']
#-------------------------------------------------------
val = range(len(sty)*len(x0sty))

for i in xrange(len(sty)):                                        
    for j in xrange(len(x0sty)):
        im_k = np.load('simpleBoxData/' + sty[i] + '/' + 'SL-' + x0sty[j] + '-result.npy')
        im_k = (-np.min(im_k)+im_k)
        im_k = im_k/np.max(im_k) # Normalize the values of im_k
        val[i*3+j] = rmse(im_k,im)
