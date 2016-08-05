import matplotlib.pyplot as plt
import numpy as np

def plt_slice(im,im_dc,im_res,axis = 0,slc = None,title = None):
    
    if axis:
        im = im.T
        im_dc = im_dc.T
        im_res = im_res.T
    
    if ((im.shape != im_dc.shape) and (im.shape != im_res.shape)):
        raise ValueError('Images are not the same size')
    
    N = im.shape
    
    if slc == None:
        slc = np.ceil(N(0)/2)
        
    
    x = np.arange(N[1])
    plt.plot(x,abs(im[slc,:]),label = 'Original')
    plt.plot(x,abs(im_dc[slc,:]),label='Start Point')
    plt.plot(x,abs(im_res[slc,:]),label = 'Result')
    plt.legend()
    plt.title(title)
    plt.show()
    