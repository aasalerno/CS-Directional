# !/usr/bin/env python -tt
#
#
#   transforms.py
#
#
#
#   Code to be a wrapper for all of the transforms that are done in order to clean up the central codes

import numpy as np
import numpy.fft as fft
from rwt import dwt, idwt
from rwt.wavelets import daubcqf

def fft2c(data_to_fft):
    FFTdata = 1/np.sqrt(data_to_fft.size)*fft.fftshift(fft.fft2(fft.ifftshift(data_to_fft))
    return FFTdata

def ifft2c(data_to_ifft):
    IFFTdata = np.sqrt(data_to_ifft.size)*fft.ifftshift(fft.ifft2(fft.fftshift(data_to_ifft))
    return IFFTdata

def xfm(data_to_xfm,scaling_factor = 4,L = 2):
    h = daubcqf(scaling_factor)[0]
    XFMdata = dwt(data_to_xfm,h,L)
    return XFMdata

def ixfm(data_to_ixfm,scaling_factor = 4,L = 2):
    h = daubcqf(scaling_factor)[0]
    IXFMdata = idwt(data_to_ixfm,h,L)
    return IXFMdata
