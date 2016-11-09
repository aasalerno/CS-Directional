'''
Wavelet checking code!

Let's use this code to check how the wavelets are calculated and things like that.

We will be using my phantom -- with Daubeauchies and Haar wavelets.

'''

from __future__ import division
import numpy as np
import numpy.fft as fft
from rwt import dwt, idwt
import rwt.wavelets as wv
import direction as d

