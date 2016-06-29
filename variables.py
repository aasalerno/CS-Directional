from __future__ import division
import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt
import os.path
import transforms as tf
import scipy.ndimage.filters
import grads
import sampling as samp
import direction as d
from scipy import optimize as opt

filename = '/home/asalerno/Documents/pyDirectionCompSense/data/SheppLogan256.npy'
strtag = ['spatial','spatial']
TVWeight = 0.01
XFMWeight = 0.01
dirWeight = 0
#DirType = 2
ItnLim = 150
epsilon = 1
l1smooth = 1e-15
xfmNorm = 1
scaling_factor = 4
L = 2
method = 'CG'
dirFile = None
nmins = None