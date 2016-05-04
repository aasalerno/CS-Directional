#!/usr/bin/env python -tt
#
#
# recon_CS.py
#
#
# We start with the data from the scanner. The inputs are:
#       - inFile (String) -- Location of the data
#                         -- Direct to a folder where all the data is
#       - 
#

from __future__ import division
import pyminc.volumes.factory
import numpy as np 
import scipy as sp
import sys
import rwt
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path
import sampling as samp

def recon_CS(filename,
             TVWeight = 0.01,
             XFMWeight = 0.01,
             TVPixWeight = 1,
             DirWeight = 0,
             DirType = 2,
             ItnLim = 150,
             epsilon = 0.02,
             l1smooth = 1e-15,
             xfmNorm = 1):
                    
    