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
import sys
import glob
import matplotlib.pyplot as plt

EPS = np.finfo(float).eps

def indata(inFolder):
    '''
    This code reads in data from mnc files in order to be worked on via the code. 
    Reads the data in and outputs it as a list
    '''
    us_data = [] 

    # get the names of the input files from the specified folder
    filenames = glob.glob(inFolder + '/*.mnc')

    # Put the data in a large dataset to be worked with
    for files in filenames:
        cnt = 0
        us_data.append(pyminc.volumes.factory.volumeFromFile(files))
        cnt += 1 
    return us_data


    
def zpad(res_sz,orig_data):
    res_sz = np.array(res_sz)
    orig_sz = np.array(orig_data.shape)
    padval = np.ceil((res_sz-orig_sz)/2)
    res = np.pad(orig_data,([padval[0],padval[0]],[padval[1],padval[1]]),mode='constant')
    return res