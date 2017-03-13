from optparse import OptionParser, Option, OptionValueError, OptionGroup
import imp
import sys
import os
from numpy import *
from numpy.fft import *
sys.path.append('/home/bjnieman/source/mri_recon')
from mri_recon import dummyopt
#from recon_genfunctions import FatalError,get_dict_value,default_recon
import varian_read_file as vrf
from mnc_output import write_to_mnc_file
import matplotlib.pyplot as plt
from fse3dmice_recon import ROshift 


#########################################################
def apply_RO_shift(kline,pixel_shift):               
    nro=kline.shape[-1]
    roramp = exp(1.j*2*pi*pixel_shift*(append(arange(nro/2),arange(-nro/2,0,1)))/nro)
    klinemod = ifft(((roramp)*fft(kline,axis=-1)),axis=-1)
    return klinemod

    
######################################################

inputdirectory='/hpf/largeprojects/MICe/asalerno/zerosfid/21feb17.fid_20170221T185820'
inputAcq = vrf.VarianAcquisition(inputdirectory)

##seqrec = seqmodule.seq_reconstruction(inputAcq,options,outputfile)

imouse=6

#nf=6, ni=6, 125 array elems
nro=inputAcq.param_dict["np"]/2
nf=inputAcq.param_dict["nf"]
etl=inputAcq.param_dict["etl"]
ni=int(inputAcq.param_dict["ni"])
narray=inputAcq.param_dict["gdiff2"].shape[0]
carray=zeros( (ni,narray,etl,nro), complex)

ntr = narray
for k in range(ni):
    for j in range(ntr):
        fid_data,errflag = inputAcq.getdatafids(k*etl*narray+j*etl,k*etl*narray+(j+1)*etl,rcvrnum=imouse)
        carray[k,j,:,:] = fid_data.copy()

carray = mean(carray[2::,:,:,:],axis=0)

carray_corr = carray.copy()

#make single B0 avg fid from echo 0
B0_fids=range(120,125)
B0_avg=mean(carray[B0_fids,0,:],axis=0)

#derive readout shift correction
Allecho_evenodd_pixshift=zeros((narray,etl),float)
for j in range(narray):
    for k in range(etl):
        Allecho_evenodd_pixshift[j,k] = ROshift(B0_avg,carray[j,k,:])
        carray_corr[j,k,:] = apply_RO_shift(carray[j,k,:],-Allecho_evenodd_pixshift[j,k]) 


#derive phase correction for each echo
maxind = argmax(abs(B0_avg))
Allecho_phasecorr=zeros((narray,etl),complex)
for j in range(narray):
    for k in range(etl):
        Allecho_phasecorr[j,k]=exp(-1.j*angle(carray_corr[j,k,maxind]))
        carray_corr[j,k,:]=carray_corr[j,k,:]*Allecho_phasecorr[j,k]


#clear impact of PE gradient amplitudes (particularly PE1) that leave residual encoding
#likely to be very problematic for the existing diff sequence and for AS CS scheme
#plan to implement an empirical correction into the diffusion sequence to correct this so
#acq'd data is as near as possible to the prescribed Cartesian grid

