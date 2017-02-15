# 2D phase unwrapping, based on,
# OPTICS LETTERS / Vol. 28, No. 14 / July 15, 2003 
# Fast phase unwrapping algorithm for interferometric applications 
# Marvin A. Schofield and Yimei Zhu 
# Implemented by,
# Nazim Bharmal n.a.bharmal.dur.ac.uk  07/Nov/2012
# with modifications to iterate in i_unwrap_2d

import numpy as np

puRadius=lambda x : np.roll( np.roll(
      np.add.outer( np.arange(-x.shape[0]/2+1,x.shape[0]/2+1)**2.0,
                    np.arange(-x.shape[1]/2+1,x.shape[1]/2+1)**2.0 ),
      x.shape[1]/2+1,axis=1), x.shape[0]/2+1,axis=0)+1e-9

idt,dt=np.fft.ifft2,np.fft.fft2
puOp=lambda x : idt( np.where(puRadius(x)==1e-9,1,puRadius(x)**-1.0)*dt(
      np.cos(x)*idt(puRadius(x)*dt(np.sin(x)))
     -np.sin(x)*idt(puRadius(x)*dt(np.cos(x))) ) )

def gen_mirrored_ip(ip):
    mirrored=np.zeros([x*2 for x in ip.shape],ip.dtype)
    mirrored[:ip.shape[0],:ip.shape[1]]=ip
    mirrored[ip.shape[0]:,:ip.shape[1]]=ip[::-1,:]
    mirrored[ip.shape[0]:,ip.shape[1]:]=ip[::-1,::-1]
    mirrored[:ip.shape[0],ip.shape[1]:]=ip[:,::-1]
    return mirrored

def phase_unwrap_2d(ip):
    mirrored=gen_mirrored_ip(ip)
    return (ip+2*np.pi*
            np.round((puOp(mirrored).real[:ip.shape[0],:ip.shape[1]]-ip)/(2*np.pi)))

def i_unwrap_2d(ip,maxiter=5):
    mirrored=gen_mirrored_ip(ip)
    ip_prime = puOp(mirrored).real[:ip.shape[0],:ip.shape[1]] #only need to calc this once, so put it outside loop
    j=0
    ipw=ip.copy()
    while (j<maxiter):
        ipnew=ip+2*np.pi*np.round( (ip_prime-ip)/(2*np.pi) )
        maxdiff = max(abs(ipnew-ipw).flat)
        ipw=ipnew
        j+=1
        if (maxdiff<0.5*np.pi):
            break
    return ipw

