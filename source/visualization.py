import numpy as np
import matplotlib.pyplot as plt

def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    return primfac

def closefac(n):
    if n<0:
        print('Fed a negative number, using |n|')
        n = abs(n)
    testNum = int(np.sqrt(n))
    while n%testNum != 0:
        testNum -= 1
    
    return testNum, int(n/testNum)
    
def figSubplots(data,dims=None,clims=None,titles=None,figsize=(24,13.5),colorbar=True):
    if isinstance(data,tuple) or isinstance(data,list):
        datahold = np.zeros(np.hstack([len(data),data[0].shape]))
        for i in range(len(data)):
            datahold[i,:,:] = data[i]
        data = datahold
        
    if not dims:
        dims = np.array(closefac(data.shape[0]))
    
    if clims is not None:
        if len(clims) == 2:
            climHold = []
            for i in range(len(data)):
                climHold.append(clims)
            clims = climHold
        elif len(clims) != len(data):
            raise TypeError('Number of limits and datasets don''t match')
    
    fig = plt.figure(figsize=figsize)
    data = np.squeeze(data)
    for i in range(data.shape[0]):
        #import pdb; pdb.set_trace()
        ax = fig.add_subplot(dims[0],dims[1],i+1)
        if clims:
            plt.imshow(data[i,:,:],clim=clims[i])
        else:
            plt.imshow(data[i,:,:],clim=clims)
        if titles:
            plt.title(titles[i])
        if colorbar:
            plt.colorbar()
        
def imDiff(ims):
    imdiff=[]
    clims=[]
    cnt=0
    for i in range(len(ims)):
        for j in range(len(ims)):
            if i<j:
                imdiff.append(abs(ims[i]-ims[j]))
                clims.append((np.min(imdiff[cnt]),np.max(imdiff[cnt])))
                cnt+=1
    return imdiff, clims
    
    


def imDiffPerc(ims):
    imdiff=[]
    clims=[]
    cnt=0
    for i in range(len(ims)):
        for j in range(len(ims)):
            if i<j:
                imdiff.append(abs(ims[i]-ims[j])/(np.max(np.squeeze(np.stack([ims[i],ims[j]],axis=0)),axis=0)))
                clims.append((np.min(imdiff[cnt]),np.max(imdiff[cnt])))
                cnt+=1
    return imdiff, clims