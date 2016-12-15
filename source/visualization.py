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
    
def figSubplots(data,dims=None,clim=(0,1),titles=None,figsize=(8,6)):
    if isinstance(data,tuple) or isinstance(data,list):
        datahold = np.zeros(np.hstack([len(data),data[0].shape]))
        for i in range(len(data)):
            datahold[i,:,:] = data[i]
        data = datahold
        
    if not dims:
        dims = np.array(closefac(data.shape[0]))
    
    fig = plt.figure(figsize=figsize)
    for i in range(data.shape[0]):
        #import pdb; pdb.set_trace()
        ax = fig.add_subplot(dims[0],dims[1],i+1)
        plt.imshow(data[i,:,:],clim=clim)
        if titles:
            plt.title(titles[i])
        
    