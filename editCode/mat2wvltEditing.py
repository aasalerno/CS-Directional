def wvlt2mat(wvlt, dims=None, dimOpt=None, dimLenOpt= None):
    if dims is None:
        dims = np.zeros(np.hstack([len(wvlt), len(wvlt[0].shape)]))
        if dims.shape[-1]==2:
            for i in range(len(wvlt)):
                if i == 0:
                    dims[i,...] = wvlt[i].shape
                else:
                    dims[i,...] =  wvlt[i][0].shape
        elif dims.shape[-1]==3:
            wvKeys = wvlt[1].keys()
            for i in range(len(wvlt)):
                if i == 0:
                    dims[i,...] = wvlt[i].shape
                else:
                    dims[i,...] = wvlt[i][wvKeys[0]].shape
    if np.any(dims[0,...] != np.zeros(len(wvlt[0].shape))):
        dims = np.vstack([np.zeros(len(wvlt[0].shape)), dims]).astype(int)
        
    if dimOpt is None:
        dimOpt = np.zeros(np.hstack([len(wvlt), len(wvlt[0].shape)]))
        dimOpt[0,...] = wvlt[0].shape
        for i in range(len(wvlt)):
            dimOpt[i,...] = np.sum(dimOpt,axis=0)
    if np.any(dimOpt[0,...] != np.zeros(len(wvlt[0].shape))):
        dimOpt =  np.vstack([np.zeros(len(wvlt[0].shape)), dimOpt]).astype(int)
        
    if dimLenOpt is None:
        dimLenOpt = np.zeros(dimOpt.shape)
        for i in range(dimOpt.shape[0]):
            dimLenOpt[i,...] = np.sum(dimOpt[0:i+1,...],axis=0)
    dimLenOpt = dimLenOpt.astype(int)
    
    sz = np.sum(dimOpt,axis=0,dtype=int)
    mat = np.zeros(sz,complex)
    
    if dims.shape[-1]==2:
        for i in range(1,dims.shape[0]):
            if i==1: 
                mat[0:dims[i,0],0:dims[i,1]] = wvlt[i-1]
            else: # Here we have to do the other parts, as they are split in three
                mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1]] = wvlt[i-1][0] # to the right
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1]] = wvlt[i-1][1] # below
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1]] = wvlt[i-1][2] # diagonal
    
    elif dims.shape[-1]==3:
        for i in range(1,dims.shape[0]):
            if i==1: 
                mat[0:dims[i,0],0:dims[i,1],0:dims[i,2]] = wvlt[i-1]
            else:
                mat[0:dims[i,0],0:dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]] = wvlt[i-1][wvKeys[0]]
                mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],0:dims[i,2]] = wvlt[i-1][wvKeys[1]]
                mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]] = wvlt[i-1][wvKeys[2]]
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1],0:dims[i,2]] = wvlt[i-1][wvKeys[3]]
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],0:dims[i,2]] = wvlt[i-1][wvKeys[4]]
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]] = wvlt[i-1][wvKeys[5]]
                mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]] = wvlt[i-1][wvKeys[6]]
        
    return mat, dims, dimOpt, dimLenOpt
        
def mat2wvlt(mat, dims, dimOpt, dimLenOpt):
    wvlt = [[] for i in range(len(dims)-1)]
    for i in range(1,len(wvlt)):
        wvlt[i] = [[] for kk in range(3)]
    if dims.shape[-1]==2:
        for i in range(1,dims.shape[0]):
            if i==1: 
                wvlt[i-1] = mat[0:dims[i,0],0:dims[i,1]]
            else: # Here we have to do the other parts, as they are split in three
                wvlt[i-1][0] = mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1]] # to the right
                wvlt[i-1][1] = mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1]] # below
                wvlt[i-1][2] = mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1]] # diagonal
    elif dims.shape[-1]==3:
        for i in range(1,dims.shape[0]):
            wvKeys=['dad', 'aad', 'daa', 'add', 'ada', 'dda', 'ddd']
            if i==1: 
                wvlt[i-1] = mat[0:dims[i,0],0:dims[i,1],0:dims[i,2]]
            else:
                wvlt[i-1] = {wvKeys[0]: mat[0:dims[i,0],0:dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]],
                wvKeys[1]: mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],0:dims[i,2]],
                wvKeys[2]: mat[0:dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]],
                wvKeys[3]: mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1],0:dims[i,2]],
                wvKeys[4]: mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],0:dims[i,2]],
                wvKeys[5]: mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],0:dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]],
                wvKeys[6]: mat[dimLenOpt[i-1,0]:dimLenOpt[i-1,0]+dims[i,0],dimLenOpt[i-1,1]:dimLenOpt[i-1,1]+dims[i,1],dimLenOpt[i-1,2]:dimLenOpt[i-1,2]+dims[i,2]]}
        
    return wvlt
