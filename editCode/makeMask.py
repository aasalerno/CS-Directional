def signalMask(im, thresh=0.1, iters = None):
    if not iters:
        iters = int(0.1*x.shape[0])
    
    mask = np.zeros(im.shape)
    highVal = np.where(im>thresh)
    mask[highVal] = 1

    yLen,xLen = mask.shape
    output = mask.copy()
    for iter in xrange(iters):
        #plt.imshow(output)
        #plt.show()
        for y in xrange(yLen):
            for x in xrange(xLen):
                if (y > 0        and mask[y-1,x]) or \
                (y < yLen - 1 and mask[y+1,x]) or \
                (x > 0        and mask[y,x-1]) or \
                (x < xLen - 1 and mask[y,x+1]): 
                #print('TRUE')
                output[y,x] = 1
        mask = output.copy()

    mask = ndimage.filters.gaussian_filter(mask,int(iters*0.5))
    return mask