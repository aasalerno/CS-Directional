ind = np.arange(N)  # the x locations for the groups
width = 0.8       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, rmses, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('RMSE')
ax.set_title('RMSE for Different Percentage of Samp')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(labs)
plt.show()


for i in range(N[0]):
    data[i,:,:] = np.fft.fftshift(k[i,:,:])*tf.fft2c(im[i,:,:], ph=ph_ones)
    dataDir[i,:,:] = np.fft.fftshift(kDir[i,:,:])*tf.fft2c(im[i,:,:], ph=ph_ones)
    dataFull[i,:,:] = np.fft.fftshift(tf.fft2c(im[i,:,:], ph=ph_ones))
    im_scan_wph = tf.ifft2c(data[i,:,:], ph=ph_ones)
    im_scan_wphDir = tf.ifft2c(dataDir[i,:,:], ph=ph_ones)
    ph_scan[i,:,:] = np.angle(gaussian_filter(im_scan_wph.real,2) +  1.j*gaussian_filter(im_scan_wph.imag,2))
    ph_scanDir[i,:,:] = np.angle(gaussian_filter(im_scan_wphDir.real,2) +  1.j*gaussian_filter(im_scan_wphDir.imag,2))
    ph_scan[i,:,:] = np.exp(1j*ph_scan[i,:,:])
    ph_scanDir[i,:,:] = np.exp(1j*ph_scanDir[i,:,:])
    im_scan[i,:,:] = tf.ifft2c(data[i,:,:], ph=ph_scan[i,:,:])
    im_scanDir[i,:,:] = tf.ifft2c(dataDir[i,:,:], ph=ph_scanDir[i,:,:])





for q in xrange(dirs.shape[0]):
    r = inds[q,:]
    
    # Assume the data is coming in as image space data and pull out what we require
    Iq = x0[q,:]
    Ir = x0[r,:]
    
    #A = np.zeros(np.hstack([r.shape,3]))
    Irq = Ir - Iq # Iq will be taken from Ir for each part of axis 0
    #Aleft = np.linalg.solve((A.T*A),A.T)
    #beta = np.zeros(np.hstack([Iq.shape,3]))

    Gdiffsq[q,:] = np.dot(np.dot(Irq,M[q,:,:]),Irq)