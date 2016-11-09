
wvlt=tf.xfm(img,mode="zpd")
wvltimg,ax=tf.toMatrix(wvlt)
plt.imshow(wvltimg)
plt.show()

wrec=tf.ixfm(wvlt)
plt.imshow(wrec)
plt.show()

plt.imshow(img-wrec); plt.colorbar()
plt.show()




callist=[]
estlist=[]
delta=0.05
gcal=grads.gXFM(im_dc,N)
for gind in range(0,64):
    im_dcmod=im_dc.copy()
    im_dcmod[gind]+=delta
    gest=(optfun(im_dcmod,N,TVWeight,XFMWeight,data,k,strtag,ph,a=a)-optfun(im_dc,N,TVWeight,XFMWeight,data,k,strtag,ph,a=a))/delta
    print "estimate: %f, calcd: %f"%(gest,gcal.flat[gind])
    callist.append(gcal.flat[gind])
    estlist.append(gest)


plt.plot(callist,estlist,'o-')
plt.show()