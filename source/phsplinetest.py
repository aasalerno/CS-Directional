
from numpy import *
from numpy.fft import *
from numpy.random import rand
from scipy.interpolate import RectBivariateSpline,BivariateSpline
import pywt
import os.path
import sys
sys.path.append("/home/asalerno/Documents/pyDirectionCompSense/")
sys.path.append("/home/asalerno/Documents/pyDirectionCompSense/source/")
sys.path.append("/home/asalerno/Documents/pyDirectionCompSense/source/wavelet")
os.chdir('/home/asalerno/Documents/pyDirectionCompSense/')
from unwrap2d import i_unwrap_2d
import optimize as opt
import matplotlib.pyplot as plt


#funcs
def compose_wvlt_from_vec(pin,imshape,wavelet='db4',mode='symmetric'):
    #currently limited to square geometry
    w=pywt.Wavelet(wavelet)
    Nl=pywt.dwt_max_level(max(imshape),w.dec_len)
    levlens=[pywt.dwt_coeff_len(max(imshape),w.dec_len,mode=mode)] 
    for j in range(1,Nl):
        levlens.append( pywt.dwt_coeff_len(levlens[-1],w.dec_len,mode=mode) )
    levlens.append(levlens[-1])
    levlens=levlens[::-1]
    p_wvlt=[reshape(array(pin[0:levlens[0]**2],float),(levlens[0],levlens[0]))]
    celem=levlens[0]**2
    for j in range(1,Nl+1):
        t1=reshape(array(pin[celem:celem+levlens[j]**2],float),(levlens[j],levlens[j]))
        celem+=levlens[j]**2
        t2=reshape(array(pin[celem:celem+levlens[j]**2],float),(levlens[j],levlens[j]))
        celem+=levlens[j]**2
        t3=reshape(array(pin[celem:celem+levlens[j]**2],float),(levlens[j],levlens[j]))
        celem+=levlens[j]**2
        p_wvlt.append( (t1,t2,t3) )
    return p_wvlt


def compose_vec_from_wvlt(p_wvlt,imshape,wavelet='db4',mode='symmetric'):
    #currently limited to square geometry
    w=pywt.Wavelet(wavelet)
    Nl=pywt.dwt_max_level(max(imshape),w.dec_len)
    levlens=[pywt.dwt_coeff_len(max(imshape),w.dec_len,mode=mode)] 
    for j in range(1,Nl):
        levlens.append( pywt.dwt_coeff_len(levlens[-1],w.dec_len,mode=mode) )
    levlens.append(levlens[-1])
    levlens=levlens[::-1]
    Na=sum(levlens)
    ret_arr = empty( (Na*Na,),float)
    ret_arr[0:levlens[0]**2] = ravel(p_wvlt[0])
    celem=levlens[0]**2
    for j in range(1,Nl+1):
        for k in range(3):
           ret_arr[celem:celem+levlens[j]**2]=ravel(p_wvlt[j][k])
           celem+=levlens[j]**2
    return ret_arr

def L_sigdiff(p, (t2_array,t1_array), kacq, Sb, wavelet="db4", mode="symmetric", l_wvlt=1e-2, 
             l_tv=1e-2, a=10.0,l_cumR=0.1,s_sigma=0.1):
    p_sb=p[-Sb.shape[0]::]
    expphase=exp(1.j*1.0*sum(p_sb[:,newaxis,newaxis]*Sb,axis=0))
    p_mo=pywt.waverec2(compose_wvlt_from_vec(p[0:-Sb.shape[0]],Sb.shape[1::],wavelet=wavelet,mode=mode),wavelet=wavelet,mode=mode)
    p_mo=where(p_mo<0.0,0.0,p_mo)
    img_est=p_mo*expphase
    k_est=fftshift(fft2(img_est))[t2_array,t1_array]
    d_con = sum( abs(kacq-k_est)**2 )/prod(Sb.shape[1::])
    abs_wvlt = sum( (1/a)*log(cosh(a*p_mo)) )
    abs_tvx = sum( (1/a)*log(cosh(a*(roll(p_mo,-1,axis=-1)-p_mo))) )
    #tvx_kern=array([[-0.5,0,0.5],[-1,0,1],[-0.5,0,0.5]],'float')
    #tvx_kern=array([[0,0,-1.0],[0,0,0],[1,0,0]],'float')    
    #tvy_kern=array([[-0.5,-1,-0.5],[0,0,0],[0.5,1,0.5]],'float')
    #tvy_kern=array([[-1.0,0.0,0.0],[0,0,0],[0.0,0,1.0]],'float')
    #abs_tvx = sum( (1/a)*log(cosh(a*convolve2d(p_mo,tvx_kern,mode='same'))) )
    abs_tvy = sum( (1/a)*log(cosh(a*(roll(p_mo,-1,axis=-2)-p_mo))) )
    #abs_tvy = sum( (1/a)*log(cosh(a*convolve2d(p_mo,tvy_kern,mode='same'))) )
    #cumRdistr = sum( exp(-p_mo**2/(2*s_sigma**2)) )
    return d_con + l_wvlt*abs_wvlt + l_tv*(abs_tvx+abs_tvy) #+ l_cumR*cumRdistr

def grad_L_sigdiff(p,(t2_array,t1_array),kacq,Sb,wavelet="db4",mode="symmetric",l_wvlt=1e-2,l_tv=1e-2,a=10.0,l_cumR=0.1,s_sigma=0.1):
    p_sb=p[-Sb.shape[0]::]
    expphase=exp(1.j*1.0*sum(p_sb[:,newaxis,newaxis]*Sb,axis=0))
    p_mo=pywt.waverec2(compose_wvlt_from_vec(p[0:-Sb.shape[0]],Sb.shape[1::],wavelet=wavelet,mode=mode),wavelet=wavelet,mode=mode)
    p_mo=where(p_mo<0.0,0.0,p_mo)
    img_est=p_mo*expphase
    k_est=fftshift(fft2(img_est))
    kacqfull=zeros(img_est.shape,complex)
    kacqfull[t2_array,t1_array]=kacq
    m=zeros(img_est.shape,float)
    m[t2_array,t1_array]=1.0
    diffimg=conj(expphase)*ifft2(ifftshift(m*(k_est-kacqfull)))
    g_abs_wvlt = tanh(a*p_mo)
    xdiff = roll(p_mo,-1,axis=-1)-p_mo
    ydiff = roll(p_mo,-1,axis=-2)-p_mo
    abs_tvx =  (1/a)*log(cosh(a*xdiff))
    abs_tvy =  (1/a)*log(cosh(a*ydiff))
    #tvx_kern=array([[-0.5,0,0.5],[-1,0,1],[-0.5,0,0.5]],'float')
    #tvx_kern=array([[0,0,-1.0],[0,0,0],[1,0,0]],'float')    
    #tvy_kern=array([[-0.5,-1,-0.5],[0,0,0],[0.5,1,0.5]],'float')
    #tvy_kern=array([[-1.0,0.0,0.0],[0,0,0],[0.0,0,1.0]],'float')
    #abs_tvx = (1/a)*log(cosh(a*convolve2d(p_mo,tvx_kern,mode='same')))
    #abs_tvy = (1/a)*log(cosh(a*convolve2d(p_mo,tvy_kern,mode='same')))
    g_abs_tvx = -tanh(a*xdiff)+tanh(a*roll(xdiff,1,axis=-1))
    #g_abs_tvx = convolve2d(tanh(a*p_mo),tvx_kern[::-1,::-1],mode='same')
    g_abs_tvy = -tanh(a*ydiff)+tanh(a*roll(ydiff,1,axis=-2))
    #g_abs_tvy = convolve2d(tanh(a*p_mo),tvy_kern[::-1,::-1],mode='same')
    #g_cumRdistr = -(p_mo/s_sigma**2)*exp(-p_mo**2/(2.0*s_sigma**2))
    g_mo_img=2.0*diffimg.real+l_wvlt*g_abs_wvlt+l_tv*(g_abs_tvx+g_abs_tvy) #+l_cumR*g_cumRdistr
    g_mo_img=where((p_mo<=0.0)*(g_mo_img>0.0),0.0,g_mo_img)
    g_mo=compose_vec_from_wvlt(pywt.wavedec2(g_mo_img
                                             ,wavelet=wavelet,mode=mode),p_mo.shape,wavelet=wavelet,mode=mode)
    g_sb=sum(sum( 2.0*( 1.0*p_mo*(-1.j)*diffimg ).real[newaxis,:,:]*Sb ,axis=1),axis=-1)
    return append(ravel(g_mo),g_sb)





def L_pmo_sigdiff(p,(t2_array,t1_array),kacq,phasemap,wavelet="db4",mode="symmetric",l_wvlt=1e-2,l_tv=1e-2,a=10.0):
    expphase=exp(1.j*phasemap)
    p_mo=pywt.waverec2(compose_wvlt_from_vec(p,phasemap.shape,wavelet=wavelet,mode=mode),wavelet=wavelet,mode=mode)
    p_mo=where(p_mo<0.0,0.0,p_mo)
    img_est=p_mo*expphase
    k_est=fftshift(fft2(img_est))[t2_array,t1_array]
    d_con = sum( abs(kacq-k_est)**2 )/prod(Sb.shape[1::])
    abs_wvlt = sum( (1/a)*log(cosh(a*p_mo)) )
    abs_tvx = sum( (1/a)*log(cosh(a*(roll(p_mo,-1,axis=-1)-p_mo))) )
    #tvx_kern=array([[-0.5,0,0.5],[-1,0,1],[-0.5,0,0.5]],'float')
    #tvy_kern=array([[-0.5,-1,-0.5],[0,0,0],[0.5,1,0.5]],'float')
    #abs_tvx = sum( (1/a)*log(cosh(a*convolve2d(p_mo,tvx_kern,mode='same'))) )
    abs_tvy = sum( (1/a)*log(cosh(a*(roll(p_mo,-1,axis=-2)-p_mo))) )
    #abs_tvy = sum( (1/a)*log(cosh(a*convolve2d(p_mo,tvy_kern,mode='same'))) )
    return d_con + l_wvlt*abs_wvlt + l_tv*(abs_tvx+abs_tvy)

def L_psb_sigdiff(p,(t2_array,t1_array),kacq,p_mo,l_wvlt=1e-2,l_tv=1e-2,a=10.0):
    p_sb=p[-Sb.shape[0]::]
    expphase=exp(1.j*sum(p_sb[:,newaxis,newaxis]*Sb,axis=0))
    img_est=p_mo*expphase
    k_est=fftshift(fft2(img_est))[t2_array,t1_array]
    d_con = sum( abs(kacq-k_est)**2 )/prod(Sb.shape[1::])
    abs_wvlt = sum( (1/a)*log(cosh(a*p_mo)) )
    abs_tvx = sum( (1/a)*log(cosh(a*(roll(p_mo,-1,axis=-1)-p_mo))) )
    #tvx_kern=array([[-0.5,0,0.5],[-1,0,1],[-0.5,0,0.5]],'float')
    #tvy_kern=array([[-0.5,-1,-0.5],[0,0,0],[0.5,1,0.5]],'float')
    #abs_tvx = sum( (1/a)*log(cosh(a*convolve2d(p_mo,tvx_kern,mode='same'))) )
    abs_tvy = sum( (1/a)*log(cosh(a*(roll(p_mo,-1,axis=-2)-p_mo))) )
    #abs_tvy = sum( (1/a)*log(cosh(a*convolve2d(p_mo,tvy_kern,mode='same'))) )
    return d_con + l_wvlt*abs_wvlt + l_tv*(abs_tvx+abs_tvy)


def grad_L_pmo_sigdiff(p,(t2_array,t1_array),kacq,phasemap,wavelet="db4",mode="symmetric",l_wvlt=1e-2,l_tv=1e-2,a=10.0):
    expphase=exp(1.j*phasemap)
    p_mo=pywt.waverec2(compose_wvlt_from_vec(p,phasemap.shape,wavelet=wavelet,mode=mode),wavelet=wavelet,mode=mode)
    p_mo=where(p_mo<0.0,0.0,p_mo)
    img_est=p_mo*expphase
    k_est=fftshift(fft2(img_est))
    kacqfull=zeros(img_est.shape,complex)
    kacqfull[t2_array,t1_array]=kacq
    m=zeros(img_est.shape,float)
    m[t2_array,t1_array]=1.0
    diffimg=conj(expphase)*ifft2(ifftshift(m*(k_est-kacqfull)))
    g_abs_wvlt = tanh(a*p_mo)
    xdiff = roll(p_mo,-1,axis=-1)-p_mo
    ydiff = roll(p_mo,-1,axis=-2)-p_mo
    abs_tvx =  (1/a)*log(cosh(a*xdiff))
    abs_tvy =  (1/a)*log(cosh(a*ydiff))
    #tvx_kern=array([[-0.5,0,0.5],[-1,0,1],[-0.5,0,0.5]],'float')
    #tvy_kern=array([[-0.5,-1,-0.5],[0,0,0],[0.5,1,0.5]],'float')
    #abs_tvx = (1/a)*log(cosh(a*convolve2d(p_mo,tvx_kern,mode='same')))
    #abs_tvy = (1/a)*log(cosh(a*convolve2d(p_mo,tvy_kern,mode='same')))
    g_abs_tvx = -tanh(a*xdiff)+tanh(a*roll(xdiff,1,axis=-1))
    #g_abs_tvx = convolve2d(tanh(a*p_mo),tvx_kern[::-1,::-1],mode='same')
    #g_abs_tvy = -tanh(a*abs_tvy)+tanh(a*roll(abs_tvy,1,axis=-2))
    #g_abs_tvy = convolve2d(tanh(a*p_mo),tvy_kern[::-1,::-1],mode='same')
    g_abs_tvy = -tanh(a*ydiff)+tanh(a*roll(ydiff,1,axis=-2))
    g_mo_img=2.0*diffimg.real+l_wvlt*g_abs_wvlt+l_tv*(g_abs_tvx+g_abs_tvy)
    g_mo_img=where((p_mo<=0.0)*(g_mo_img>0.0),0.0,g_mo_img)
    g_mo=compose_vec_from_wvlt(pywt.wavedec2(g_mo_img
                                             ,wavelet=wavelet,mode=mode),p_mo.shape,wavelet=wavelet,mode=mode)
    return ravel(g_mo)

def grad_L_psb_sigdiff(p,(t2_array,t1_array),kacq,p_mo,l_wvlt=1e-2,l_tv=1e-2,a=10.0):
    p_sb=p[-Sb.shape[0]::]
    expphase=exp(1.j*sum(p_sb[:,newaxis,newaxis]*Sb,axis=0))
    img_est=p_mo*expphase
    k_est=fftshift(fft2(img_est))
    kacqfull=zeros(img_est.shape,complex)
    kacqfull[t2_array,t1_array]=kacq
    m=zeros(img_est.shape,float)
    m[t2_array,t1_array]=1.0
    diffimg=conj(expphase)*ifft2(ifftshift(m*(k_est-kacqfull)))
    g_sb=sum(sum( 2.0*( p_mo*(-1.j)*diffimg ).real[newaxis,:,:]*Sb ,axis=1),axis=-1)
    g_sb=g_sb/max(abs(g_sb))
    return g_sb

#generate phantom and kfull
Mo=zeros((256,256),float)
p_cen=[[45,0],[-45,-52],[-45,52]]
p_rad=[50,50,50]
p_mo=[1.2,1.0,0.8]
p_r1=[1.0,2.0,4.0]
for j in range(len(p_cen)):
    r = sqrt( ((arange(Mo.shape[0])-Mo.shape[0]/2) - p_cen[j][0])[:,newaxis]**2 + \
              ((arange(Mo.shape[1])-Mo.shape[1]/2) - p_cen[j][1])[newaxis,:]**2 )
    exparg = -(p_rad[j]-r)/0.2
    try:
        Is = where(exparg<15,1.0/(1.0+exp(exparg)),0.0)
    except RuntimeWarning:
        pass #overflow inside where() is fine
    Mo += p_mo[j]*Is

phasemap=zeros(Mo.shape,float)
xind,yind=meshgrid(arange(Mo.shape[0])-Mo.shape[0]/2,arange(Mo.shape[0])-Mo.shape[0]/2)
phasemap = (xind-20)**2 + 0.2*yind**2 + 6*(xind-20)*yind + 4*(xind-20) - 2*yind   
phasemap = phasemap*10*pi/max(abs(phasemap).flat)
kfull = fftshift(fft2(Mo*exp(1.j*phasemap)))

#undersampling
yc,xc=mgrid[-Mo.shape[0]/2:Mo.shape[0]/2,-Mo.shape[1]/2:Mo.shape[1]/2]
rc=sqrt(yc**2+xc**2)
randacq=rand(Mo.shape[0],Mo.shape[1])
randacq=where(rc<10,1.0,randacq)
nsamp=0.75*prod(Mo.shape)
randacq_thresh=sort(ravel(randacq))[prod(Mo.shape)-nsamp]
acqmask=where(randacq<randacq_thresh,0.0,1.0)
t2_array,t1_array=nonzero(acqmask)
t2_array=t2_array #-Mo.shape[0]/2
t1_array=t1_array #-Mo.shape[1]/2
kacq=kfull[t2_array,t1_array]



#now try to solve by grad descent
kacq2d=zeros(kfull.shape,complex)
kacq2d[t2_array,t1_array]=kacq
imgus=ifft2(ifftshift(kacq2d))
imgwvlt=pywt.wavedec2(abs(imgus),wavelet='db4',mode='symmetric')
Ny,Nx=imgus.shape

#fit smoothed phase, evaluate phase basis functions
ph_splfit=RectBivariateSpline(arange(imgus.shape[0]),arange(imgus.shape[1]),1.0*i_unwrap_2d(angle(imgus)),s=1.7e5)
spl_tx=ph_splfit.tck[0]
spl_ty=ph_splfit.tck[1]
spl_kx,spl_ky=ph_splfit.degrees
spl_ncoeffs=ph_splfit.get_coeffs().shape[0]
Sb=zeros([spl_ncoeffs,Nx,Ny],float)
spl_BVS=BivariateSpline()
spl_BVS.degrees=(spl_kx,spl_ky)
for j in range(spl_ncoeffs):
    spl_coeffs=zeros(spl_ncoeffs,float)
    spl_coeffs[j]=1.0
    spl_BVS.tck=(spl_tx,spl_ty,spl_coeffs)
    spl_bfunc=spl_BVS.ev(arange(Nx*Ny)%Ny,arange(Nx*Ny)/Ny)
    spl_bfunc.shape=(Nx,Ny)
    Sb[j,:,:]=transpose(spl_bfunc[:,:]) #unfortunately, spline funcs use transposed orientation convention


phest=ph_splfit.get_coeffs()
p0=append(compose_vec_from_wvlt(imgwvlt,imgus.shape,wavelet='db4',mode='symmetric'),phest)
l_w=[0.2]
for c_l_w in l_w:
    args=((t2_array,t1_array),kacq,Sb,"db4","symmetric",0.0,c_l_w,20.0)
    for j in range(10):
        optresult = optp.minimize(L_sigdiff,p0,args=args,method="CG",jac=grad_L_sigdiff,tol=1e-1,
                                  options={'maxiter': 8, 'gtol': 1e-2,'disp': True, 'xtol':1.0, 'c':0.6, 'alpha_0': 0.1, 'lineSearchItnLim':20 })
        pfit=optresult['x']
        p0=pfit.copy()
        phasefit=sum(pfit[-Sb.shape[0]::,newaxis,newaxis]*Sb,axis=0)
        expphase=exp(1.j*1.0*phasefit)
        p_mo=pywt.waverec2(compose_wvlt_from_vec(pfit[0:-Sb.shape[0]],Sb.shape[1::],wavelet='db4',mode='symmetric'),wavelet='db4',mode='symmetric')
        plt.subplot(1,2,1); plt.imshow(abs(p_mo))
        plt.subplot(1,2,2); plt.imshow(phasefit)
        plt.show()









phest=ph_splfit.get_coeffs()
phest_map=sum(phest[:,newaxis,newaxis]*Sb[:,:,:],axis=0)
p0=append(compose_vec_from_wvlt(imgwvlt,imgus.shape,wavelet='db4',mode='symmetric'),phest)
l_w=0.2
p_mo_wvlt=imgwvlt
for j in range(12):
    #phase estimate
    p_mo=pywt.waverec2(p_mo_wvlt,wavelet='db4',mode='symmetric')
    args=((t2_array,t1_array),kacq,p_mo,l_w,l_w,10.0)
    optresult = optp.minimize(L_psb_sigdiff,phest,args=args,method="CG",jac=grad_L_psb_sigdiff,tol=1e-1,
                              options={'maxiter': 10, 'gtol': 1e-2,'disp': True, 'xtol':1.0, 'c':0.6, 'alpha_0': 0.05, 'lineSearchItnLim':20 })
    phest=optresult['x']
    phest_map=sum(phest[:,newaxis,newaxis]*Sb[:,:,:],axis=0)
    #intensity estimate
    args=((t2_array,t1_array),kacq,phest_map,"db4","symmetric",l_w,l_w,10.0)
    p_mo_vec=compose_vec_from_wvlt(p_mo_wvlt,phest_map.shape,wavelet='db4',mode='symmetric')
    optresult = optp.minimize(L_pmo_sigdiff,p_mo_vec,args=args,method="CG",jac=grad_L_pmo_sigdiff,tol=1e-1,
                              options={'maxiter': 10, 'gtol': 1e-2,'disp': True, 'xtol':1.0, 'c':0.6, 'alpha_0': 0.1, 'lineSearchItnLim':20 })
    p_mo_vec=optresult['x']
    p_mo_wvlt=compose_wvlt_from_vec(p_mo_vec,phest_map.shape,wavelet='db4',mode='symmetric')
    plt.subplot(1,2,1); plt.imshow(abs(p_mo))
    plt.subplot(1,2,2); plt.imshow(phest_map)
    plt.show()


pfit=optresult['x']
phasefit=sum(pfit[-Sb.shape[0]::,newaxis,newaxis]*Sb,axis=0)
expphase=exp(1.j*phasefit)
p_mo=pywt.waverec2(compose_wvlt_from_vec(pfit[0:-Sb.shape[0]],Sb.shape[1::],wavelet='db4',mode='symmetric'),wavelet='db4',mode='symmetric')
imgfit=p_mo*expphase

plt.imshow(abs(p_mo)); plt.colorbar()
plt.figure(2); plt.imshow(phasefit); plt.colorbar()
plt.show()

#grad test
p0mod=p0.copy()
LvalA=L_sigdiff(p0mod,(t2_array,t1_array),kacq,Sb,"db4","symmetric",l_wvlt=0.1,l_tv=0.1,a=10.0)
gcalc=grad_L_sigdiff(p0mod,(t2_array,t1_array),kacq,Sb,"db4","symmetric",l_wvlt=0.1,l_tv=0.1,a=10.0)
for j in [90,1000,9000,31500,-60,-5]:
    p0mod[j]+=0.001
    LvalB=L_sigdiff(p0mod,(t2_array,t1_array),kacq,Sb,"db4","symmetric",l_wvlt=0.1,l_tv=0.1,a=10.0)
    p0mod[j]-=0.001
    gvalest=(LvalB-LvalA)/0.001
    print "estimate: %f, calc: %f"%(gvalest,gcalc[j])



#grad test 2
p_mo_vec=compose_vec_from_wvlt(p_mo_wvlt,phest_map.shape,wavelet='db4',mode='symmetric')
p_mo_vec_mod=p_mo_vec.copy()
gcalc=grad_L_pmo_sigdiff(p_mo_vec,(t2_array,t1_array),kacq,phest_map,"db4","symmetric",0*l_w,l_w,10.0)
LvalA=L_pmo_sigdiff(p_mo_vec,(t2_array,t1_array),kacq,phest_map,"db4","symmetric",0*l_w,l_w,10.0)
for j in [90,1000,9000]:
    p_mo_vec_mod[j]+=0.0001
    LvalB=L_pmo_sigdiff(p_mo_vec_mod,(t2_array,t1_array),kacq,phest_map,"db4","symmetric",0*l_w,l_w,10.0)
    p_mo_vec_mod[j]-=0.0001
    gvalest=(LvalB-LvalA)/0.0001
    print "estimate: %f, calc: %f"%(gvalest,gcalc[j])


