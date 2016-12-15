from numpy import *
import sys
sys.path.append("/home/asalerno/Documents/pyDirectionCompSense/source/")#path to modified optimize directory here
#import scipy.optimize as opt
import optimize as opt                                              #load by proper optimize directory name here

def f_Ab(p,A,b):
    return sum( (b-dot(A,p))**2 )

def g_f_Ab(p,A,b):
    gp = empty((len(p),),float)
    resids = 2*(b-dot(A,p))
    for j in range(len(gp)):
        gp[j] = dot(resids,(-A[:,j]))
    return gp

b=array([1,0,-1],float)
A=array([[1,2,0],[2,1,0],[0,1,2]],float)
p0=array([0,0,0],float)
args=(A,b)
optresult = opt.minimize(f_Ab,p0,args=args,method="CG",jac=g_f_Ab,
                         tol=1e-8,options={'maxiter': 100, 'gtol': 1e-6,'disp': True, 'c':0.6,'alpha_0':0.6,'xtol': 1e-6,'lineSearchItnLim': 30 })