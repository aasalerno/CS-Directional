import numpy as np
import matplotlib.pyplot as plt

def gradTest(f,df,x,argsf,argsdf,h=1e-5):
    df_n = (f(x+h,*argsf) - f(x-h,*argsf))/(2*h) # numerical 
    df_a = df(x,*argsdf)
    import pdb; pdb.set_trace()
    relerr = abs(df_a-df_n)/np.max(np.hstack([abs(df_a).flat,abs(df_n).flat]))
    print('Relative error: %.2e' % np.max(relerr))
    return relerr