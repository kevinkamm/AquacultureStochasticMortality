import time

import numpy as np
import numba as nb

@nb.njit()
def mean(X,axis):
    if axis==0:
        X=X.T
        s=np.zeros((X.shape[0],1),dtype=X.dtype)
        for i in range(0,X.shape[0]):
            s[i,0]=np.mean(X[:,i])
        return s.T
    else:
        s=np.zeros((X.shape[0],1),dtype=X.dtype)
        for i in range(0,X.shape[0]):
            s[i,0]=np.mean(X[i,:])
        return s

@nb.njit()
def linspace(start,end,N):
    dt = (end-start)/(N-1)
    out=dt*np.ones((N,1))
    out[0]=start
    out = np.cumsum(out)
    # dt = (end-start)/(N-1)
    # for i in range(0,N):
    #     out[i]=start+i*dt
    return out



if __name__=="__main__":
    import time 
    d=10000
    X=np.random.randn(d,d).astype(np.float32)
    M=mean(X,0) #compile
    t=linspace(0.0,3.0,10)

    tic = time.time()
    t=linspace(0.0,3.0,d).astype(np.float32).reshape((-1,1))
    # print(t)
    print(t.dtype)
    print(f'NB Elapsed time {time.time()-tic} s')

    tic = time.time()
    t=np.linspace(0,3,d,endpoint=True,dtype=np.float32).reshape((-1,1))
    print(t.dtype)
    print(f'NP Elapsed time {time.time()-tic} s')



    tic = time.time()
    M=mean(X,0)
    print(M.shape)
    print(M.dtype)
    print(np.mean(M))
    print(f'NB Elapsed time {time.time()-tic} s')

    tic = time.time()
    M=np.mean(X,axis=0)
    print(M.dtype)
    print(np.mean(M))
    print(f'NP Elapsed time {time.time()-tic} s')

    tic = time.time()
    M=mean(X,1)
    print(M.dtype)
    print(np.mean(M))
    print(f'NB Elapsed time {time.time()-tic} s')

    tic = time.time()
    M=np.mean(X,axis=1)
    print(M.dtype)
    print(np.mean(M))
    print(f'NP Elapsed time {time.time()-tic} s')