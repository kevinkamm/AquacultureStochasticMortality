import numpy as np
import numexpr as ne
from typing import Tuple, Optional
from numpy.typing import DTypeLike, ArrayLike

def brownianMotion(T:float,N:int,M:int,
                   d:Optional[int]=1,
                   rho:Optional[float]=None,
                   antithetic:Optional[bool]=False,
                   dtype:Optional[DTypeLike]=np.float64,
                   rng:Optional[np.random.Generator]=np.random.default_rng())\
    -> np.ndarray:

    dt=T/(N-1)
    dW=np.sqrt(dt)*rng.standard_normal((N-1,M,d),dtype=dtype)#4x faster than np.random.randn
    W=np.zeros((N,M,d),dtype=dtype)
    W[1:,:,:]=np.cumsum(dW,axis=0)

    if rho is not None:
        if d==2:
            if np.size(rho)==1:
                r=rho
            else:
                r=rho[0,1]
            W=np.stack([W[:,:,0],r*W[:,:,0]+np.sqrt(1-r**2)*W[:,:,1]],axis=2)
        else:
            L=np.linalg.cholesky(rho).reshape((1,1,d,d))
            W=np.squeeze(L@W.reshape((N,M,d,1)),axis=3)
    if antithetic:
        return np.concatenate([W,-W],axis=1)
    else:
        return W


def schwartz2factor(delta0:float,S0:float,r:float,sigma1:float,sigma2:float,kappa:float,alpha:float,l:float,t:np.ndarray,W:np.ndarray,
                    dtype:Optional[DTypeLike]=np.float64)\
    ->Tuple[np.ndarray,np.ndarray]:
    N=t.size
    dt=t[-1]/(N-1)

    W1=W[:,:,0]
    dW2=np.diff(W[:,:,1],1,axis=0) # shape=(time,simulations)

    "Ornstein-Uhlenbeck"
    expkt=np.exp(-kappa*t)
    expkt2=np.exp(kappa*t)
    sInt=stochInt(expkt2,dW2,dtype=dtype)
    delta = delta0*expkt+(alpha-l/kappa)*(1-expkt)+expkt*sigma2*sInt 

    # no difference to numpy speed:
    # expkt2=np.exp(kappa*t)
    # sInt=stochInt(expkt2,dW2,dtype=dtype)
    # delta = ne.evaluate("delta0*exp(-kappa*t)+(alpha-l/kappa)*(1-exp(-kappa*t))+exp(-kappa*t)*sigma2*sInt")

    "Geometrical Brownian Motion"
    I_delta_dt = lebInt(delta,dt,N,dtype=dtype)
    S=S0*np.exp((r-sigma1**2/2)*t-I_delta_dt+sigma1*W1)
    return np.stack([S,delta],axis=2)


def stochInt(f:np.ndarray,dW:np.ndarray,
             dtype:Optional[DTypeLike]=np.float64)\
    -> np.ndarray:
    [N,M]=dW.shape
    I_f_dW = np.zeros((N+1,M),dtype=dtype)
    I_f_dW[1:,:]=np.cumsum(f[:-1].reshape(N,-1)*dW,axis=0)
    return I_f_dW

def lebInt(f:np.ndarray,dt:float,N:int,
             dtype:Optional[DTypeLike]=np.float64)\
    -> np.ndarray:
    [_,M]=f.shape
    I_dt = np.zeros((N,M),dtype=dtype)
    I_dt[1:,:]=np.cumsum(f[:-1,:]*dt,axis=0)
    return I_dt

class Commodity():
    def __init__(self,
                 params:ArrayLike,
                 t:np.ndarray,
                 dtype:DTypeLike=np.float32,
                 rng:np.random.Generator=np.random.default_rng()):
        self.params=np.array(params,dtype=dtype)
        self.dtype=dtype
        self.rng=rng
        self.t=t

    def sample(self,batch_size:int)->np.ndarray:
        pass

    def setgen(self,rng:np.random.Generator=np.random.default_rng()):
        self.rng=rng

class Schwartz2Factor(Commodity):
    def __init__(self, 
                 params: ArrayLike, 
                 t:np.ndarray,
                 dtype: DTypeLike = np.float32, 
                 rng: np.random.Generator = np.random.default_rng()):
        super().__init__(params, t, dtype, rng)

    def sample(self,batch_size:int)->np.ndarray:
        W=brownianMotion(self.t[-1],self.t.size,batch_size,d=2,rho=self.params[-3],dtype=self.dtype,rng=self.rng)
        return schwartz2factor(*self.params[-2:],*self.params[:-3],self.t,W,dtype=self.dtype)

if __name__=="__main__":
    import time
    dtype=np.float32

    T=3.0
    N=3*24*10
    M=10
    rho=.9

    t=np.linspace(0,T,N,endpoint=True,dtype=dtype).reshape(N,1)

    rng=np.random.default_rng(0)
    # r,sigma1,sigma2,kappa,alpha,lambda,rho,d0,S0
    params=[.03,1.0,1.0,1.0,1.0,0.5,rho,.03,100]
    print(params[-2:])
    print(params[:-3])
    C=Schwartz2Factor(params,t,dtype=dtype,rng=rng)
    tic = time.time()
    sd=C.sample(M)
    toc = time.time()
    print(f'Elapsed time {toc-tic} s')
    print(sd)

    print(sd.shape)
    print(sd.dtype)

    tic = time.time()
    brownianMotion(T,N,M,d=2,rho=rho,rng=rng)
    toc = time.time()
    print(f'Elapsed time {toc-tic} s')