import numpy as np
import tensorflow as tf
from typing import Union, Optional
from joblib import Parallel,delayed
from psutil import cpu_count
import tensorflow_probability as tfp
import time
tfd = tfp.distributions

class Mortality():
    def __init__(self,
                 t:Union[np.ndarray,tf.Tensor],
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None):
        self.t=t
        self.dtype=t.dtype
        self.isStoch=False
        self.d=0
        if rng is None:
            if type(t)==np.ndarray:
                self.rng=np.random.default_rng()
            else:
                self.rng=tf.random.Generator.from_seed(0)
        else:
            self.rng=rng

    def sample(self,batch_size:int):
        pass

    def setgen(self,gen:Union[tf.random.Generator,np.random.Generator]):
        self.rng=gen

    def populationSize(self,M:np.ndarray):
        pass

    def treatmentCost(self,M):
        return 0


class ConstMortatlity(Mortality):
    def populationSizeNP(n0:int,M:int,m:float,t:np.ndarray):
        return n0*np.exp(-m*t)

    @tf.function
    def populationSizeTF(n0:int,M:int,m:float,t:tf.Tensor):
        return n0*tf.math.exp(-m*t)
    
    def __init__(self, 
                 t:Union[np.ndarray,tf.Tensor],
                 n0:int,
                 m:float,
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None):
        super().__init__(t,rng)
        if type(t)==np.ndarray:
            self.m=m
            self.n0=n0
            self._populationSize=ConstMortatlity.populationSizeNP
        else:
            self.m=tf.constant(m,dtype=self.dtype)
            self.n0=tf.constant(n0,dtype=self.dtype)
            self._populationSize=ConstMortatlity.populationSizeTF

    def populationSize(self,
                       M:np.ndarray # host process
                       ):
        return self._populationSize(self.n0,M,self.m,self.t)
    

class HostParasite(Mortality):
    def hostParasiteNP(params,H0,P0,T,N,dtype):
        # mu=params[0]
        # alpha=params[1]
        # lambda=params[2]
        # b=params[3]
        
        dt=T/(N-1)
        H=np.zeros((N,1),dtype=dtype)
        P=np.zeros((N,1),dtype=dtype)
        H[0,:]=H0
        P[0,:]=P0
        for ti in range(0,N-1):
            Htmp=H[ti]
            Ptmp=P[ti]
            PHtmp=Ptmp/Htmp

            H[ti+1,:]=Htmp-(params[0]*Htmp+params[1]*Ptmp)*dt
            P[ti+1,:]=Ptmp+(params[2]*Htmp/H0-(params[3]+params[0])-params[1]*PHtmp)*Ptmp *dt

        return H,P

    
    @tf.function
    def hostParasiteTF(params,H0,P0,T,N,dtype):
        # mu=params[0]
        # alpha=params[1]
        # lambda=params[2]
        # b=params[3]
        
        dt=T/(N-1)
        H=tf.zeros((N,1),dtype=dtype)
        P=tf.zeros((N,1),dtype=dtype)
        H[0,:]=H0
        P[0,:]=P0
        for ti in range(0,N-1):
            Htmp=H[ti]
            Ptmp=P[ti]
            PHtmp=Ptmp/Htmp

            H[ti+1,:]=Htmp-(params[0]*Htmp+params[1]*Ptmp)*dt
            P[ti+1,:]=Ptmp+(params[2]*Htmp/H0-(params[3]+params[0])-params[1]*PHtmp)*Ptmp *dt

        return H,P
    
    
    def __init__(self,
                 t:Union[np.ndarray,tf.Tensor],
                 params:Union[np.ndarray,tf.Tensor],
                 beta,
                 H0:float,
                 P0:float,
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None):
        super().__init__(t,rng)
        self.isStoch=True
        self.d=2
        self.num_cores=cpu_count(logical=False)
        self.threshold=0.5
        if type(t)==np.ndarray:
            self.params=np.array(params,dtype=self.dtype)
            self.betarnd=lambda x:self.rng.beta(*beta,x).astype(self.dtype)
            self.unifrnd=lambda x:self.rng.uniform(low=0.995,high=1.0,size=x).astype(self.dtype)
            self.H0=np.array(H0,dtype=self.dtype)
            self.P0=np.array(P0,dtype=self.dtype)
        else:
            self.params=tf.constant(params,dtype=self.dtype)
            self.H0=tf.constant(H0,dtype=self.dtype)
            self.P0=tf.constant(P0,dtype=self.dtype)
            self.betarnd=lambda x:tfd.Beta(*beta).sample(x)
            self.unifrnd=lambda x:self.rng.uniform(x,minval=0.995,maxval=1.0,dtype=self.dtype)

    def simHostParasiteNP(self,X,Y):
        N=self.t.shape[0]
        H=np.zeros_like(self.t)
        P=np.zeros_like(self.t)
        M=np.zeros(N,dtype=np.int32)
        T=self.t[-1]
        currT=self.t[-1]
        currt=self.t.copy()
        currN=N
        currH0=self.H0
        currP0=self.P0
        iLast=0
        for _ in range(0,N): #can maximal jump at each time
            if currN<=2:
                break
            [Htmp,Ptmp]=HostParasite.hostParasiteNP(self.params,currH0,currP0,currT,currN,self.dtype)
            ind=np.where(Ptmp/Htmp>=self.threshold)[0]
            if np.size(ind)>0:
                i=ind[0]
                P[iLast:iLast+i]=Ptmp[:i]
                H[iLast:iLast+i]=Htmp[:i]
                M[iLast+i-1]=1
                currH0=Htmp[i]*X[iLast+i]
                # print(currH0)
                currP0=Ptmp[i]*Y[iLast+i]*0.8
                if currP0/currH0>=0.5:
                    print(':()')
                iLast=iLast+i
                currT=T-currt[i-1]
                currt=currt[i:]
                currN=np.size(currt)
            else:
                break
        [Htmp,Ptmp]=HostParasite.hostParasiteNP(self.params,currH0,currP0,currT,currN,self.dtype)

        P[iLast:]=Ptmp
        H[iLast:]=Htmp
        # print(np.sum(M))
        return H,P,M
    def simHostParasiteTF(self):
        pass

    def simHostParasite(self,X,Y):
        if type(t)==np.ndarray:
            return self.simHostParasiteNP(X,Y)
        else:
            return self.simHostParasiteTF(X,Y)

    def sample(self,batch_size:int):
        N=self.t.shape[0]
        tic=time.time()
        X=self.unifrnd((N,batch_size))#avoids issues with rng generator in parallel
        Y=self.betarnd((N,batch_size))
        ctime=time.time()-tic
        print(f'Elapsed time {ctime} s {X[0,0]}')
        tic=time.time()
        X=self.unifrnd((N,batch_size))#avoids issues with rng generator in parallel
        Y=self.betarnd((N,batch_size))
        ctime=time.time()-tic
        print(f'Elapsed time {ctime} s {X[0,0]}')
        # tic=time.time()
        # X=self.unifrnd((N,batch_size)).numpy() #avoids issues with rng generator in parallel
        # Y=self.betarnd((N,batch_size)).numpy()
        # ctime=time.time()-tic
        # print(f'Elapsed time {ctime} s')
        # tic=time.time()
        # X=self.unifrnd((N,batch_size)).numpy() #avoids issues with rng generator in parallel
        # Y=self.betarnd((N,batch_size)).numpy()
        # ctime=time.time()-tic
        # print(f'Elapsed time {ctime} s')
        # H,P,M = zip(*Parallel(n_jobs=4,backend='threading')(delayed(self.simHostParasite)(X[:,wi],Y[:,wi]) for wi in range(0,batch_size) ) ) 
        for wi in range(0,batch_size):
            H,P,M = self.simHostParasite(X[:,wi],Y[:,wi])
        H,P,M = self.simHostParasite(X[:,0],Y[:,0])
        return H,P,M

if __name__=="__main__":
    T=3.0
    N=int(T*4*12+1)
    batch_size=100
    dtype=np.float32
    t=np.linspace(0,T,N,endpoint=True,dtype=dtype).reshape((-1,1))
    # t=tf.constant(t)
    params=[0.05,0.1,8.71,0.05]
    beta=[0.0835,0.0244]
    H0=10000.0
    P0=1

    mort = HostParasite(t,params,beta,H0,P0)
    tic=time.time()
    H,P,M=mort.sample(batch_size)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s')
    