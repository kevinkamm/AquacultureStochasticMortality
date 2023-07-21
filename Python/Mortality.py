import numpy as np
import tensorflow as tf
from typing import Union, Optional
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
    def hostParasiteNP(params,H0,P0,T,N,batch_size,unifrnd,betarnd,threshold,dtype):
        
        dt=T/(N-1)
        H=np.zeros((N,batch_size),dtype=dtype)
        P=np.zeros((N,batch_size),dtype=dtype)
        M=np.zeros((N,batch_size),dtype=dtype)
        H[0,:]=H0
        P[0,:]=P0
        for ti in range(0,N-1):
            Htmp=H[ti]
            Ptmp=P[ti]
            PHtmp=Ptmp/Htmp
            ind = PHtmp >= threshold
            hits = np.sum(ind)
            M[ti,ind]=1
            X=unifrnd((hits,))
            Y=betarnd((hits,))
            Htmp[ind]=X*Htmp[ind]
            Ptmp[ind]=(0.1+0.8*Y)*Ptmp[ind] # something between [0.1,0.8] effectiveness
            PHtmp=Ptmp/Htmp

            H[ti+1,:]=Htmp-(params[0]*Htmp+params[1]*Ptmp)*dt
            P[ti+1,:]=Ptmp+(params[2]*Htmp/H0-(params[3]+params[0])-params[1]*PHtmp)*Ptmp *dt

        M=np.cumsum(M,axis=0)
        return np.stack([H,P,M],axis=2)
    
    @tf.function(jit_compile=True)
    def hostParasiteTF(params,H0,P0,T,N,batch_size,X,Y,threshold,dtype):
        dt=T/(N-1)
        Htmp=H0*tf.ones((1,batch_size),dtype=dtype)
        Ptmp=P0*tf.ones((1,batch_size),dtype=dtype)
        M=tf.zeros_like(Htmp)
        H=Htmp #auto copy
        P=Ptmp
        for ti in range(0,N-1):
            PHtmp=Ptmp/Htmp
            ind = tf.cast(PHtmp >= threshold,dtype=dtype)
            M=tf.concat([M,ind],axis=0)
            Htmp=(1-ind)*Htmp+ind*X[ti,:]*Htmp
            Ptmp=(1-ind)*Ptmp+ind*(0.1+0.8*Y[ti,:])*Ptmp
            PHtmp=Ptmp/Htmp

            Htmp=Htmp-(params[0]*Htmp+params[1]*Ptmp)*dt
            Ptmp=Ptmp+(params[2]*Htmp/H0-(params[3]+params[0])-params[1]*PHtmp)*Ptmp *dt
            H=tf.concat([H,Htmp],axis=0)
            P=tf.concat([P,Ptmp],axis=0)

        M=tf.cumsum(M,axis=0)
        return tf.stack([H,P,M],axis=2)
    
    
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
        self.threshold=0.5
        if type(t)==np.ndarray:
            self.beta=beta
            self.params=np.array(params,dtype=self.dtype)
            self.betarnd=lambda x:self.rng.beta(*beta,x).astype(self.dtype)
            self.unifrnd=lambda x:self.rng.uniform(low=0.995,high=1.0,size=x).astype(self.dtype)
            self.H0=np.array(H0,dtype=self.dtype)
            self.P0=np.array(P0,dtype=self.dtype)
            self._hostParasite=HostParasite.hostParasiteNP
        else:
            self.beta=tf.constant(beta,dtype=self.dtype)
            self.params=tf.constant(params,dtype=self.dtype)
            # self.betarnd=lambda x:tfd.Beta(*beta).sample(x)
            # self.unifrnd=lambda x:self.rng.uniform(x,0.995,1.0,dtype=dtype)
            
            self.H0=tf.constant(H0,dtype=self.dtype)
            self.P0=tf.constant(P0,dtype=self.dtype)
            self._hostParasite=HostParasite.hostParasiteTF

    # def setgen(self,gen:Union[tf.random.Generator,np.random.Generator]):
    #     self.rng=np.random.default_rng()

    def sample(self,batch_size:int):
        if type(self.t)==np.ndarray:
            return self._hostParasite(self.params,self.H0,self.P0,self.t[-1],self.t.shape[0],batch_size,self.unifrnd,self.betarnd,self.threshold,self.dtype)
        else:
            Y=tfd.Beta(*self.beta).sample((self.t.shape[0],batch_size))
            X=self.rng.uniform((self.t.shape[0],batch_size),0.995,1.0,dtype=self.dtype)
            return self._hostParasite(self.params,self.H0,self.P0,self.t[-1],self.t.shape[0],batch_size,X,Y,self.threshold,self.dtype)
    
    def populationSize(self,
                       M:Union[np.ndarray,tf.Tensor] # host process
                       ):
        return M[:,:,0]
    
    def treatmentCost(self,
                       M:Union[np.ndarray,tf.Tensor] # host process
                       ):
        return 0.01*M[:,:,-1]
    
class DetermHostParasite(HostParasite):
    def __init__(self,
                 t:Union[np.ndarray,tf.Tensor],
                 params:Union[np.ndarray,tf.Tensor],
                 beta,
                 H0:float,
                 P0:float,
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None):
        super().__init__(t,params,beta,H0,P0,rng)
        self.isStoch=False
        self.d=0

    def populationSize(self, M):
        if type(self.t)==np.ndarray:
            return np.mean(super().populationSize(M),axis=1,keepdims=True)
        else:
            return tf.reduce_mean(super().populationSize(M),axis=1,keepdims=True)
        
    def treatmentCost(self, M):
        if type(self.t)==np.ndarray:
            return np.mean(super().treatmentCost(M),axis=1,keepdims=True)
        else:
            return tf.reduce_mean(super().treatmentCost(M),axis=1,keepdims=True)
    

if __name__=="__main__":
    # tf.config.set_visible_devices([], 'GPU')
    T=3.0
    N=int(T*2*12)*10
    batch_size=2**12
    dtype=np.float32
    t=np.linspace(0,T,N,endpoint=True,dtype=dtype).reshape((-1,1))
    t=tf.constant(t)
    params=[0.05,0.1,8.71,0.05]
    beta=[0.0835,0.0244]
    H0=10000.0
    P0=1

    mort = DetermHostParasite(t,params,beta,H0,P0)
    tic=time.time()
    HPM=mort.sample(batch_size)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s')
    tic=time.time()
    HPM=mort.sample(batch_size)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s')
    tic=time.time()
    HPM=mort.sample(batch_size)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s')
    