import numpy as np
import numba as nb
import tensorflow as tf
from typing import Union, Optional
import tensorflow_probability as tfp
import time
tfd = tfp.distributions

class Mortality():
    def __init__(self,
                 t:Union[np.ndarray,tf.Tensor],
                 isStoch:bool,
                 d:int,
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None,
                 treatmentCosts:float=0.01):
        self.t=t
        self.dtype=t.dtype
        self.isStoch=isStoch
        self.d=d
        self.treatmentCosts=treatmentCosts
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
        return self.treatmentCosts


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
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None,
                 treatmentCosts:float=0.3):
        super().__init__(t,False,0,rng,treatmentCosts)
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
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None,
                 treatmentCosts:float=0.01):

        super().__init__(t,True,2,rng,treatmentCosts)
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
        return self.treatmentCosts*M[:,:,-1]
    
class DetermHostParasite(HostParasite):
    def __init__(self,
                 t:Union[np.ndarray,tf.Tensor],
                 params:Union[np.ndarray,tf.Tensor],
                 beta,
                 H0:float,
                 P0:float,
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None,
                 treatmentCosts:float=0.01):
        super().__init__(t,params,beta,H0,P0,rng,treatmentCosts)
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
        
class Poisson(Mortality):

    @tf.function
    def inhomPoissonTF(t,l,batch_size):

        Nt=tf.zeros((tf.size(t)*batch_size,),dtype=t.dtype)
        LT=l[-1]
        r=tf.random.poisson((batch_size,),LT,dtype=tf.int32)
        njumps=tf.cast(tf.reduce_sum(r),dtype=tf.int32)
        u=tf.random.uniform((njumps,),0,1,dtype=t.dtype)
        a=tf.zeros_like(u,dtype=t.dtype)
        b=t[-1]*tf.ones_like(u,dtype=t.dtype)
        for _ in range(0,10):
            tmp=tfp.math.interp_regular_1d_grid((a+b)/2,t[0,0],t[-1,0],l)/LT
            less=tf.cast(tmp<=u,t.dtype)
            greater=1-less
            a=(a+b)/2 *less + a * (1-less)
            b=(a+b)/2 *greater + b * (1-greater)
        Ftinv=tf.reshape((a+b)/2,(-1,1))
        tI=tf.cast(tf.argmax(Ftinv<tf.transpose(t),axis=1),dtype=tf.int32)
        ind=(tf.size(t)*tf.reshape(tf.range(0,batch_size,dtype=tf.int32),(-1,1)))+tf.RaggedTensor.from_row_lengths(tI,r)
        
        ind=tf.reshape(tf.concat(ind.to_tensor(),0),(-1,1))
        Nt=tf.tensor_scatter_nd_add(Nt,ind,tf.ones(shape=(tf.size(ind)),dtype=t.dtype)) #add a lot of ones to t=0, <- assume no jumps at t=0 and remove later
        Nt=tf.transpose(tf.reshape(Nt,(batch_size,tf.size(t)))) #row major -> also better memory layout for cumsum
        return tf.concat([tf.zeros((1,batch_size),dtype=t.dtype),tf.cumsum(Nt[1:,:],0)],axis=0)
    
    def inhomPoissonNP(t,l,batch_size,rng):
        Lfunc= lambda x: np.interp(x,t.flatten(),l)

        t=t.reshape((-1,1))
        Nt=np.zeros((t.size,batch_size),dtype=t.dtype)
        LT=Lfunc(t[-1])
        r=rng.poisson(LT,(batch_size,)).astype(np.int32)
        njumps=int(np.sum(r))
        u=rng.uniform(0,1,(njumps,1)).astype(t.dtype)
        a=np.zeros_like(u,dtype=t.dtype)
        b=t[-1]*np.ones_like(u,dtype=t.dtype)
        for _ in range(0,10):
            tmp=Lfunc((a+b)/2)/LT
            less=tmp<=u
            greater=np.logical_not(less)
            a[less]=(a[less]+b[less])/2
            b[greater]=(a[greater]+b[greater])/2
        Ftinv=(a+b)/2

        lastJumps=0
        tI=np.argmax(Ftinv<t.T,axis=1)
        for wi in range(0,batch_size):
            currJumps=tI[lastJumps:lastJumps+r[wi]]
            lastJumps=lastJumps+r[wi]
            for ji in range(0,currJumps.size):
                ti=currJumps[ji]
                Nt[ti,wi]+=1
        return np.cumsum(Nt,0)
    
    @tf.function
    def populationSizeTF(n0,m,t,l,batch_size):
        
        Mt=tf.zeros((tf.size(t)*batch_size,),dtype=t.dtype)
        Nt=tf.ones((tf.size(t)*batch_size,),dtype=t.dtype)
        LT=l[-1]
        r=tf.random.poisson((batch_size,),LT,dtype=tf.int32)
        njumps=tf.cast(tf.reduce_sum(r),dtype=tf.int32)
        u=tf.random.uniform((njumps,),0,1,dtype=t.dtype)
        a=tf.zeros_like(u,dtype=t.dtype)
        b=t[-1]*tf.ones_like(u,dtype=t.dtype)
        for _ in range(0,10):
            tmp=tfp.math.interp_regular_1d_grid((a+b)/2,t[0,0],t[-1,0],l)/LT
            less=tf.cast(tmp<=u,t.dtype)
            greater=1-less
            a=(a+b)/2 *less + a * (1-less)
            b=(a+b)/2 *greater + b * (1-greater)
        Ftinv=tf.reshape((a+b)/2,(-1,1))
        tI=tf.cast(tf.argmax(Ftinv<tf.transpose(t),axis=1),dtype=tf.int32)
        ind=(tf.size(t)*tf.reshape(tf.range(0,batch_size,dtype=tf.int32),(-1,1)))+tf.RaggedTensor.from_row_lengths(tI,r)
        
        ind=tf.reshape(tf.concat(ind.to_tensor(),0),(-1,1))
        update=tf.random.uniform((tf.size(ind),),0.995,1.0,dtype=t.dtype)

        Mt=tf.tensor_scatter_nd_add(Mt,ind,tf.ones(shape=(tf.size(ind)),dtype=t.dtype)) #add a lot of ones to t=0, <- assume no jumps at t=0 and remove later
        Nt=tf.tensor_scatter_nd_update(Nt,ind,update) #makes a mistake, no double jumps, but this is fine, these are rare events and 0.995^2=0.990 is close, otherwise high computational cost

        Mt=tf.transpose(tf.reshape(Mt,(batch_size,tf.size(t)))) #row major -> also better memory layout for cumsum in axis 0
        Nt=tf.transpose(tf.reshape(Nt,(batch_size,tf.size(t)))) #row major -> also better memory layout for cumprod in axis 0

        Mt=tf.concat([tf.zeros((1,batch_size),dtype=t.dtype),tf.cumsum(Mt[1:,:],axis=0)],axis=0) #jump process
        Nt=n0*tf.math.exp(-m*t)*tf.concat([tf.ones((1,batch_size),dtype=t.dtype),tf.math.cumprod(Nt[1:,:],0)],axis=0)#population process
        return tf.stack([Nt,Mt],axis=2)
    
    def populationSizeNP(n0,m,t,l,batch_size,rng):
        Lfunc= lambda x: np.interp(x,t.flatten(),l)

        t=t.reshape((-1,1))
        Mt=np.zeros((t.size,batch_size),dtype=t.dtype)
        Nt=np.ones((t.size,batch_size),dtype=t.dtype)
        LT=Lfunc(t[-1])
        r=rng.poisson(LT,(batch_size,)).astype(np.int32)
        njumps=int(np.sum(r))
        u=rng.uniform(0,1,(njumps,1)).astype(t.dtype)
        a=np.zeros_like(u,dtype=t.dtype)
        b=t[-1]*np.ones_like(u,dtype=t.dtype)
        for _ in range(0,10):
            tmp=Lfunc((a+b)/2)/LT
            less=tmp<=u
            greater=np.logical_not(less)
            a[less]=(a[less]+b[less])/2
            b[greater]=(a[greater]+b[greater])/2
        Ftinv=(a+b)/2

        lastJumps=0
        tI=np.argmax(Ftinv<t.T,axis=1)
        k=0
        X=rng.uniform(0.995,1.0,(njumps,)).astype(t.dtype)
        for wi in range(0,batch_size):
            currJumps=tI[lastJumps:lastJumps+r[wi]]
            lastJumps=lastJumps+r[wi]
            for ji in range(0,currJumps.size):
                ti=currJumps[ji]
                Nt[ti,wi]*=X[k]
                Mt[ti,wi]+=1
                k+=1
        return np.stack([n0*np.exp(-m*t)*np.cumprod(Nt,0),np.cumsum(Mt,0)],axis=2)

    def __init__(self, 
                 H0:float,
                 m:Union[np.ndarray,tf.Tensor],
                 t:Union[np.ndarray,tf.Tensor],
                 tData:Union[np.ndarray,tf.Tensor],
                 dm:Union[np.ndarray,tf.Tensor],
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None,
                 treatmentCosts:float=0.01):
        super().__init__(t,True,1,rng,treatmentCosts)
        
        dt=np.array(t[-1]/(t.shape[0]-1))
        if type(t)==np.ndarray:
            self.m=m
            self.n0=H0
            self.l=(np.cumsum(np.interp(t,tData,dm),axis=0)*dt).flatten()
            # self._inhomPoisson=Poisson.inhomPoissonNP
            self._populationSize=Poisson.populationSizeNP
        else:
            self.m=tf.constant(m,dtype=t.dtype)
            self.n0=H0
            lfunc= (np.cumsum(np.interp(np.array(t),np.array(tData),np.array(dm)),axis=0)*dt).flatten()
            self.l=tf.constant(lfunc,dtype=t.dtype)
            # self._inhomPoisson=Poisson.inhomPoissonTF
            self._populationSize=Poisson.populationSizeTF

    def sample(self, batch_size: int):
        if type(self.t)==np.ndarray:
            return self._populationSize(self.n0,self.m,self.t,self.l,batch_size,self.rng)
        else:
            return self._populationSize(self.n0,self.m,self.t,self.l,batch_size)
        
    def populationSize(self,
                       M:Union[np.ndarray,tf.Tensor] # host + jump process
                       ):
        return M[:,:,0]
    
    def treatmentCost(self,
                       M:Union[np.ndarray,tf.Tensor] # host + jump process
                       ):
        return self.treatmentCosts*M[:,:,1]


class DetermPoisson(Poisson):
    def __init__(self,
                 H0:float,
                 m:Union[np.ndarray,tf.Tensor],
                 t:Union[np.ndarray,tf.Tensor],
                 tData:Union[np.ndarray,tf.Tensor],
                 dm:Union[np.ndarray,tf.Tensor],
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None,
                 treatmentCosts:float=0.01):
        super().__init__(H0,m,t,tData,dm,rng,treatmentCosts)
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

    """Test Host Parasite
    """

    # T=3.0
    # N=int(T*2*12)*10
    # batch_size=2**12
    # dtype=np.float32
    # t=np.linspace(0,T,N,endpoint=True,dtype=dtype).reshape((-1,1))
    # t=tf.constant(t)
    # params=[0.05,0.1,8.71,0.05]
    # beta=[0.0835,0.0244]
    # H0=10000.0
    # P0=1

    # mort = DetermHostParasite(t,params,beta,H0,P0)
    # tic=time.time()
    # HPM=mort.sample(batch_size)
    # ctime=time.time()-tic
    # print(f'Elapsed time {ctime} s')
    # tic=time.time()
    # HPM=mort.sample(batch_size)
    # ctime=time.time()-tic
    # print(f'Elapsed time {ctime} s')
    # tic=time.time()
    # HPM=mort.sample(batch_size)
    # ctime=time.time()-tic
    # print(f'Elapsed time {ctime} s')

    """Test Poisson
    """
    T=3.0
    N=int(T*2*12)*10
    batch_size=2**12
    dtype=np.float32
    t=np.linspace(0,T,N,endpoint=True,dtype=dtype).reshape((-1,1))
    t=tf.constant(t)
    params=[0.05,0.1,8.71,0.05]
    beta=[0.0835,0.0244]
    H0=10000.0
    
    tData=np.array([0.018868,0.037736,0.056604,0.075472,0.09434,0.11321,0.13208,0.15094,0.16981,0.18868,0.20755,0.22642,0.24528,0.26415,0.28302,0.30189,0.32075,0.33962,0.35849,0.37736,0.39623,0.41509,0.43396,0.45283,0.4717,0.49057,0.50943,0.5283,0.54717,0.56604,0.58491,0.60377,0.62264,0.64151,0.66038,0.67925,0.69811,0.71698,0.73585,0.75472,0.77358,0.79245,0.81132,0.83019,0.84906,0.86792,0.88679,0.90566,0.92453,0.9434,0.96226,0.98113,1,1.0189,1.0377,1.0755,1.0943,1.1132,1.1321,1.1509,1.1698,1.1887,1.2075,1.2264,1.2453,1.2642,1.283,1.3019,1.3208,1.3396,1.3585,1.3774,1.3962,1.4151,1.434,1.4528,1.4717,1.4906,1.5094,1.5283,1.5472,1.566,1.5849,1.6038
],dtype=np.float32).reshape((-1,))
    dm=np.array([0.45003,0,0.45003,0,0,0.90005,0,0,1.3501,0.90005,1.8001,1.3501,1.8001,2.2501,3.1502,1.8001,1.8001,2.7002,2.2501,2.2501,2.7002,1.8001,3.1502,3.1502,3.1502,5.8503,5.8503,6.7504,4.9503,6.3004,4.0502,7.2004,7.6504,8.5505,9.0005,9.9006,11.701,12.601,11.701,13.951,14.401,13.051,12.601,14.401,12.601,11.251,14.401,13.951,16.201,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501
],dtype=np.float32).reshape((-1,))
    
    
    # dt=T/(N-1)
    # lfunc= (np.cumsum(np.interp(t,tData[1:],dm),axis=0)*dt).flatten()

    # Lfunc= lambda x: np.interp(x,t.flatten(),lfunc)

    # rngNP=np.random.default_rng(0)
    # rngTF=tf.random.Generator.from_seed(0)

    # tic=time.time()
    # Nt=Poisson.inhomPoissonNP(t,lfunc,batch_size,rngNP)
    # ctime=time.time()-tic
    # print(f'Elapsed time {ctime} s')
    # print(np.mean(Nt[-1,:]))
    # print(np.mean((np.mean(Nt,axis=1)-lfunc)**2))

    # tic=time.time()
    # Nt=Poisson.inhomPoissonTF(tf.constant(t),tf.constant(lfunc,dtype=t.dtype),batch_size)
    # ctime=time.time()-tic
    # print(f'Elapsed time {ctime} s')
    # print(np.mean(Nt[-1,:]))

    # tic=time.time()
    # Nt=Poisson.inhomPoissonTF(tf.constant(t),tf.constant(lfunc,dtype=t.dtype),batch_size)
    # ctime=time.time()-tic
    # print(f'Elapsed time {ctime} s')
    # print(np.mean(Nt[-1,:]))

    # print(np.mean((np.mean(Nt,axis=1)-lfunc)**2))

    mort = Poisson(H0,params[0],t,tData[1:],dm)
    tic=time.time()
    HM=mort.sample(batch_size)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s')
    print(HM.shape)
    print(np.mean(HM[-1,:,0]))
    print(np.mean(HM[-1,:,1]))
    tic=time.time()
    HM=mort.sample(batch_size)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s')
    print(np.mean(HM[-1,:,0]))
    print(np.mean(HM[-1,:,1]))



# """ Old Code: Thinning method too slow """
# class Poisson(Mortality):

#     @nb.njit() #slight improvement
#     def _thinningNbSingle(t,lfunc,lambdaM,Nt):
#         T=t[-1]
#         X=0.0
#         ti=0.0
#         J=0
#         # s=[]
#         # Nt=np.zeros((t.size,1))
#         while ti<T:
#             # u=np.random.rand(1)[0]#not bottleneck
#             X=-1/lambdaM[J]*np.log(np.random.rand(1))[0]
#             if ti+X>pwT[J+1]:
#                 if J>=pwT.size-2:
#                     break
#                 else:
#                     X=(X-pwT[J+1]+ti)*lambdaM[J]/lambdaM[J+1]
#                     ti=pwT[J+1]
#                     J=J+1
#             else:
#                 ti=ti+X
#                 U=np.random.rand(1)[0]
#                 if U <=lfunc(np.array(ti))/lambdaM[J]:
#                     currJ=np.where(t>=ti)[0][0]
#                     Nt[currJ]=Nt[currJ]+1
    
#     # @nb.njit() #makes it actually slower
#     def _thinningBatch(inner,t,lfunc,lambdaM,batch_size,Nt):
#         for wi in range(0,batch_size):
#             inner(t,lfunc,lambdaM,Nt[:,wi]) #inplace Nt[:,wi]

#     def thinningNb(tData,dm,pwT,t,batch_size,rng):
#         tData=tData.astype(np.float64)
#         dm=dm.astype(np.float64)
#         t=t.astype(np.float64)

#         @nb.njit()
#         def lfunc(x): #not the bottleneck
#             # sz=x.shape
#             # t=tData[1:]
#             # y=dm
#             # vec=x.flatten()
#             # for i in range(0,vec.size):
#             #     x=vec[i]
#             #     if x<=t[0]:
#             #         vec[i]=y[0]
#             #     elif x>=t[-1]:
#             #         vec[i]=y[-1]  
#             #     else:
#             #         j=np.where(x>=t)[0][0]
#             #         vec[i]=y[j-1]+(y[j]-y[j-1])/(t[j]-t[j-1])*(vec[i]-t[j-1])
#             # return vec.reshape(sz)
#             return np.interp(x,tData[1:],dm)
        
#         T=t[-1]
#         pwT=pwT.reshape((-1,1))
#         t0=pwT[:-1]
#         t1=pwT[1:]
#         mesh=np.linspace(0,1,100,endpoint=True,dtype=np.float64).reshape((1,-1))
#         tgrid=t0+(t1-t0)*mesh
#         sz=tgrid.shape
#         lambdaM=np.max(lfunc(tgrid.flatten()).reshape(sz),axis=1).astype(np.float64)
#         Nt=np.zeros((t.size,batch_size))
#         Poisson._thinningBatch(Poisson._thinningNbSingle,t,lfunc,lambdaM,batch_size,Nt)#manipulates Nt inplace
#         Nt=np.cumsum(Nt,axis=0)
#         return Nt
    
#     def thinningNP(lfunc,pwT,t,batch_size,rng):
#         T=t[-1]
#         pwT=pwT.reshape((-1,1))
#         t0=pwT[:-1]
#         t1=pwT[1:]
#         mesh=np.linspace(0,1,100,endpoint=True,dtype=t.dtype).reshape((1,-1))
#         tgrid=t0+(t1-t0)*mesh
#         sz=tgrid.shape
#         lambdaM=np.max(lfunc(tgrid.flatten()).reshape(sz),axis=1)

#         Nt=np.zeros((t.size,batch_size),dtype=t.dtype)
#         for wi in range(0,batch_size):
#             ti=0
#             J=0

#             while ti<T:
#                 X=-1/lambdaM[J]*np.log(rng.uniform(low=0.0,high=1.0,size=(1,1)).astype(t.dtype))
#                 if ti+X>pwT[J+1]:
#                     if J>=pwT.size-2:
#                         break
#                     X=(X-pwT[J+1]+ti)*lambdaM[J]/lambdaM[J+1]
#                     ti=pwT[J+1]
#                     J=J+1
#                 else:
#                     ti=ti+X
#                     U=rng.uniform(low=0.0,high=1.0,size=(1,1)).astype(t.dtype)
#                     if U <=lfunc(ti)/lambdaM[J]:
#                         currJ=np.where(t>=ti)[0][0]
#                         Nt[currJ,wi]=Nt[currJ,wi]+1

#         Nt=np.cumsum(Nt,axis=0)
#         return Nt
    
#     # @tf.function
#     def thinningTF(lfunc,pwT,t,batch_size,rng):

#         T=t[-1]
#         pwT=tf.reshape(pwT,(-1,1))
#         t0=pwT[:-1]
#         t1=pwT[1:]
#         mesh=tf.reshape(tf.linspace(tf.constant(0,dtype=tf.float32),tf.constant(1,dtype=tf.float32),100),(1,-1)) 
#         tgrid=t0+(t1-t0)*mesh
#         sz=tgrid.shape
#         tgrid=tf.reshape(tgrid,(-1,1))
#         lambdaM=tf.reshape(tf.math.reduce_max(tf.reshape(lfunc(tgrid),sz),axis=1),(-1,1)) #this block takes 0.008 sec in Eager mode

#         # Nt=tf.zeros((tf.size(t),batch_size),dtype=t.dtype)

#         @tf.function
#         def body(ti,J,X,s):
#             X=-1/lambdaM[J[0]]*tf.math.log(rng.uniform((1,),0.0,1.0,dtype=t.dtype))
#             if ti+X>pwT[J[0]+1]:
#                 if J[0]>=tf.size(pwT)-2:
#                     ti=T+1
#                 else:
#                     X=(X-pwT[J[0]+1]+ti)*lambdaM[J[0]]/lambdaM[J[0]+1]
#                     ti=pwT[J[0]+1]
#                     J=J+1
#             else:
#                 ti=ti+X
#                 U=tf.math.log(rng.uniform((1,),0.0,1.0,dtype=t.dtype))
#                 if U <=lfunc(ti)/lambdaM[J[0]]:
#                     tmp=tf.cast(tf.where(t>=ti)[0][0],tf.int32)
#                     s=tf.concat([s,tf.reshape(tmp,(1,1))],axis=0)

#             return ti,J,X,s

#         @tf.function
#         def condition(ti,J,X,s):
#             return tf.less(ti,T)
        
#         @tf.function
#         def sample(i):
#             ti=tf.constant([0.0],dtype=t.dtype)
#             X=tf.constant([0.0],dtype=t.dtype)
#             J=tf.constant([0],dtype=tf.int32)
#             s=tf.constant([0],shape=(1,1),dtype=tf.int32)
#             out = tf.while_loop(condition, body, [ti,J,X,s],shape_invariants=[ti.get_shape(),J.get_shape(),X.get_shape(),tf.TensorShape([None,1])])
#             stf=tf.reshape(out[3][1:],(-1,))
#             Nti=tf.zeros((tf.size(t),1),dtype=t.dtype)
#             onesInt=tf.zeros_like(stf,dtype=tf.int32)
#             onesFloat=tf.ones_like(stf,dtype=tf.float32)
#             return tf.tensor_scatter_nd_add(Nti,tf.stack([stf,onesInt],axis=1),onesFloat)
        
#         # return tf.map_fn(sample,tf.range(0,batch_size),parallel_iterations=10) # does not work for some reason
#         nt=[]
#         for wi in range(0,batch_size):
#             nt.append(sample(tf.constant(wi)))
#         return tf.cumsum(tf.concat(nt,axis=1),axis=0)

#         # for wi in range(0,batch_size):
#         #     # ti=tf.constant([0.0],dtype=t.dtype)
#         #     # X=tf.constant([0.0],dtype=t.dtype)
#         #     # J=tf.constant([0],dtype=tf.int32)
#         #     # s=tf.constant([0],shape=(1,1),dtype=tf.int32)
#         #     # out = tf.while_loop(condition, body, [ti,J,X,s],shape_invariants=[ti.get_shape(),J.get_shape(),X.get_shape(),tf.TensorShape([None,1])])
#         #     # stf=tf.reshape(out[3][1:],(-1,))
#         #     # stf=tf.cast(tf.concat(s,axis=0),dtype=tf.int32)
#         #     stf=sample()
#         #     onesInt=tf.ones_like(stf,dtype=tf.int32)
#         #     onesFloat=tf.ones_like(stf,dtype=tf.float32)
#         #     Nt=tf.tensor_scatter_nd_add(Nt,tf.stack([stf,tf.cast(wi,tf.int32)*onesInt],axis=1),onesFloat) #not the bottleneck
#         # return tf.cumsum(Nt,axis=0)