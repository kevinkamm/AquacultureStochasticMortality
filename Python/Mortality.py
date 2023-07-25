import numpy as np
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
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None):
        self.t=t
        self.dtype=t.dtype
        self.isStoch=isStoch
        self.d=d
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
        super().__init__(t,False,0,rng)
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

        super().__init__(t,True,2,rng)
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
        
class Poisson(Mortality):
    def thinningNP(lfunc,pwT,t,batch_size,rng):
        T=t[-1]
        pwT=pwT.reshape((-1,1))
        t0=pwT[:-1]
        t1=pwT[1:]
        mesh=np.linspace(0,1,100,endpoint=True,dtype=t.dtype).reshape((1,-1))
        tgrid=t0+(t1-t0)*mesh
        sz=tgrid.shape
        lambdaM=np.max(lfunc(tgrid.flatten()).reshape(sz),axis=1)

        Nt=np.zeros((t.size,batch_size),dtype=t.dtype)
        for wi in range(0,batch_size):
            ti=0
            J=0
            s=[]
            while ti<T:
                X=-1/lambdaM[J]*np.log(rng.uniform(low=0.0,high=1.0,size=(1,1)).astype(t.dtype))
                if ti+X>pwT[J+1]:
                    if J>=pwT.size-2:
                        break
                    X=(X-pwT[J+1]+ti)*lambdaM[J]/lambdaM[J+1]
                    ti=pwT[J+1]
                    J=J+1
                else:
                    ti=ti+X
                    U=rng.uniform(low=0.0,high=1.0,size=(1,1)).astype(t.dtype)
                    if U <=lfunc(ti)/lambdaM[J]:
                        s.append(ti)
            for si in s:
                currJ=np.where(t>=si)[0][0]
                Nt[currJ,wi]=Nt[currJ,wi]+1
        Nt=np.cumsum(Nt,axis=0)
        return Nt
    
    @tf.function
    def thinningTF(lfunc,pwT,t,batch_size,rng):

        T=t[-1]
        pwT=tf.reshape(pwT,(-1,1))
        t0=pwT[:-1]
        t1=pwT[1:]
        mesh=tf.reshape(tf.linspace(tf.constant(0,dtype=tf.float32),tf.constant(1,dtype=tf.float32),10),(1,-1)) 
        tgrid=t0+(t1-t0)*mesh
        sz=tgrid.shape
        tgrid=tf.reshape(tgrid,(-1,1))
        lambdaM=tf.reshape(tf.math.reduce_max(tf.reshape(lfunc(tgrid),sz),axis=1),(-1,1)) #this block takes 0.008 sec in Eager mode

        Nt=tf.zeros((tf.size(t),batch_size),dtype=t.dtype)

        # def cond(ti,T):
        #     return tf.logical_and(tf.less(ti,T),)
        # def body(ti,J,s):
        #     X=-1/lambdaM[J]*tf.math.log(rng.uniform((1,1),0.0,1.0,dtype=t.dtype)) # 0.0009s
        #     # print(f'Time {time.time()-tic}')
        #     if ti+X>pwT[J+1]:
        #         if J>=tf.size(pwT)-2:
        #             ti=T+1
        #             # break
        #         # tic=time.time()
        #         else:
        #             X=(X-pwT[J+1]+ti)*lambdaM[J]/lambdaM[J+1]
        #             ti=pwT[J+1]
        #             J=J+1
        #         # print(f'Time {time.time()-tic}') #0.0009s
        #     else:
        #         ti=ti+X
        #         U=rng.uniform((1,1),0.0,1.0,dtype=t.dtype) # 0.0009s
        #         if U <=lfunc(ti)/lambdaM[J]:
        #             # s.append(ti)
        #             s.append(tf.where(t>=ti)[0][0])


        for wi in range(0,batch_size):
            ti=tf.constant(0.0,dtype=t.dtype,shape=(1,1))
            X=0
            J=0
            s=tf.reshape(tf.constant([0],dtype=tf.int64),(1,1))
            while ti<T: #one iteration 0.005 sec in Eager mode
                # tic=time.time()
                X=-1/lambdaM[J]*tf.math.log(rng.uniform((1,1),0.0,1.0,dtype=t.dtype)) # 0.0009s
                # print(f'Time {time.time()-tic}')
                if ti+X>pwT[J+1]:
                    if J>=tf.size(pwT)-2:
                        ti=T+1.0
                    # tic=time.time()
                    else: 
                        X=(X-pwT[J+1]+ti)*lambdaM[J]/lambdaM[J+1]
                        ti=pwT[J+1]
                        J=J+1
                    # print(f'Time {time.time()-tic}') #0.0009s
                else:
                    ti=ti+X
                    U=rng.uniform((1,1),0.0,1.0,dtype=t.dtype) # 0.0009s
                    if U <=lfunc(ti)/lambdaM[J]:
                        # s.append(ti)
                        # s.append(tf.where(t>=ti)[0][0])
                        tmp=tf.where(t>=ti)[0][0]
                        s=tf.concat([s,tf.reshape(tmp,(1,1))],axis=0)
                ti=tf.reshape(ti,(1,1))

                
                # print(f'Time {time.time()-tic}')
            s2=s
            # stf=tf.cast(tf.concat(s,axis=0),dtype=tf.int32)
            # onesInt=tf.ones_like(s,dtype=tf.int32)
            # onesFloat=tf.ones_like(s,dtype=tf.float32)
            # Nt=tf.tensor_scatter_nd_update(Nt,tf.stack([stf,tf.cast(wi,tf.int32)*onesInt],axis=1),onesFloat) #not the bottleneck
        Nt=tf.cumsum(Nt,axis=0)
        return Nt

    def __init__(self, 
                 t:Union[np.ndarray,tf.Tensor],
                 params:Union[np.ndarray,tf.Tensor],
                 l:Union[np.ndarray,tf.Tensor],
                 pwT:Union[np.ndarray,tf.Tensor],
                 rng:Optional[Union[np.random.Generator,tf.random.Generator]]=None):
        super().__init__(t, rng,True,1)
        if type(t)==np.ndarray:
            self.params=params
            self.l=l
            self.pwT=pwT
        else:
            self.params=tf.constant(params)
            self.l=tf.constant(l)
            self.pwT=tf.constant(pwT)


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
    batch_size=10
    dtype=np.float32
    t=np.linspace(0,T,N,endpoint=True,dtype=dtype).reshape((-1,1))
    # t=tf.constant(t)
    params=[0.05,0.1,8.71,0.05]
    beta=[0.0835,0.0244]
    H0=10000.0
    
    tData=np.array([0.018868,0.037736,0.056604,0.075472,0.09434,0.11321,0.13208,0.15094,0.16981,0.18868,0.20755,0.22642,0.24528,0.26415,0.28302,0.30189,0.32075,0.33962,0.35849,0.37736,0.39623,0.41509,0.43396,0.45283,0.4717,0.49057,0.50943,0.5283,0.54717,0.56604,0.58491,0.60377,0.62264,0.64151,0.66038,0.67925,0.69811,0.71698,0.73585,0.75472,0.77358,0.79245,0.81132,0.83019,0.84906,0.86792,0.88679,0.90566,0.92453,0.9434,0.96226,0.98113,1,1.0189,1.0377,1.0755,1.0943,1.1132,1.1321,1.1509,1.1698,1.1887,1.2075,1.2264,1.2453,1.2642,1.283,1.3019,1.3208,1.3396,1.3585,1.3774,1.3962,1.4151,1.434,1.4528,1.4717,1.4906,1.5094,1.5283,1.5472,1.566,1.5849,1.6038
],dtype=np.float32).reshape((-1,))
    dm=np.array([0.45003,0,0.45003,0,0,0.90005,0,0,1.3501,0.90005,1.8001,1.3501,1.8001,2.2501,3.1502,1.8001,1.8001,2.7002,2.2501,2.2501,2.7002,1.8001,3.1502,3.1502,3.1502,5.8503,5.8503,6.7504,4.9503,6.3004,4.0502,7.2004,7.6504,8.5505,9.0005,9.9006,11.701,12.601,11.701,13.951,14.401,13.051,12.601,14.401,12.601,11.251,14.401,13.951,16.201,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501
],dtype=np.float32).reshape((-1,))
    lfunc= lambda x: np.interp(x,tData[1:],dm)
    rng=np.random.default_rng(0)
    pwT=np.array([0.020862,0.20862,0.49235,0.60501,0.70097,1.0014,1.0974,1.1892,1.2851,3],dtype=t.dtype) #slightly faster in np
    # pwT=np.array([t[0],t[-1]],dtype=t.dtype)

    tic=time.time()
    Nt=Poisson.thinningNP(lfunc,pwT,t,batch_size,rng)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s')
    print(np.mean(Nt[-1,:]))

    t=tf.constant(t)
    tData=tf.constant(tData)
    dm=tf.constant(dm)
    lfunc= lambda x: tfp.math.interp_regular_1d_grid(x,tData[1],tData[-1],dm)
    pwT=tf.constant(pwT)
    rng=tf.random.Generator.from_seed(0)


    tic=time.time()
    Nt=Poisson.thinningTF(lfunc,pwT,t,batch_size,rng)
    ctime=time.time()-tic
    print(f'Elapsed time {ctime} s')
    print(tf.reduce_mean(Nt[-1,:]))
