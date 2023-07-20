import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Layer,BatchNormalization, Dense, Reshape, Activation, Flatten
# from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback
from tqdm import tqdm

from joblib import Parallel,delayed
from psutil import cpu_count

from typing import Optional, Callable, Generator, List
from numpy.typing import DTypeLike

num_cores=cpu_count(logical=False)

class Basis:
    def __init__(self,
                 dtype:DTypeLike=np.float32) -> None:
        self.dtype=dtype
    
    def fit_eval(self):
        pass

class Polynomial(Basis):
    def __init__(self,
                 deg:int =2,
                 dtype:DTypeLike=np.float32):
        super().__init__(dtype=dtype)
        self.deg=deg
        self.poly = PolynomialFeatures(deg)
        self.poly_reg_model = LinearRegression()

    def fit_eval(self,X,Y):
        poly_features = self.poly.fit_transform(X)
        self.poly_reg_model.fit(poly_features, Y)
        self.poly_reg_model.predict(poly_features)
        return self.poly_reg_model.predict(poly_features)
    
class DeepOS(Model):
    def __init__(self, 
                d:int = 1, # dimension of process
                N:int = 2, # number of time steps
                latent_dim: List[int]=[51], 
                outputDims: int=1, 
                name="deeposnet", 
                **kwargs
                ) \
            -> None:
        super().__init__(name=name,**kwargs)
        self.latent_dim = latent_dim
        self.outputDims = outputDims
        self.d = d+1 # since price adds a dimension
        self.N = N-1 # stopping at t=N excluded
        self.D = []
        self.Rbefore=[Flatten(name='reshape01')]
        # self.Rbefore=[Reshape((self.N*self.d,), name='reshape01')]
        self.BN=[BatchNormalization(epsilon=1e-6,axis=-1,momentum=0.9)]
        self.Rafter=[Reshape((self.N , self.d), name='reshape02')]
        self.A=[]
        for i, hD in enumerate(self.latent_dim):
            self.D.append(Dense(units=hD,activation=None,name=f'dense{i + 1}'))
            self.Rbefore.append(Flatten(name=f'reshape{i + 1}1'))
            # self.Rbefore.append(Reshape((self.N*hD,),name=f'reshape{i + 1}1'))
            self.BN.append(BatchNormalization(epsilon=1e-6, axis=-1, momentum=0.9))
            self.Rafter.append(Reshape((self.N, hD), name=f'reshape{i + 1}2'))
            self.A.append(Activation('relu'))
        i += 1
        self.D.append(Dense(units=self.outputDims,activation=None,name='denseout'))
        self.Rbefore.append(Flatten(name=f'reshape{i + 1}1'))
        # self.Rbefore.append(Reshape((self.N*self.outputDims,),name=f'reshape{i + 1}1'))
        self.BN.append(BatchNormalization(epsilon=1e-6,axis=-1,momentum=0.9))
        self.Rafter.append(Reshape((self.N, self.outputDims), name=f'reshape{i + 1}2'))
        self.A.append(Activation('sigmoid'))

    def call(self,inputs,training=False):
        x=self.Rbefore[0](inputs)
        x=self.BN[0](x,training)
        x=self.Rafter[0](x)
        for i in range(1,len(self.BN)):
            x=self.D[i-1](x)
            x=self.Rbefore[i](x)
            x=self.BN[i](x,training)
            x=self.Rafter[i](x)
            x=self.A[i-1](x)
        return x

    # @tf.function
    # def train_step(self,data,opt):
    def train_step(self,data):
        # data shape = (Batch,Processes+Price,Time)
        p=data[:,-1,:]
        with tf.GradientTape() as tape:
            nets = self(tf.transpose(data[:,:,:-1],(0,2,1)),training=True)
            nets = tf.transpose(nets,(0,2,1))
            u_list = [nets[:, :, 0]]
            u_sum = u_list[-1]
            for k in range(1, self.N):
                u_list.append(nets[:, :, k] * (1. - u_sum))
                u_sum += u_list[-1]

            u_list.append(1. - u_sum)
            u_stack = tf.concat(u_list, axis=1)
            # p = tf.squeeze(p, axis=1)
            loss = tf.reduce_mean(tf.reduce_sum(-u_stack * p, axis=1))
            # loss = self.compiled_loss(tf.reduce_sum(-u_stack * p, axis=1),0,regularization_losses=self.losses)

        var_list = self.trainable_variables
        gradients = tape.gradient(loss,var_list)
        self.optimizer.apply_gradients(zip(gradients,var_list))
        # opt.apply_gradients(zip(gradients,var_list))

        idx = tf.argmax(tf.cast(tf.cumsum(u_stack, axis=1) + u_stack >= 1,
                                dtype=tf.uint8),
                        axis=1,
                        output_type=tf.int32)
        batch_size=data.shape[0]
        stopped_payoffs = tf.reduce_mean(tf.gather_nd(p, tf.stack([tf.range(0, batch_size, dtype=tf.int32), idx],
                                                                  axis=1)))
        return {'loss':loss, 'payoff':stopped_payoffs}

    def predict_step(self,data):
        # data shape = (Batch,Processes+Price,Time), Price Last
        p=data[:,-1,:] # shape= (Batch,Time)
        nets = self(tf.transpose(data[:,:,:-1],(0,2,1)),training=False)
        nets = tf.transpose(nets,(0,2,1))
        u_list = [nets[:, :, 0]]
        u_sum = u_list[-1]
        for k in range(1, self.N):
            u_list.append(nets[:,:,  k] * (1. - u_sum))
            u_sum += u_list[-1]

        u_list.append(1. - u_sum)
        u_stack = tf.concat(u_list, axis=1)
        # p = tf.squeeze(p, axis=1)

        idx = tf.argmax(tf.cast(tf.cumsum(u_stack, axis=1) + u_stack >= 1,
                                dtype=tf.uint8),
                        axis=1,
                        output_type=tf.int32)
        batch_size=data.shape[0]
        stopped_payoffs = tf.gather_nd(p, tf.stack([tf.range(0, batch_size, dtype=tf.int32), idx],
                                                                  axis=1))

        tau = tf.reshape(tf.range(0,self.N+1) ,(1,-1))*tf.ones((batch_size,1),dtype=tf.int32)
        stopped_index = tf.gather_nd(tau, tf.stack([tf.range(0, batch_size, dtype=tf.int32), idx],
                                        axis=1))
        return stopped_index, stopped_payoffs

class OptimalStopping:
    def __init__(self,
                 r:float,
                 t:np.ndarray,
                 gen:Callable[[int], np.ndarray]) -> None:
        self.r=r
        self.t=t
        self.gen=gen

    def train(self,batch_size:int,batches:int):
        pass

    def evaluate(self,batch_size:int,batches:int):
        pass

    # def solve(self):
    #     if self.batches >1:
    #         out = Parallel(n_jobs=num_cores)(delayed(self.solveBatch)() for i in range(self.batches))
    #     else:
    #         out = self.solveBatch()
    #     return out

    # def solveBatch(self): 
    #     pass

class DeepOptS(OptimalStopping):
    def __init__(self, 
                 r: float, 
                 t: np.ndarray, 
                 gen: Callable[[int], np.ndarray],
                 d:int ) -> None:
        super().__init__(r, t, gen)
        self.N=t.shape[0]
        self.d=d
        self.lr_values = [0.05, 0.005, 0.0005]
        self.neurons = [d + 50, d + 50]
        self.train_steps = 3000 + d
        self.lr_boundaries = [int(500 + d / 5), int(1500 + 3 * d / 5)]
        self.learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.lr_boundaries, self.lr_values)
        self.opt = Adam(self.learning_rate_fn,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
        self.model = DeepOS(d=d,N=self.N,latent_dim=self.neurons)
        self.model.compile(optimizer=self.opt, jit_compile=False, run_eagerly=True,steps_per_execution=1)
        if type(t)==np.ndarray:
            def datagen():
                while True:
                    X,V,_,_ = self.gen(batch_size)
                    yield np.concatenate([X,np.expand_dims(V,axis=2)],axis=2).transpose((1,2,0))
            self.dataset = tf.data.Dataset.from_generator(datagen,output_signature=(tf.TensorSpec(shape=(batch_size,self.d+1,self.N))))
        else:
            def datagen():
                while True:
                    X,V,_,_ = self.gen(batch_size)
                    yield tf.transpose(tf.concat([X,tf.expand_dims(V,axis=2)],axis=2),(1,2,0))
            self.dataset = tf.data.Dataset.from_generator(datagen,output_signature=(tf.TensorSpec(shape=(batch_size,self.d+1,self.N))))
            self.t=t.numpy()
        
    
    def train(self,batch_size:int,batches:int):
        self.model.fit(self.dataset,epochs=self.train_steps,steps_per_epoch=1,verbose=0, callbacks=[TqdmCallback(verbose=0)],
                       workers=1,
                       use_multiprocessing=False) #check if opt updates correctly

    def evaluate(self, batch_size:int, batches: int):
        stopped_index, stopped_payoffs = self.model.predict(self.dataset,steps=batches)
        tau=self.t[stopped_index]
        return tau,stopped_payoffs

class LSMC(OptimalStopping):
    def __init__(self,
                 r:float,
                 t:np.ndarray,
                 gen:Callable,
                 b:Basis) -> None:
        super().__init__(r,t,gen)
        self.b=b

        if type(t)==np.ndarray:
            self._gen=gen
        else:
            def newgen(x):
                X,V,VH,ft = self.gen(batch_size)
                return X.numpy(),V.numpy(),VH.numpy(),ft.numpy()
            self._gen=newgen
            self.t=self.t.numpy()

    def evaluateBatch(self,batch_size:int):
        X,V,VH,ft = self._gen(batch_size)
        N=X.shape[0]
        M=X.shape[1]

        Vtmp=VH[-1,:].copy()
        VC=np.empty_like(Vtmp)
        exercise=(N-1)*np.ones((M,1),dtype=np.int32)

        dt=self.t[-1]/(N-1)
        discount=np.exp(-self.r*dt)
        # Glassermann p. 461
        for ti in range(N-1,0,-1): #[N-1,...,1]
            if ft is not None:
                VC=discount * self.b.fit_eval(X[ti,:,:],Vtmp) -ft[ti]*dt
                Vtmp=discount*Vtmp -ft[ti]*dt # Longstaff-Schwartz
                #Vtmp=VC # Tsitsiklis and Van Roy
            else:
                VC= discount * self.b.fit_eval(X[ti,:,:],Vtmp) 
                Vtmp=discount*Vtmp # Longstaff-Schwartz
                #Vtmp=VC # Tsitsiklis and Van Roy
            ind = VC <= VH[ti,:]
            exercise[ind]=ti

            Vtmp[ind]=VH[ti,ind]

        Vtau=np.empty_like(Vtmp)
        tau=np.empty_like(Vtmp)
        for wi in range(0,M):
            ti=exercise[wi]
            tau[wi]=self.t[ti]
            Vtau[wi]=V[ti,wi]

        return tau,Vtau

    def evaluate(self, batch_size:int,batches: int):
        # tau,Vtau = zip(*Parallel(n_jobs=num_cores)(delayed(self.evaluateBatch)(batch_size) for _ in range(0,batches) ) ) #generator not thread safe
        # return tau,Vtau
        tau=[]
        Vtau=[]
        for _ in range(batches):
            t,V = self.evaluateBatch(batch_size)
            tau.append(t)
            Vtau.append(V)
        return np.concatenate(tau),np.concatenate(Vtau)

if __name__=="__main__":
    import time
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ["TF_ENABLE_ONEDNN_OPTS"]="1"
    # tf.config.set_visible_devices([], 'GPU')
    T=3.0
    N=int(T*2*12)
    r=0.0303
    d=0
    Nsim = 10
    dtype=np.float32
    seed=1

    t=np.linspace(0,T,N*Nsim,endpoint=True,dtype=dtype).reshape((-1,1))
    t=tf.constant(t)

    from Commodities import Schwartz2Factor
    'Salmon'
    # mu, sigma1, sigma2, kappa, alpha, lambda, rho, delta0, P0
    # salmonParam=[0.12, 0.23, 0.75, 2.6, 0.02, 0.01, 0.9, 0.57, 95] # down,down
    salmonParam=[0.12, 0.23, 0.75, 2.6, 0.02, 0.2, 0.9, 0.57, 95] # down,up
    # salmonParam=[0.12, 0.23, 0.75, 2.6, 0.02, 0.6, 0.9, 0.57, 95] # up,up

    'Soy'
    # mu, sigma1, sigma2, kappa, alpha, lambda, rho, delta0, P0
    # soyParam=[0.15, 0.5, 0.4, 1.2, 0.06, 0.14, 0.44, 0.0, 1500] # low vol
    soyParam=[0.15, 1, 0.4, 1.2, 0.06, 0.14, 0.44, 0.0, 1] # medium vol
    # soyParam=[0.15, 2, 0.4, 1.2, 0.06, 0.14, 0.44, 0.0, 1500] # high vol

    'Risk neutral dynamics'
    salmonParam[0]=r
    soyParam[0]=r

    "Fish feeding 25% of production cost, disease 30%, harvest 10%. Total production cost = 50% of price = labor, smolt, ..."
    salmonPrice=salmonParam[-1] #NOK/KG
    harvestingCosts=salmonPrice*0.5*0.1 # roughly 10%
    feedingCosts=salmonPrice*0.5*0.25
    initialSalmon=0.5*salmonPrice+feedingCosts+harvestingCosts #we add the costs to salmon price since they are respected in the model, other costs are fixed and thus removed
    salmonParam[-1]=initialSalmon
    print(f'Feeding costs {feedingCosts} and Harvesting costs {harvestingCosts}')
    # soyParam[-1]=feedingCosts # to save the right dataset, since initial price is not relevant for soy model


    soy=Schwartz2Factor(soyParam,t,dtype=dtype)
    salmon=Schwartz2Factor(salmonParam,t,dtype=dtype)


    from Harvest import Harvest
    hc = harvestingCosts
    harvest = Harvest(hc)

    from Growth import Bertalanffy
    wInf=6
    a=1.113
    b=1.097
    c=1.43
    growth = Bertalanffy(t,wInf,a,b,c)

    from Price import Price
    price = Price(salmon)

    from Feed import StochFeed,DetermFeed
    cr=1.1
    fc=feedingCosts
    # feed = StochFeed(fc,cr,r,t,soy)
    feed = DetermFeed(fc,cr,r,t,soy)

    from Mortality import ConstMortatlity
    n0=10000
    m=0.1
    mort = ConstMortatlity(t,n0,m)

    from FishFarm import fishFarm
    farm = fishFarm(growth,feed,price,harvest,mort,stride=Nsim,seed=seed)
    farm.seed(seed)

    batch_size=2**12
    batches=20
    gen = farm.generateFishFarm

    basis = Polynomial(deg=2,dtype=dtype)
    opt=LSMC(r,farm.tCoarse,gen,basis)

    # opt=DeepOptS(r,farm.tCoarse,gen,d=farm.d)

    opt.train(batch_size,batches)

    farm.seed(seed+1)
    tic=time.time()
    tau,Vtau=opt.evaluate(batch_size,batches)
    ctimeEval=time.time()-tic

    print(tau.shape)
    print(f'Mean stopping time {np.mean(tau)} with mean value {np.mean(Vtau)} in {ctimeEval} s')