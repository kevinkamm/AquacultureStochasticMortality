import numpy as np
import numexpr as ne
import tensorflow as tf
import time
from pathlib import Path

from Growth import Growth
from Feed import Feed
from Price import Price
from Harvest import Harvest
from Mortality import Mortality
from OptimalStopping import OptimalStopping

from typing import Optional,List

"Auxiliary Functions"

class fishFarm():
    def compareStoppingTimes(V:np.ndarray,taus:List[np.ndarray],t:np.ndarray):
        M=V.shape[1]
        Vout=[]
        Tout=[]
        for tau in taus:
            Vtmp=0
            Ttmp=0
            for wi in range(0,M):
                ti = np.where(t>=tau[wi])[0][0]
                Vtmp += V[ti,wi]
                Ttmp += tau[wi]
            Vout.append(Vtmp/M)
            Tout.append(Ttmp/M)
        return Tout, Vout

    def coarseNP(X,stride):
        if stride>1 and np.size(X)>1:
            sz=X.shape
            return np.concatenate([X[0].reshape((1,)+sz[1:]),X[stride-1::stride]],axis=0)
        else:
            return X
    def coarseTF(X,stride):
        if stride>1 and tf.size(X)>1:
            sz=X.shape
            return tf.concat([tf.reshape(X[0],(1,)+sz[1:]),X[stride-1::stride]],axis=0)
        else:
            return X
    def __init__(self,
                 growth:Growth,
                 feed:Feed,
                 price:Price,
                 harvest:Harvest,
                 mort:Mortality,
                 stride:int = 1,
                 seed:Optional[int] = None):
        self.r=feed.r
        self.growth=growth
        self.feed=feed
        self.price=price
        self.harvest=harvest
        self.mort=mort
        self.stride=stride
        if type(feed.t)==np.ndarray:
            self._coarse=fishFarm.coarseNP
            self.tCoarse=self._coarse(feed.t,self.stride)
            self.discount=np.exp(-self.r*self.tCoarse) # discount factor
        else:
            self.stride=tf.constant(stride)
            self._coarse=fishFarm.coarseTF
            self.tCoarse=self._coarse(feed.t,self.stride)
            self.discount=tf.math.exp(-self.r*self.tCoarse) # discount factor
        d=0
        d+=harvest.d
        d+=growth.d
        d+=price.d
        d+=feed.d
        d+=mort.d
        self.d=d # number of stochastic processes
        if seed is not None:
            self.seed(seed)

    def seed(self,seed:int):
        seed=int(seed)
        if type(self.feed.t)==np.ndarray:
            rngGrowth=np.random.default_rng(seed*100+1)
            rngFeed=np.random.default_rng(seed*100+2)
            rngPrice=np.random.default_rng(seed*100+3)
            rngHarvest=np.random.default_rng(seed*100+4)
            rngMort=np.random.default_rng(seed*100+5)
        else:
            rngGrowth=tf.random.Generator.from_seed(seed*100+1)
            rngFeed=tf.random.Generator.from_seed(seed*100+2)
            rngPrice=tf.random.Generator.from_seed(seed*100+3)
            rngHarvest=tf.random.Generator.from_seed(seed*100+4)
            rngMort=tf.random.Generator.from_seed(seed*100+5)
        self.growth.setgen(rngGrowth)
        self.feed.setgen(rngFeed)
        self.price.setgen(rngPrice)
        self.harvest.setgen(rngHarvest)
        self.mort.setgen(rngMort)


    def generateFishFarm(self,batch_size:int):
        "Growth of Fish"
        W=self.growth.sample(batch_size) # sample processes
        wt=self.growth.weight(W) # compute weight

        "Number of Fishes"
        N = self.mort.sample(batch_size) # sample processes
        nt = self.mort.populationSize(N) # compute population size
        tt = self.mort.treatmentCost(N) # compute treatment cost

        if self.mort.isStoch:
            N=N[:,:,:self.mort.d]
        else:
            N=None

        "Total Biomass"
        bt=wt*nt

        "Price of Feed"
        F=self.feed.sample(batch_size) # sample processes
        ft=self.feed.cost(F,wt,nt) # compute price
        cumft=self.feed.cumtotalCost(ft) # compute price
        if not self.feed.isStoch:
            F=None

        "Price of Fish"
        P=self.price.sample(batch_size) # sample processes
        pt=self.price.price(P) # compute price

        "Harvesting Costs"
        H=self.harvest.sample(batch_size) # sample processes
        ht=self.harvest.totalCost(H,bt) # compute costs

        "sim points -> eval points"
        if type(self.feed.t)==np.ndarray:
            X = np.concatenate([self._coarse(Y,self.stride) for Y in [W,N,P,F,H] if Y is not None] ,axis=2) #time x simulations x processes
        else:
            X = tf.concat([self._coarse(Y,self.stride) for Y in [W,N,P,F,H] if Y is not None] ,axis=2) #time x simulations x processes
        # t = coarse(self.feed.t,self.stride)
        tt = self._coarse(tt,self.stride)
        ft = self._coarse(ft,self.stride)
        pt = self._coarse(pt,self.stride)
        bt = self._coarse(bt,self.stride)
        ht = self._coarse(ht,self.stride)
        cumft = self._coarse(cumft,self.stride)

        VH=(1-tt)*pt*bt-ht # harvesting value
        V=self.discount*VH-cumft # total value of farm

        return X,V,VH,ft
    
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
    feed = StochFeed(fc,cr,r,t,soy)
    # feed = DetermFeed(fc,cr,r,t,soy)

    from Mortality import ConstMortatlity,HostParasite,DetermHostParasite,Poisson,DetermPoisson
    # n0=10000
    # m=0.1
    # mort = ConstMortatlity(t,n0,m)
    params=[0.05,0.1,8.71,0.05]
    beta=[0.0835,0.0244]
    H0=10000.0
    P0=1
    tData=np.array([0.018868,0.037736,0.056604,0.075472,0.09434,0.11321,0.13208,0.15094,0.16981,0.18868,0.20755,0.22642,0.24528,0.26415,0.28302,0.30189,0.32075,0.33962,0.35849,0.37736,0.39623,0.41509,0.43396,0.45283,0.4717,0.49057,0.50943,0.5283,0.54717,0.56604,0.58491,0.60377,0.62264,0.64151,0.66038,0.67925,0.69811,0.71698,0.73585,0.75472,0.77358,0.79245,0.81132,0.83019,0.84906,0.86792,0.88679,0.90566,0.92453,0.9434,0.96226,0.98113,1,1.0189,1.0377,1.0755,1.0943,1.1132,1.1321,1.1509,1.1698,1.1887,1.2075,1.2264,1.2453,1.2642,1.283,1.3019,1.3208,1.3396,1.3585,1.3774,1.3962,1.4151,1.434,1.4528,1.4717,1.4906,1.5094,1.5283,1.5472,1.566,1.5849,1.6038
    ],dtype=np.float32).reshape((-1,))
    dm=np.array([0.45003,0,0.45003,0,0,0.90005,0,0,1.3501,0.90005,1.8001,1.3501,1.8001,2.2501,3.1502,1.8001,1.8001,2.7002,2.2501,2.2501,2.7002,1.8001,3.1502,3.1502,3.1502,5.8503,5.8503,6.7504,4.9503,6.3004,4.0502,7.2004,7.6504,8.5505,9.0005,9.9006,11.701,12.601,11.701,13.951,14.401,13.051,12.601,14.401,12.601,11.251,14.401,13.951,16.201,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501,13.501
    ],dtype=np.float32).reshape((-1,))

    # mort_s = HostParasite(t,params,beta,H0,P0)
    # mort_d = DetermHostParasite(t,params,beta,H0,P0)
    mort = Poisson(H0,params[0],t,tData[1:],dm)
    # mort = DetermPoisson(H0,params[0],t,tData[1:],dm)

    from FishFarm import fishFarm
    farm = fishFarm(growth,feed,price,harvest,mort,stride=Nsim,seed=seed)
    farm.seed(seed)

    batch_size=2**13
    batches=20
    gen = farm.generateFishFarm

    tic=time.time()
    X,V,VH,ft = gen(batch_size)
    ctime=time.time() - tic
    print(f'Elapsed time {ctime} s')
    tic=time.time()
    X,V,VH,ft = gen(batch_size)
    ctime=time.time() - tic
    print(f'Elapsed time {ctime} s')
    tic=time.time()
    X,V,VH,ft = gen(batch_size)
    ctime=time.time() - tic
    print(f'Elapsed time {ctime} s')
    tic=time.time()
    X,V,VH,ft = gen(batch_size)
    ctime=time.time() - tic
    print(f'Elapsed time {ctime} s')