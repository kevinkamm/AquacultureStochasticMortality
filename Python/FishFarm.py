import numpy as np
import numexpr as ne
import time
from pathlib import Path

from Growth import Growth
from Feed import Feed
from Price import Price
from Harvest import Harvest
from Mortality import Mortality
from OptimalStopping import OptimalStopping

from typing import List

"Auxiliary Functions"
def coarse(X,stride):
    if stride>1 and np.size(X)>1:
        sz=X.shape
        return np.concatenate([X[0].reshape((1,)+sz[1:]),X[stride-1::stride]],axis=0)
    else:
        return X


class fishFarm():
    def __init__(self,
                 growth:Growth,
                 feed:Feed,
                 price:Price,
                 harvest:Harvest,
                 mort:Mortality,
                 stride:int = 1):
        self.r=feed.r
        self.growth=growth
        self.feed=feed
        self.price=price
        self.harvest=harvest
        self.mort=mort
        self.stride=stride
        self.tCoarse=coarse(feed.t,stride)

    def generateFishFarm(self,batch_size:int):
        # while True:
        "Growth of Fish"
        W=self.growth.sample(batch_size) # sample processes
        wt=self.growth.weight(W) # compute weight

        "Number of Fishes"
        N = self.mort.sample(batch_size) # sample processes
        nt = self.mort.populationSize(N) # compute population size
        tt = self.mort.treatmentCost(N) # compute treatment cost

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
        X = np.concatenate([coarse(Y,self.stride) for Y in [W,N,P,F,H] if Y is not None] ,axis=2) #time x simulations x processes
        # t = coarse(self.feed.t,self.stride)
        tt = coarse(tt,self.stride)
        ft = coarse(ft,self.stride)
        pt = coarse(pt,self.stride)
        bt = coarse(bt,self.stride)
        ht = coarse(ht,self.stride)
        cumft = coarse(cumft,self.stride)

        VH=(1-tt)*(pt*bt-ht) # harvesting value
        discount=np.exp(-self.r*self.tCoarse) # discount factor
        V=discount*VH-cumft # total value of farm

        return X,V,VH,ft
            # yield X,V,VH,ft


if __name__=="__main__":
    import time
    T=3.0
    N=int(T*2*12)
    Nsim = 10
    dtype=np.float32
    seed=1

    t=np.linspace(0,T,N*Nsim,endpoint=True,dtype=dtype).reshape((-1,1))
    r=0.0303

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

    rngSoy=np.random.default_rng(seed*100+1)
    soy=Schwartz2Factor(soyParam,t,dtype=dtype,rng=rngSoy)
    rngSalmon=np.random.default_rng(seed*100+2)
    salmon=Schwartz2Factor(salmonParam,t,dtype=dtype,rng=rngSalmon)


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
    feed = StochFeed(cr,fc,r,t,soy)
    # feed = DetermFeed(cr,fc,r,t,soy)

    from Mortality import ConstMortatlity
    n0=10000
    m=0.1
    mort = ConstMortatlity(t,n0,m)

    batch_size=8196
    farm = fishFarm(growth,feed,price,harvest,mort,stride=Nsim)
    X,V,VH,ft  = farm.generateFishFarm(batch_size)
    print(V[-1,1])