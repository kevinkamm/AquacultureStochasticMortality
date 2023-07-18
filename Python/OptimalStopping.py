import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from joblib import Parallel,delayed
from psutil import cpu_count

from typing import Optional, Callable, Generator
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

class OptimalStopping:
    def __init__(self,
                 r:float,
                 t:np.ndarray,
                 gen:Callable) -> None:
        self.r=r
        self.t=t
        self.gen=gen

    def train(self,batch_size,batches:int):
        pass

    def evaluate(self,batch_size,batches:int):
        pass

    # def solve(self):
    #     if self.batches >1:
    #         out = Parallel(n_jobs=num_cores)(delayed(self.solveBatch)() for i in range(self.batches))
    #     else:
    #         out = self.solveBatch()
    #     return out

    # def solveBatch(self):
    #     pass


class LSMC(OptimalStopping):
    def __init__(self,
                 r:float,
                 t:np.ndarray,
                 gen:Callable,
                 b:Basis) -> None:
        super().__init__(r,t,gen)
        self.b=b

    def evaluateBatch(self,batch_size:int):
        X,V,VH,ft = self.gen(batch_size)
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
        tau,Vtau = zip(*Parallel(n_jobs=num_cores)(delayed(self.evaluateBatch)(batch_size) for _ in range(0,batches) ) )
        # tau,Vtau = self.evaluateBatch(batch_size)
        return tau,Vtau

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
    feed = StochFeed(fc,cr,r,t,soy)
    # feed = DetermFeed(fc,cr,r,t,soy)

    from Mortality import ConstMortatlity
    n0=10000
    m=0.1
    mort = ConstMortatlity(t,n0,m)

    from FishFarm import fishFarm
    farm = fishFarm(growth,feed,price,harvest,mort,stride=Nsim)
    gen = farm.generateFishFarm

    deg=2
    batch_size=100000
    batches=1
    gen = farm.generateFishFarm

    basis = Polynomial(deg=deg,dtype=dtype)

    opt=LSMC(r,farm.tCoarse,gen,basis)
    # opt.train(batches)

    rngSoy=np.random.default_rng(seed*100+1)
    rngSalmon=np.random.default_rng(seed*100+2)
    salmon.setgen(rngSalmon)
    soy.setgen(rngSoy)
    tau,Vtau=opt.evaluate(batch_size,batches)

    print(f'Mean stopping time {np.mean(tau)} with mean value {np.mean(Vtau)}')