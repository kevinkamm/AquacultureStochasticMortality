import numpy as np
from Commodities import Commodity

from numpy.typing import DTypeLike, ArrayLike

class Feed():
    def __init__(self,
                 fc:float, # feeding costs per fish per kg in one year
                 cr: float, # conversion rate kg feed to kg fish
                 r:float, # interest rate
                 t:np.ndarray # time axis
                 ):
        self.fc=fc
        self.cr=cr
        self.r=r
        self.t=t
        self.isStoch=False

    def sample(self,
               batch_size:int):
        pass

    def cost(self,
             C:np.ndarray, # commodity process
             wt:np.ndarray, # growth process
             nt:np.ndarray # population size process
             ):
        dt = self.t[-1]/(self.t.size-1)
        dwt=np.diff(wt,1,axis=0)/dt
        cf=np.zeros_like(nt)
        cf[1:]=dwt*nt[1:]*self.cr*self.fc
        return cf
    
    def cumtotalCost(self,
                     ft:np.ndarray, # cost process
                     ):
        dt=self.t[-1]/(np.size(self.t)-1)
        return np.cumsum(np.exp(-self.r*self.t)*ft*dt,axis=0)

class StochFeed(Feed):
    def __init__(self,
                 fc: float,
                 cr: float,
                 r : float,
                 t : np.ndarray,
                 C : Commodity):
        super().__init__(fc,cr,r,t)
        self.C=C
        self.isStoch=True

    def sample(self,
               batch_size:int)->np.ndarray:
        return self.C.sample(batch_size)
    
    def cost(self,
             C:np.ndarray, # commodity process
             wt:np.ndarray, # growth process
             nt:np.ndarray # population size process
             ):
        c=C[:,:,0]/(C[0,:,0]).reshape((1,-1))
        dt = self.t[-1]/(self.t.size-1)
        dwt=np.diff(wt,1,axis=0)/dt
        cf=np.zeros_like(c)
        cf[1:,:]=dwt*nt[1:]*self.cr*self.fc*c[1:,:]
        return cf
    
class DetermFeed(StochFeed):
    def __init__(self,
                 fc: float, 
                 cr: float, 
                 r: float, 
                 t: np.ndarray, 
                 C: Commodity):
        super().__init__(fc, cr, r, t, C)
        self.isStoch=False
    
    def cost(self, 
             C: np.ndarray, 
             wt: np.ndarray, 
             nt: np.ndarray):
        return np.mean(super().cost(C, wt, nt),axis=1,keepdims=True)