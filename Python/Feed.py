import numpy as np
import tensorflow as tf
from Commodities import Commodity

from typing import Union
from numpy.typing import DTypeLike, ArrayLike

class Feed():
    def costNP( t:np.ndarray,
                cr:float,
                fc:float,
                C:np.ndarray, # commodity process
                wt:np.ndarray, # growth process
                nt:np.ndarray # population size process
                ):
        dt = t[-1]/(t.shape[0]-1)
        dwt=np.diff(wt,1,axis=0)/dt
        cf=np.zeros_like(nt)
        cf[1:]=dwt*nt[1:]*cr*fc
        return cf
    
    @tf.function
    def costTF( t:tf.Tensor,
                cr:tf.Tensor,
                fc:tf.Tensor,
                C:tf.Tensor, # commodity process
                wt:tf.Tensor, # growth process
                nt:tf.Tensor # population size process
                ):
        dt = t[-1]/(t.shape[0]-1)
        dwt=(wt[1:]-wt[:-1])/dt
        N=t.shape[0]
        M=dwt.shape[1]
        dtype=dwt.dtype
        return tf.concat([tf.zeros((1,M),dtype),dwt*nt[1:]*cr*fc],axis=0)

    def cumtotalCostNP(t:np.ndarray,ft:np.ndarray,r:float):
        dt=t[-1]/(np.size(t)-1)
        return np.cumsum(np.exp(-r*t)*ft*dt,axis=0)
    
    @tf.function
    def cumtotalCostTF(t:tf.Tensor,ft:tf.Tensor,r:tf.Tensor):
        dt=t[-1]/(t.shape[0]-1)
        return tf.cumsum(tf.math.exp(-r*t)*ft*dt,axis=0)

    def __init__(self,
                 fc:float, # feeding costs per fish per kg in one year
                 cr: float, # conversion rate kg feed to kg fish
                 r:float, # interest rate
                 t:Union[np.ndarray,tf.Tensor] # time axis
                 ):
        self.t=t
        self.dtype=t.dtype
        self.isStoch=False
        self.d=0
        if type(t)==np.ndarray:
            self.fc=np.array(fc,dtype=self.dtype)
            self.cr=np.array(cr,dtype=self.dtype)
            self.r=np.array(r,dtype=self.dtype)
            self._cost=Feed.costNP
            self._cumtotalCost=Feed.cumtotalCostNP
        else:
            self.fc=tf.constant(fc,dtype=self.dtype)
            self.cr=tf.constant(cr,dtype=self.dtype)
            self.r=tf.constant(r,dtype=self.dtype)
            self._cost=Feed.costTF
            self._cumtotalCost=Feed.cumtotalCostTF

    def sample(self,
               batch_size:int):
        pass

    def setgen(self,gen:Union[tf.random.Generator,np.random.Generator]):
        self.C.setgen(gen)

    def cost(self,
             C:Union[np.ndarray,tf.Tensor], # commodity process
             wt:Union[np.ndarray,tf.Tensor], # growth process
             nt:Union[np.ndarray,tf.Tensor] # population size process
             ):
        return self._cost(self.t,self.cr,self.fc,C,wt,nt)
    
    def cumtotalCost(self,
                     ft:Union[np.ndarray,tf.Tensor], # cost process
                     ):
        return self._cumtotalCost(self.t,ft,self.r)

class StochFeed(Feed):
    def costNP( t:np.ndarray,
                cr:float,
                fc:float,
                C:np.ndarray, # commodity process
                wt:np.ndarray, # growth process
                nt:np.ndarray # population size process
                ):
        c=C[:,:,0]/(C[0,:,0]).reshape((1,-1))
        dt = t[-1]/(t.shape[0]-1)
        dwt=np.diff(wt,1,axis=0)/dt
        cf=np.zeros_like(c)
        cf[1:]=dwt*nt[1:]*cr*fc*c[1:]
        return cf
    
    @tf.function
    def costTF( t:tf.Tensor,
                cr:tf.Tensor,
                fc:tf.Tensor,
                C:tf.Tensor, # commodity process
                wt:tf.Tensor, # growth process
                nt:tf.Tensor # population size process
                ):
        c=C[:,:,0]/tf.reshape((C[0,:,0]),(1,-1))
        dt = t[-1]/(t.shape[0]-1)
        dwt=(wt[1:]-wt[:-1])/dt
        N=t.shape[0]
        M=c.shape[1]
        dtype=c.dtype
        return tf.concat([tf.zeros((1,M),dtype),dwt*nt[1:]*cr*fc*c[1:]],axis=0)
        
    def __init__(self,
                 fc: float,
                 cr: float,
                 r : float,
                 t : np.ndarray,
                 C : Commodity):
        super().__init__(fc,cr,r,t)
        self.C=C
        self.isStoch=True
        self.d=C.d
        if type(t)==np.ndarray:
            self._cost=StochFeed.costNP
        else:
            self._cost=StochFeed.costTF

    def sample(self,
               batch_size:int)->np.ndarray:
        return self.C.sample(batch_size)
    
class DetermFeed(StochFeed):

    def __init__(self,
                 fc: float, 
                 cr: float, 
                 r: float, 
                 t: np.ndarray, 
                 C: Commodity):
        super().__init__(fc, cr, r, t, C)
        self.isStoch=False
        self.d=0
    
    def cost(self, 
             C: np.ndarray, 
             wt: np.ndarray, 
             nt: np.ndarray):
        if type(self.t)==np.ndarray:
            return np.mean(super().cost(C, wt, nt),axis=1,keepdims=True)
        else:
            return tf.reduce_mean(super().cost(C, wt, nt),axis=1,keepdims=True)