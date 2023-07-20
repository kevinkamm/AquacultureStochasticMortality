import numpy as np
import tensorflow as tf
from typing import Union

class Mortality():
    def __init__(self):
        self.isStoch=False
        self.d=0
        pass

    def sample(self,batch_size:int):
        pass

    def setgen(self,gen:Union[tf.random.Generator,np.random.Generator]):
        pass

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
                 m:float):
        super().__init__()
        self.t=t
        self.dtype=t.dtype
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
    def __init__(self):
        super().__init__()
        self.isStoch=True

    def sample(self,batch_size:int):
        pass