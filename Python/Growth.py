import numpy as np
import tensorflow as tf

from typing import Union

class Growth():
    def __init__(self,
                 t:np.ndarray):
        self.dtype=t.dtype
        self.t=t
        self.isStoch=False
        self.d=0

    def sample(self,batch_size:int):
        pass

    def weight(self,G:np.ndarray):
        pass

    def setgen(self,gen:Union[tf.random.Generator,np.random.Generator]):
        pass

class Bertalanffy(Growth):
    def weightNP(params:np.ndarray,t:np.ndarray):
        return params[0]*(params[1]-params[2]*np.exp(-params[3]*t))**3

    @tf.function
    def weightTF(params:tf.Tensor,t:tf.Tensor):
        return params[0]*(params[1]-params[2]*tf.math.exp(-params[3]*t))**3
    def __init__(self, 
                 t:np.ndarray,
                 wInf:float, # asymptotic weight (kg)
                 a:float,
                 b:float,
                 c:float):
        super().__init__(t)
        self.params=[wInf,a,b,c]
        if type(t)==np.ndarray:
            self._weight=Bertalanffy.weightNP
        else:
            self.params=tf.constant(self.params,dtype=self.dtype)
            self._weight=Bertalanffy.weightTF

    def weight(self,G:Union[np.ndarray,tf.Tensor]):
        return self._weight(self.params,self.t)# Bertalanffy’s growth function