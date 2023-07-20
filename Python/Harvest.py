import numpy as np
import tensorflow as tf
from typing import Union
class Harvest():
    def __init__(self,
                 hc:float):
        self.hc = hc
        self.isStoch=False
        self.d=0
        pass

    def sample(self,
               batch_size:int):
        pass

    def setgen(self,gen:Union[tf.random.Generator,np.random.Generator]):
        pass

    def cost(self, C:np.ndarray):
        return self.hc

    def totalCost(self,C:np.ndarray,bt:np.ndarray):
        c=self.cost(C)
        return c * bt