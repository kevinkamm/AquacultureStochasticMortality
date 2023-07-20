import numpy as np
import tensorflow as tf
from Commodities import Commodity

from typing import Union

class Price():
    def __init__(self, 
                 C : Commodity):
        self.C=C
        self.isStoch=True
        self.d=C.d

    def sample(self, batch_size: int):
        return self.C.sample(batch_size)
    
    def setgen(self,gen:Union[tf.random.Generator,np.random.Generator]):
        self.C.setgen(gen)
    
    def price(self,C):
        return C[:,:,0]