import numpy as np
from Commodities import Commodity
class Price():
    def __init__(self, 
                 C : Commodity):
        self.C=C
        self.isStoch=True

    def sample(self, batch_size: int):
        return self.C.sample(batch_size)
    
    def price(self,C):
        return C[:,:,0]