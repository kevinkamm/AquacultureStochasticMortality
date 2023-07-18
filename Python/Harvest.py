import numpy as np

class Harvest():
    def __init__(self,
                 hc:float):
        self.hc = hc
        self.isStoch=False
        pass

    def sample(self,
               batch_size:int):
        pass

    def cost(self, C:np.ndarray):
        return self.hc

    def totalCost(self,C:np.ndarray,bt:np.ndarray):
        c=self.cost(C)
        return c * bt