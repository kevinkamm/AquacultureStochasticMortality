import numpy as np
class Mortality():
    def __init__(self):
        self.isStoch=False
        pass

    def sample(self,
               batch_size:int):
        pass

    def populationSize(self,
                       M:np.ndarray # host process
                       ):
        pass

    def treatmentCost(self,M):
        return 0

class ConstMortatlity(Mortality):
    def __init__(self, 
                 t:np.ndarray,
                 n0:int,
                 m:float):
        super().__init__()
        self.m=m
        self.n0=n0
        self.t=t

    def populationSize(self,
                       M:np.ndarray # host process
                       ):
        return self.n0*np.exp(-self.m*self.t)
        # return ne.evaluate("n0*exp(-m*t)",local_dict={'n0':self.n0,'m':self.m,'t':self.t})
    

class HostParasite(Mortality):
    def __init__(self):
        super().__init__()
        self.isStoch=True

    def sample(self,batch_size:int):
        pass