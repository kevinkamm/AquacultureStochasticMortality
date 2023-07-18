import numpy as np

class Growth():
    def __init__(self,
                 t:np.ndarray):
        self.dtype=t.dtype
        self.t=t
        self.isStoch=False

    def sample(self,
               batch_size:int):
        pass

    def weight(self,G:np.ndarray):
        pass

class Bertalanffy(Growth):
    def __init__(self, 
                 t:np.ndarray,
                 wInf:float, # asymptotic weight (kg)
                 a:float,
                 b:float,
                 c:float):
        super().__init__(t)
        self.a=a
        self.b=b
        self.c=c
        self.wInf=wInf

    def weight(self,G:np.ndarray):
        return self.wInf*(self.a-self.b*np.exp(-self.c*self.t))**3 # Bertalanffy’s growth function
    # return ne.evaluate("wInf*(a-b*exp(-c*t))**3 ",local_dict={'wInf':self.wInf,'a':self.a,'b':self.b,'c':self.c,'t':self.t})