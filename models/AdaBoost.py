import numpy as np
from models import ModelBaseClass



class AdaBoost(ModelBaseClass):
    def __init__(self,lr=0.01):
        self.lr=lr
        self.alpha=None
        self.b=None
        self.w=None

    def train(self, features: np.ndarray, labels: np.ndarray, *args, **dicts):
        super().train(features, labels, *args, **dicts)

    def predict(self, features: np.ndarray):
        super().predict(features)

    def loadPara(self):
        super().loadPara()

class DesicionStump:
    def __init__(self,step=0.01):
        self.step=step

    def train(self,features:np.ndarray,weight:np.ndarray):
        


if __name__ == "__main__":
    pass