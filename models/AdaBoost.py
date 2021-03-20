import numpy as np
from models import ModelBaseClass



class Perceptron(ModelBaseClass):
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


if __name__ == "__main__":
    pass