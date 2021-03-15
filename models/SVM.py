import numpy as np
import json
import sklearn
from models import ModelBaseClass

class SVM(ModelBaseClass):
    def __init__(self):
        self.featureNum=0
    def rbf(self,x1,x2):
        return np.exp(-np.power((x1-x2),2))/self.featureNum
    def getKernelMatrix(self,kernel)->np.ndarray:
        pass
    def train(self, features: np.ndarray, labels: np.ndarray, *args, **dicts):
        self.featureNum=features.shape[1]

    def predict(self, features: np.array):
        super().predict(features)

        sklearn.svm()
    def loadPara(self):
        super().loadPara()