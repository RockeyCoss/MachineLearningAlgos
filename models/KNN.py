import numpy as np
from models import ModelBaseClass
from utilities import loadConfigWithName


class KNN(ModelBaseClass):
    def train(self, features: np.array, labels: np.array, *args, **dicts):
        super().train(features, labels, *args, **dicts)

    def predict(self, features: np.array):
        super().predict(features)

    def loadPara(self):
        super().loadPara()


class Node:
    def __init__(self, points: list = None, father=None, lChild=None, rChild=None, axis: int = None):
        self.points = points
        self.father = father
        self.lChild = lChild
        self.rChild = rChild
        self.axis = axis


class kdTree:
    def __init__(self, root=None):
        self.root = root

    def medianSplit(self, data, axis):
        sortedData = np.array(sorted(data, key=lambda x: x[axis]))
        medianIndex = sortedData.shape[0] // 2
        leftSame, rightSame = True
        leftStep = 1
        rightStep = 1
        medianValue = sortedData[medianIndex, axis]
        while leftSame or rightSame:
            if sortedData[medianIndex - leftStep, axis] < medianValue:
                leftSame = False
            else:
                leftStep += 1

            if sortedData[medianIndex + rightStep, axis] > medianValue:
                rightSame = False
            else:
                rightStep += 1

        return sortedData[:medianIndex - leftStep + 1], sortedData[
                                                        medianIndex - leftStep + 1:medianIndex + rightStep], sortedData[
                                                                                                             medianIndex + rightStep:]

    def createKdTree(self, features: np.array):
        self.root = Node(axis=0)
