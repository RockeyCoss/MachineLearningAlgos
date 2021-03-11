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
    def __init__(self, points: np.array = None, father=None, lChild=None, rChild=None, axis: int = None):
        self.points = points
        self.father = father
        self.lChild = lChild
        self.rChild = rChild
        self.axis = axis


class kdTree:
    def __init__(self, root=None):
        self.root = root

    def medianSplit(self, features: np.ndarray, axis: int):
        if features.shape[0] == 1:
            return None, features, None
        sortedData = np.array(sorted(features, key=lambda x: x[axis]))
        medianIndex = sortedData.shape[0] // 2
        leftSame, rightSame = True
        leftStep = 1
        rightStep = 1
        medianValue = sortedData[medianIndex, axis]

        while leftSame or rightSame:
            if medianIndex - leftStep < 0 or sortedData[medianIndex - leftStep, axis] < medianValue:
                leftSame = False
            else:
                leftStep += 1

            if medianIndex + rightStep >= sortedData.shape[0] or sortedData[
                medianIndex + rightStep, axis] > medianValue:
                rightSame = False
            else:
                rightStep += 1

        leftData = sortedData[:medianIndex - leftStep + 1]
        medianData = sortedData[medianIndex - leftStep + 1:medianIndex + rightStep]
        rightData = sortedData[medianIndex + rightStep:]

        if leftData.shape[0] == 0:
            leftData = None
        if rightData.shape[0] == 0:
            rightData = None

        return leftData, medianData, rightData

    def createKdTree(self, features: np.ndarray):
        self.root = Node()
        assert type(features) == np.ndarray and (features.shape) == 2
        self.createChild(features, self.root, 0)

    def createChild(self, features: np.ndarray, currentNode: Node, depth: int):
        axis = depth % features.shape[1]
        currentNode.axis = axis
        leftData, medianData, rightData = self.medianSplit(features, axis=axis)
        assert len(medianData.shape) == 2
        currentNode.points = medianData
        if leftData == None:
            currentNode.lChild = None
        else:
            currentNode.lChild = Node(father=currentNode)
            self.createChild(leftData, currentNode.lChild, depth + 1)

        if rightData == None:
            currentNode.rChild = None
        else:
            currentNode.rChild = Node(father=currentNode)
            self.createChild(rightData, currentNode.rChild, depth + 1)

        return
