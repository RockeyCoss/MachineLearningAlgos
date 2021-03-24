import collections

import numpy as np
from models import ModelBaseClass
# use cart algorithm to generate decision tree
from utilities import loadConfigWithName


class DecisionTree(ModelBaseClass):
    def __init__(self):
        self.minSampleNum=int(loadConfigWithName("DecisionTreeConfig", "minSampleNum"))
        self.giniThreshold=float(loadConfigWithName("DecisionTreeConfig", "giniThreshold"))
        self.root:Node=None

    def train(self, features: np.ndarray, labels: np.ndarray, *args, **dicts):
        self.root=Node()
        if features <= self.minSampleNum:
            self.root.classValue = np.max(labels)
            # save还没写
            return
        gini=self.__gini(labels)





    def predict(self, features: np.ndarray):
        super().predict(features)

    def loadPara(self):
        super().loadPara()

    def __gini(self,labels:np.ndarray)->float:
        counter=collections.Counter(labels)
        totalNum=labels.shape[0]
        squareSum=0
        for key in counter.keys():
            squareSum+=np.power(counter[key]/totalNum,2)
        return 1-squareSum

    def __constructTree(self,features:np.ndarray,labels:np.ndarray):
        if features <= self.minSampleNum:
            self.root.classValue = np.max(labels)
            return
        gini=self.__gini(labels)
        for column in range(features.shape[1]):
            


class Node:
    def __init__(self,cutColumn=0,cutValue=0,classValue=0,lChild=None,rChild=None,index=-1):
        self.cutColumn=cutColumn
        self.cutValue=cutValue
        self.classValue=classValue
        #可以为Node类或指示类序号的int，方便恢复一棵树。
        self.lChild=lChild
        self.rChild=rChild
        self.index=index