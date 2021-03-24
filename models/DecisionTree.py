import collections

import numpy as np
from models import ModelBaseClass
# use cart algorithm to generate decision tree
from utilities import loadConfigWithName
from array import array


class DecisionTree(ModelBaseClass):
    def __init__(self):
        self.minSampleNum=int(loadConfigWithName("DecisionTreeConfig", "minSampleNum"))
        self.giniThreshold=float(loadConfigWithName("DecisionTreeConfig", "giniThreshold"))
        self.classNum=int(loadConfigWithName("DecisionTreeConfig","classNum"))
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

    def __giniDivided(self,preState:array,postState:array,totalNum)->float:
        preState=np.array(preState)
        postState=np.array(postState)

        preStateSum=np.sum(preState)
        postStateSum=np.sum(postState)
        preGini=(preStateSum/totalNum)*(1-np.sum(np.power((preState/preStateSum),2)))
        postGini=(postStateSum/totalNum)*(1-np.sum(np.power((postState/postStateSum),2)))
        return preGini+postGini

    def __constructTree(self,features:np.ndarray,labels:np.ndarray):
        if features <= self.minSampleNum:
            self.root.classValue = np.max(labels)
            return
        gini=self.__gini(labels)
        globalMinGini=float("inf")
        for column in range(features.shape[1]):
            featureLabel=np.concatenate((features[:,column].reshape(-1,1),labels.reshape(-1,1)),axis=1)
            sortedFeatureBel=sorted(featureLabel,key=lambda x:x[0])
            #state:
            preList=[0 for dummy in range(self.classNum)]
            preList[int(sortedFeatureBel[0,1])]=1
            preState=array('b',preList)
            counter=collections.Counter(sortedFeatureBel[1:,1].astype(np.int))
            postState=array('b',[counter[i] if i in counter.keys() else 0 for i in range(self.classNum)])
            #divide the data into [:currentCutPoint+1] and [currentCutPoint+1,:] two parts
            currentCutPoint=0
            currentMinGini = self.__giniDivided(preState,postState,features.shape[0])
            currentMinCut = 0



class Node:
    def __init__(self,cutColumn=0,cutValue=0,classValue=0,lChild=None,rChild=None,index=-1):
        self.cutColumn=cutColumn
        self.cutValue=cutValue
        self.classValue=classValue
        #可以为Node类或指示类序号的int，方便恢复一棵树。
        self.lChild=lChild
        self.rChild=rChild
        self.index=index