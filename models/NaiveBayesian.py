import numpy as np
import json
import collections
from models import ModelBaseClass
#Use Laplacian smoothing
#To avoid underflow, use log to convert mutiplication to addition
#only work for designated form of MNIST data

class NaiveBayesian(ModelBaseClass):
    def __init__(self,lambdaPara=1):
        self.lambdaPara=1
        self.classNum=10
        self.featureNum=28*28
        self.featurePossibleValueNum=256
        self.Py=None
        self.Pxy=None
    def train(self, features, labels, *args, **dicts):
        #estimate P(Y=ck)
        labelCount=collections.Counter(labels)
        sampleCount=labels.shape[0]
        PyList=np.zeros(self.classNum)
        for aLabel in labelCount:
            PyList[aLabel]=(labelCount[aLabel]+self.lambdaPara)/(sampleCount+self.classNum*self.lambdaPara)

        #estimate P(Xi|Y)
        #PxyList:[label][featureI][featureValue]
        PxyList=np.zeros((self.classNum,self.featureNum,self.featurePossibleValueNum))
        for sampleIndex in range(features.shape[0]):
            for featureIndex in range(features.shape[1]):
                PxyList[labels[sampleIndex],featureIndex,features[sampleIndex,featureIndex]]+=1
        for aLabelIndex in range(PxyList.shape[0]):
            PxyList[aLabelIndex,:,:]=(PxyList[aLabelIndex,:,:]+self.lambdaPara)/(labelCount[aLabel]+self.featurePossibleValueNum*self.lambdaPara)

        PyList=np.log(PyList)
        PxyList=np.log(PxyList)
        saveDict={}
        saveDict["Py"]=PyList.tolist()
        saveDict["Pxy"]=PxyList.tolist()
        self.save(saveDict)

    def predict(self, features):
        self.loadPara()
        result = []
        for aFeature in features:
            possibilityOfAClass=self.Py.copy()
            for aLabel in range(self.classNum):
                for featureI in range(aFeature.shape[0]):
                    possibilityOfAClass[aLabel]+=self.Pxy[aLabel,featureI,aFeature[featureI]]
            predictClass=np.argmax(possibilityOfAClass)
            result.append(predictClass)
        return np.array(result)


    def loadPara(self):
        para=self.loadJson()
        self.Py=np.array(para["Py"])
        self.Pxy=np.array(para["Pxy"])
