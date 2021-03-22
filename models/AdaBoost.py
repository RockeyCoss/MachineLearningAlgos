import numpy as np
import array
import collections
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
    def __init__(self):
        self.cutValue=0
        self.cutColumn=0
        self.preLabel=0
        self.postLabel=0

    def train(self,features:np.ndarray,labels:np.ndarray,weight:np.ndarray):
        featuresLabelsWeight=np.concatenate((features,labels.reshape(-1,1),weight.reshape(-1,1)),axis=1)
        globalCurrentMinFault=np.sum(weight)
        for column in features.shape[1]:
            sortedFeaturesLabelsWeight=np.array(sorted(featuresLabelsWeight, key=lambda x:x[column]))
            sortedLabels= sortedFeaturesLabelsWeight[:, sortedFeaturesLabelsWeight.shape[1] - 2].astype(np.int)
            sortedWeights=sortedFeaturesLabelsWeight[:, sortedFeaturesLabelsWeight.shape[1] - 1]
            #use dynamic programing to find the cut point in O(N) time
            preState=array.array('f',[sortedWeights[0],0] if sortedLabels[0]==-1 else [0,sortedWeights[0]])
            weightSumOfMiuns1=np.sum(sortedWeights[np.where(sortedLabels==-1)])
            weightSumOf1=1-weightSumOfMiuns1
            postState=array.array('f',[weightSumOfMiuns1,weightSumOf1])
            preLabel=-1 if np.argmax(preState)==0 else 1
            postLabel=-1 if np.argmax(postState)==0 else 1
            currentMinFault=min(preState)+min(postState)
            currentMinCut=0
            #cutPoint表示分为[:cutPoint+1],[cutPoint+1,:]俩组
            for cutPoint in range(1,sortedLabels.shape[0]):
                if sortedLabels[cutPoint]==-1:
                    preState[0]+=sortedWeights[cutPoint]
                    postState[0]-=sortedWeights[cutPoint]
                else:
                    preState[1]+=sortedWeights[cutPoint]
                    postState[1]-=sortedWeights[cutPoint]
                currentFault=min(preState)+min(postState)
                if currentFault<currentMinFault:
                    currentMinFault=currentFault
                    preLabel = -1 if np.argmax(preState) == 0 else 1
                    postLabel = -1 if np.argmax(postState) == 0 else 1
                    currentMinCut=cutPoint
            if globalCurrentMinFault>currentMinFault:
                globalCurrentMinFault=currentMinFault
                self.cutValue=sortedFeaturesLabelsWeight[currentMinCut, column]
                self.cutColumn=column
                self.preLabel=preLabel
                self.postLabel=postLabel

    def predictOne(self,feature:np.ndarray):
        if self.preLabel==self.postLabel:
            return self.preLabel
        label=0
        if feature[self.cutColumn]<=self.cutValue:
            label=self.preLabel
        else:
            label=self.postLabel
        return label

    def predictABunch(self,features:np.ndarray):
        if self.preLabel == self.postLabel:
            return np.ones(features.shape[0])*self.preLabel

        judgeVector=features[:,self.cutColumn]<=self.cutValue
        labels=np.ones(features.shape[0])
        if self.preLabel==-1:
            labels[np.where(judgeVector==True)]=-1
        else:
            labels[np.where(judgeVector!=True)]=-1
        return labels





if __name__ == "__main__":
    pass