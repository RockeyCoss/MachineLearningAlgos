import numpy as np
import array
import collections
from models import ModelBaseClass
from utilities import loadConfigWithName


class AdaBoost(ModelBaseClass):
    def __init__(self):
        self.weakClassifiers=[]
        self.classifierNum=int(loadConfigWithName("AdaBoostConfig", "classfierNum"))

    def train(self, features: np.ndarray, labels: np.ndarray, *args, **dicts):
        weight=np.ones(features.shape[0])*(1/features.shape[0])
        #reuse one decision stump instance, save memory and time
        stump = DesicionStump()
        for dummy in range(self.classifierNum):
            Em=stump.train(features,labels,weight)
            alpham=0.5*np.log((1-Em)/Em)
            Gm=stump.predictABunch(features)
            exp=np.exp(-1*alpham*labels*Gm)
            weightExp=weight*exp
            Zm=np.sum(weightExp)
            weight=weightExp/Zm
            self.weakClassifiers.append([alpham, stump.getState()])

        self.save(self.weakClassifiers)

    def predict(self, features: np.ndarray):
        self.loadPara()
        stump=DesicionStump()
        result=[]
        for aFeature in features:
            fx=0
            for aClassifier in self.weakClassifiers:
                stump.loadState(*aClassifier[1])
                Gm=stump.predictOne(aFeature)
                fx+=aClassifier[0]*Gm
            result.append(1) if fx>=0 else result.append(-1)
        return np.array(result)

    def loadPara(self):
        self.weakClassifiers=self.loadJson()

class DesicionStump:
    def __init__(self,cutValue=0,cutColumn=0,preLabel=0,postLabel=0):
        self.cutValue=cutValue
        self.cutColumn=cutColumn
        self.preLabel=preLabel
        self.postLabel=postLabel

    def train(self,features:np.ndarray,labels:np.ndarray,weight:np.ndarray)->float:
        featuresLabelsWeight=np.concatenate((features,labels.reshape(-1,1),weight.reshape(-1,1)),axis=1)
        globalCurrentMinFault=1

        for column in range(features.shape[1]):
            #data preparation
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
        #return Em
        return globalCurrentMinFault

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

        labels=np.ones(features.shape[0])
        if self.preLabel==-1:
            labels[np.where(features[:,self.cutColumn]<=self.cutValue)]=-1
        else:
            labels[np.where(features[:,self.cutColumn]>self.cutValue)]=-1
        return labels

    def __repr__(self):
        returnString=f"cutValue:{self.cutValue} cutColumn:{self.cutColumn} preLabel:{self.preLabel} postLabel:{self.postLabel}"
        return returnString

    def getState(self)->list:
        return [self.cutValue,self.cutColumn,self.preLabel,self.postLabel]

    def loadState(self,cutValue,cutColumn,preLabel,postLabel):
        self.cutValue = cutValue
        self.cutColumn = cutColumn
        self.preLabel = preLabel
        self.postLabel = postLabel




if __name__ == "__main__":
    pass