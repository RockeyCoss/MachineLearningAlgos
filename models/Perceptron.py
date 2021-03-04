import numpy as np
import json
from models import ModelBaseClass



class Perceptron(ModelBaseClass):
    def __init__(self,lr=0.01):
        self.lr=lr
        self.alpha=None
        self.b=None
        self.w=None

    def getGram(self,features):
        gram=np.zeros((len(features),len(features)))
        for i in range(gram.shape[0]):
            for j in range(0,i+1):
                value=np.sum(features[i]*features[j])
                gram[i,j]=gram[j,i]=value
        return gram

    def train(self,features,labels,*args,**dicts):
        gram=self.getGram(features)
        self.alpha = np.zeros(len(features))
        self.b = 0
        epoch=1
        while(True):
            allClassfied = True
            count=0
            for index in range(len(features)):
                xx =gram[:,index]
                linear=np.sum(self.alpha*labels*xx)+self.b
                if labels[index]*linear<=0:
                    self.alpha[index]+=self.lr
                    self.b+=self.lr*labels[index]
                    allClassfied=False
                    count+=1
            print(f"epoch:{epoch} b:{self.b} total:{len(features)} misclassfied:{count}")
            if allClassfied:
                break

            epoch+=1
        result={}
        w=np.zeros(features.shape[1])
        alphaLabels=self.alpha*labels
        for index in range(features.shape[0]):
            w+=alphaLabels[index]*features[index]
        result["w"]=w.tolist()
        result["b"]=self.b
        self.save(result)
        print("Learning Completed")

    def predict(self, features):
        self.loadPara()
        result=[]
        for aFeature in features:
            linear = np.sum(self.w*aFeature) + self.b
            result.append(-1 if linear<0 else 1)
        return np.array(result)

    def loadPara(self):
        para=self.loadJson()
        self.w=np.array(para["w"])
        self.b=para["b"]



if __name__ == "__main__":
    pass