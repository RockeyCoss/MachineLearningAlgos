import numpy as np
from models import ModelBaseClass
from utilities import BFGSAlgo,loadConfigWithName


# Remember that for logistic model, the label yi is either 0 or 1, not -1 or 1!
# 使用SGD或拟牛顿法优化
# 使用拟牛顿法优化时有可能遇到数值问题……以后再调试了
class Logistic(ModelBaseClass):
    def __init__(self):
        self.w = None

    def f(self,w,features,labels):
        wxi = np.sum(w.reshape(-1, ) * features, axis=1)
        resultBlock = labels * wxi - np.log(1 + np.exp(wxi))
        assert len(resultBlock.shape)==1
        result = np.sum(resultBlock)
        return -1 * result

    def g(self,w,features,labels):
        # refer to the note on page 117
        YiXij = labels.reshape(-1, 1) * features
        WjXij = w * features
        expWXi = np.exp(np.sum(WjXij, axis=1)).reshape(-1, 1)
        rightSide = (features * expWXi) / (1 + expWXi)
        beforeSummation = rightSide - YiXij  # for finding minimum, need to minus the result
        result = np.sum(beforeSummation, axis=0)
        return result



    def train(self, features, labels, *args, **dicts):
        learningRate=float(loadConfigWithName("LogisticConfig","learningRate"))
        batchSize=int(loadConfigWithName("LogisticConfig","batchSize"))
        batchNum=features.shape[0]//batchSize+1 if features.shape[0]%batchSize!=0 else features.shape[0]//batchSize
        threshhold=float(loadConfigWithName("LogisticConfig","threshold"))
        self.useNewton=int(loadConfigWithName("LogisticConfig","useNewton"))

        #SGD
        if self.useNewton!=1:
            w=np.random.rand(features.shape[1])
            while True:
                if np.linalg.norm(self.g(w,features,labels))<threshhold:
                    self.save(w.tolist())
                    return
                for batchIndex in range(batchNum):
                    startIndex=batchIndex*batchSize
                    endIndex=startIndex+batchSize
                    batchFeatures=features[startIndex:endIndex,:]
                    batchLabel=labels[startIndex:endIndex]
                    grad=self.g(w,batchFeatures,batchLabel)

                    w=w-learningRate*grad

        #quasi-Newton(may have numerical problems)
        else:
            def f(w):
                #assert len(w.shape) == 1
                wxi = np.sum(w * features, axis=1)
                resultBlock = labels * wxi - np.log(1 + np.exp(wxi))
                #assert len(resultBlock.shape) == 1
                result = np.sum(resultBlock)
                return -1 * result

            def g(w):
                # refer to the note on page 117
                YiXij = labels.reshape(-1, 1) * features
                #assert len(w.shape) == 1
                WjXij = w * features
                expWXi = np.exp(np.sum(WjXij, axis=1)).reshape(-1, 1)
                rightSide = (features * expWXi) / (1 + expWXi)
                beforeSummation = rightSide - YiXij  # for finding minimum, need to minus the result
                result = np.sum(beforeSummation, axis=0)
                return result

            optimizedW = BFGSAlgo(f, g, features.shape[1])
            self.save(optimizedW.tolist())



    def predict(self, features):
        self.loadPara()
        predictResult = []
        for aFeature in features:
            expWX = np.exp(np.sum(self.w * aFeature))
            pYEquals1 = expWX / (1 + expWX)
            predictResult.append(1) if pYEquals1 >= 0.5 else predictResult.append(0)
        return np.array(predictResult)

    def loadPara(self):
        self.w = np.array(self.loadJson())
