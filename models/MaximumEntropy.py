import numpy as np
import collections
from models import ModelBaseClass
from utilities import BFGSAlgo, loadConfigWithName


# 实现细节：
# 经过查阅各种资料和对他人实现的参考，这里实现的（不一定正确）f(x,y)中的x不是指整个feature vector，而是指faeture vector中
# 一个feature vector中每个维度的分量（否则每个人写同一个数字的方式都不一样，对于新的feature vector，训练集中没有出现过则永远不可能有
# f(x,y)=1)。同时，根据理解，f(x,y)=1代表的是一种对从训练集中估计的P(x,y)和实际数据的P(x,y)一样的信心。因此这里可以让出现频率达到一定阈值的
# (xi,y)的特征函数f(xi,y)=1。具体实现为：对于传入的（x,y)判断特征向量x的第i个分量是否有f(xi,y)=1同时，为了减少不同人书写力度对于数字灰度值
# 大小的影响，要先对图像进行二值化处理。
# 用矩阵存储f(xi,y),为一个矩阵，第一维维度对应的是第几个label,且最后一行保存着feature funciton总数，随后featureFunc[label]为一个向量，
# 默认为-1，若f(0,y)=1,则存储0，若f(1,y)=1,则储存1，这样可以方便后续向量点乘来计算wifi(xi,y)
# label:0,1,2,3,4,...
# feature:每一个像素中存储的是0或1
# threshold>0.5
# 训练集中应保证所有label至少出现过一次
# 使用拟牛顿法求解
# 求导式和书本不同，根据MNIST数据集重新求导(即把x改为xi再求导）

class MaximumEntropy(ModelBaseClass):
    def __init__(self, threshold=None):
        if threshold == None:
            self.fFuncThreshold = float(loadConfigWithName("MaximumEntropyConfig", "featureFunctionThreshold"))
        else:
            self.fFuncThreshold = threshold
        if self.fFuncThreshold <= 0.5:
            raise Exception("threshold too low")
        self.stopThreshold=float(loadConfigWithName("MaximumEntropyConfig", "stopThreshold"))
        # feature function 个数，也是w的维度数
        self.wDimension = 0
        self.w = None
        self.featureFunc = None
        self.labelNum = 0
        self.subFeatureNum = 0
        # self.wToffList[i]=The location of the feature function that corresponds to weight wi in featureFunc
        self.wToffHashTable = None
        self.ffTowHashTable = None
        self.Px = None
        self.Pxy = None

    def Pwyx(self, X, y):
        numerator = 0
        denominator = 0
        match = self.featureFunc == X
        for labelIndex in range(match.shape[0]):
            sigma = 0
            for index2 in np.squeeze(np.argwhere(match[labelIndex] == 1)):
                sigma += self.w[self.ffTowHashTable[self.ffHash((labelIndex, index2))]]
            expSigma = np.exp(sigma)
            if labelIndex == y:
                numerator += expSigma
            denominator += expSigma
        return numerator / denominator

    def ffHash(self, coordinate):
        return (self.subFeatureNum) * coordinate[0] + coordinate[1]

    def f(self, w):
        left = 0
        right = 0
        for column in range(self.subFeatureNum):
            for xiValue in range(2):
                sigmaY = 0
                for y in range(self.labelNum):
                    sigma = 0
                    if self.featureFunc[y, column] == xiValue:
                        sigma += w[self.ffTowHashTable[self.ffHash((y, column))]]
                        right += w[self.ffTowHashTable[self.ffHash((y, column))]] * self.Pxy[y, column, xiValue]
                    sigmaY += np.exp(sigma)
                left += self.Px[xiValue, column] * np.log(sigmaY)
        return left - right

    def g(self, w):
        result = np.zeros(w.shape)
        for index in range(w.shape[0]):
            label, column = self.wToffHashTable[index]
            xiValue = self.featureFunc[label, column]
            denominator = 0
            for otherLabel in range(self.labelNum):
                if self.featureFunc[otherLabel, column] == xiValue:
                    denominator += np.exp(w[self.ffTowHashTable[self.ffHash((otherLabel, column))]])
                else:
                    denominator += 1
            result[index] = self.Px[xiValue, column] * (np.exp(w[index]) / denominator) - self.Pxy[
                label, column, xiValue]
        return result

    def train(self, features: np.array, labels: np.array, *args, **dicts):
        self.labelNum = np.unique(labels).shape[0]
        self.subFeatureNum = features.shape[1]
        featureFunc = -1 * np.ones((self.labelNum, self.subFeatureNum), dtype=int)
        self.Pxy = np.zeros((self.labelNum, self.subFeatureNum, 2))
        for aLabel in range(self.labelNum):
            labelFilteredFeature = np.squeeze(features[np.where(labels == aLabel), :])
            oneSum=np.sum(labelFilteredFeature, axis=0)
            oneProbability = oneSum / labelFilteredFeature.shape[0]
            zeroProbability = 1 - oneProbability
            featureFunc[aLabel, np.where(oneProbability >= self.fFuncThreshold)] = 1
            featureFunc[aLabel, np.where(zeroProbability >= self.fFuncThreshold)] = 0
            self.Pxy[aLabel, :, 1] = oneSum
            self.Pxy[aLabel, :, 0] = labelFilteredFeature.shape[0]-self.Pxy[aLabel,:,1]
        validPosition = np.argwhere(featureFunc != -1)
        self.featureFunc = featureFunc
        self.wDimension = validPosition.shape[0]
        self.wToffHashTable = validPosition
        self.Pxy = self.Pxy / features.shape[0]
        # w与featureFunc坐标用哈希表映射
        self.ffTowHashTable = np.array([-1 for dummy in range(self.labelNum * self.subFeatureNum)])
        for coordinateIndex in range(self.wToffHashTable.shape[0]):
            self.ffTowHashTable[self.ffHash(self.wToffHashTable[coordinateIndex])] = coordinateIndex

        oneP = np.sum(features, axis=0) / features.shape[0]
        zeroP = 1 - oneP
        self.Px = np.concatenate((zeroP.reshape((1, -1)), oneP.reshape((1, -1))), axis=0)

        optimizeW = BFGSAlgo(self.f, self.g, self.wDimension,self.stopThreshold)


        para = {}
        para["w"] = optimizeW.tolist()
        para["featureFunc"] = self.featureFunc.tolist()
        para["subFeatureNum"] = self.subFeatureNum
        para["ffTowHash"] = self.ffTowHashTable.tolist()
        para["labelNum"]=self.labelNum
        self.save(para)

    def ffHashLargeSize(self, coordinate):
        return (self.subFeatureNum) * coordinate[:,0] + coordinate[:,1]

    def save(self, para):
        super().save(para)

    def predict(self, features):
        self.loadPara()
        result=[]
        for aFeature in features:
            pwyx=np.zeros(self.labelNum)
            for label in range(self.labelNum):
                pwyx[label]=self.Pwyx(aFeature,label)
            result.append(np.argmax(pwyx))
        return np.array(result)

    def loadPara(self):
        para=self.loadJson()
        self.w=np.array(para["w"])
        self.featureFunc=np.array(para["featureFunc"])
        self.subFeatureNum=para["subFeatureNum"]
        self.ffTowHashTable=np.array(para["ffTowHash"])
        self.labelNum=para["labelNum"]

