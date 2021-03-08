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
class MaximumEntropy(ModelBaseClass):
    def __init__(self, threshold=None):
        if threshold == None:
            self.fFuncThreshold = float(loadConfigWithName("MaximumEntropyConfig", "featureFunctionThreshold"))
        else:
            self.fFuncThreshold = threshold
        if self.fFuncThreshold <= 0.5:
            raise Exception("threshold too low")
        self.featureFunc=None
        self.labelNum=None
        self.subFeatureNum=None
    
    def Pwyx(self,w,xiValue,y,column):
        numerator=np.exp(w[self.getWeightIndex(y,column)]) if self.featureFunc[y,column]==xiValue else 1
        denominator=0
        for eachY in range(self.labelNum):
            denominator+=np.exp(w[self.getWeightIndex(eachY,column)]) if self.featureFunc[eachY,column]==xiValue else 1
        return numerator/denominator

    def getWeightIndex(self,label, column):
        return label * self.subFeatureNum + column

    def train(self, features: np.array, labels: np.array, *args, **dicts):
        self.labelNum=np.unique(labels).shape[0]
        self.subFeatureNum=features.shape[1]
        featureFunc = -1 * np.ones((len(np.unique(labels)), features.shape[1]))
        # ffCount = len(featureFunc) - 1

        # P(xi, y) laebl,xi.
        # To save memory, only the possibilities of those (xi,y) who correspond
        # to at least one feature function are computed.
        pxiY = np.zeros((self.labelNum,features.shape[1],2))
        # construct feature functions
        for aLabel in np.unique(labels):
            labelFilteredFeature = np.squeeze(features[np.where(labels == aLabel), :])
            for column in range(labelFilteredFeature.shape[1]):
                oneNum = np.sum(labelFilteredFeature[:, column])
                onePercent = oneNum / labelFilteredFeature.shape[0]
                zeroPercent = 1 - onePercent
                pxiY[aLabel,column,1] = onePercent
                if onePercent >= self.fFuncThreshold:
                    featureFunc[aLabel, column] = 1
                elif zeroPercent >= self.fFuncThreshold:
                    featureFunc[aLabel, column] = 0
        pxiY[:, :, 0] = 1 - pxiY[:,:,1]
        self.featureFunc=featureFunc

        pxiY = np.array(pxiY)

        # compute P(xi)
        # value, xi
        onePosibility = np.sum(features, axis=0) / features.shape[0]
        zeroPosibility = 1 - onePosibility
        pxi = np.concatenate((zeroPosibility.reshape((1,-1)), onePosibility.reshape((1,-1))), axis=0)
        

        def f(w):
            fw=0
            for xiValue in range(2):
                for column in range(features.shape[1]):
                    sigmaExp=0
                    rightPart = 0
                    for y in range(self.labelNum):
                        if featureFunc[y,column]==xiValue:
                            sigmaExp+=np.exp(w[self.getWeightIndex(y,column)])
                            rightPart+=pxiY[y,column,xiValue]*w[self.getWeightIndex(y,column)]
                        else:
                            sigmaExp+=1
                    PxiLogSigma=pxi[xiValue,column]*np.log(sigmaExp)
                    fw+=PxiLogSigma-rightPart
            print(fw)
            return fw


        def g(w):
            featureFuctLine=self.featureFunc.reshape((-1,))
            result=np.zeros(w.shape)
            for i in range(pxiY.shape[0]):
                for j in range(pxiY.shape[1]):
                    if featureFunc[i,j]!=-1:
                        result[self.getWeightIndex(i,j)]=pxiY[i,j,int(featureFunc[i,j])]

            leftPart=0
            for xiValue in range(2):
                for column in range(features.shape[1]):
                    for y in range(self.labelNum):
                        if featureFunc[y,column]==xiValue:
                            leftPart+=pxi[xiValue,column]*self.Pwyx(w,xiValue,y,column)
            result[np.where(featureFuctLine!=-1)]+=leftPart
            return result

        optimizeW=BFGSAlgo(f,g,self.labelNum*self.subFeatureNum)
        para={}
        para["w"]=optimizeW.tolist()
        para["featureFunc"]=featureFunc.tolist()
        self.save(para)

    def save(self, para):
        super().save(para)

    def predict(self, features):
        self.loadPara()
        result=[]
        for aFeature in features:
            for y in range(self.featureFunc.shape[0]):
                pass


    def loadPara(self):
        para=self.loadJson()
        self.w=para["w"]
        self.featureFunc=para["featureFunc"]


