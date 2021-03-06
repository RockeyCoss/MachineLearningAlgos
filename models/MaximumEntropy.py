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

    def train(self, features: np.array, labels: np.array, *args, **dicts):
        labelNum=np.unique(labels).shape[0]
        featureFunc = -1 * np.ones((len(np.unique(labels)), features.shape[1]))
        # ffCount = len(featureFunc) - 1

        # P(xi, y) laebl,xi.
        # To save memory, only the possibilities of those (xi,y) who correspond
        # to at least one feature function are computed.
        pxiY = np.zeros((labelNum,features.shape[1],2))
        # construct feature functions
        for aLabel in np.unique(labels):
            labelFilteredFeature = features[np.where(labels == aLabel), :]
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


        pxiY = np.array(pxiY)

        # compute P(xi)
        # value, xi
        onePosibility = np.sum(features, axis=1) / features.shape[0]
        zeroPosibility = 1 - onePosibility
        pxi = np.concatenate(zeroPosibility, onePosibility, axis=0)

        def f(w):
            w=w.reshape(labelNum,-1)
            fw=0
            for aFeature in features:
                #left part
                SigmaexpWiFi = np.sum(np.exp(np.sum(w * (aFeature == featureFunc), axis=1)), axis=0)
                logSigma = np.log(SigmaexpWiFi)
                PxlogSigma = np.prod(pxi[aFeature, np.arange(pxi.shape[1])])*logSigma

                #right part
                wifi=np.sum(w * (aFeature == featureFunc), axis=1)
                Pxy=np.prod(pxiY[:,np.arange(aFeature.shape[0]),aFeature],axis=1)
                pxyWiFi=np.sum(Pxy*wifi,axis=0)
                fw+=PxlogSigma-pxyWiFi
            return fw

        





    def save(self, para):
        super().save(para)

    def predict(self, features):
        pass

    def loadPara(self):
        pass
