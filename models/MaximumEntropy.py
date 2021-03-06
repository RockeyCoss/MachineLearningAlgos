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
# 用矩阵存储f(xi,y),为一个三维矩阵，第一维维度对应的是第几个label,且最后一行保存着feature funciton总数，随后featureFunc[label]为一个二维矩阵，
# 其第一列保存特征向量第几个分量，
# 第二列保存特征向量对应分量的值，第三列保存wi
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
        featureFunc = [0 for dummy in np.unique(labels)]
        featureFunc.append(0)
        ffCount = len(featureFunc) - 1

        # P(xi, y) laebl,xi.
        # To save memory, only the possibilities of those (xi,y) who correspond
        # to at least one feature function are computed.
        pxiY = []
        # construct feature functions
        for aLabel in np.sort(np.unique(labels)):
            pxiY.append([])
            featureFunc[aLabel] = []
            labelFilteredFeature = features[np.where(labels == aLabel), :]
            for column in range(labelFilteredFeature.shape[1]):
                oneNum = np.sum(labelFilteredFeature[:, column])
                onePercent = oneNum / labelFilteredFeature.shape[0]
                zeroPercent = 1 - onePercent
                pxiY[aLabel].append(-1)
                if onePercent >= self.fFuncThreshold:
                    featureFunc[aLabel].append([column, 1, 0])
                    featureFunc[ffCount] += 1
                    pxiY[aLabel][column] = np.sum(labelFilteredFeature[:, column]) / labelFilteredFeature.shape[0]

                elif zeroPercent >= self.fFuncThreshold:
                    featureFunc[aLabel].append([column, 0, 0])
                    featureFunc[ffCount] += 1
                    pxiY[aLabel][column] = 1 - np.sum(labelFilteredFeature[:, column]) / labelFilteredFeature.shape[0]

        pxiY=np.array(pxiY)

        # compute P(xi)
        # xi, value
        onePosibility = np.sum(features, axis=1) / features.shape[0]
        zeroPosibility = 1 - onePosibility
        pxi = np.array(list(zip(zeroPosibility, onePosibility)))

        def f(w):



    def save(self, para):
        super().save(para)

    def predict(self, features):
        pass

    def loadPara(self):
        pass
