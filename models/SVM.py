import numpy as np
import json
import sklearn
from models import ModelBaseClass
# refer to Statistical Learning Methods and
# paper Sequential Minimal Optimization:A Fast Algorithm for Training Support Vector Machines
# optimized the program according to a zhihu answer:
# https://www.zhihu.com/question/31211585/answer/640501555?utm_source=qq&utm_medium=social&utm_oi=710101447663972352

class SVM(ModelBaseClass):
    def __init__(self):
        self.rbfGamma:float = 0
        self.sampleNum:int = 0
        self.C:float = 1
        self.epsilon:float = 1e-3
        self.alhpa:np.ndarray=None
        self.b:float=None
        self.features:np.ndarray=None
        self.labels:np.ndarray=None
        self.kernelMatrix:np.ndarray=None
        self.alphaNot0NorC:list=None

    def rbf(self, x1, x2):
        """
        以sklearn “auto” 方式确定gamma
        """
        return np.exp(-np.sum(np.power((x1 - x2), 2)) / self.rbfGamma)

    def getKernelMatrix(self, kernel) -> np.ndarray:
        """
        计算K(xi,xj)矩阵
        :param kernel: 核函数
        :return: K(xi,xj)矩阵。考虑到训练集中样例较少，因此matrix暂不用cache方式而是整体储存
        """
        kernelMatrix = np.zeros((self.sampleNum, self.sampleNum))
        for i in range(self.sampleNum):
            for j in range(i + 1):
                kernelMatrix[i, j] = kernelMatrix[j, i] = kernel(self.features[i], self.features[j])
        return kernelMatrix

    def train(self, features: np.ndarray, labels: np.ndarray, *args, **dicts):
        self.rbfGamma = features.shape[1] if features.shape[1] > 0 else 1
        self.sampleNum = features.shape[0]
        self.C = dicts['C'] if 'C' in dicts.keys() else 1
        self.alhpa=np.zeros(self.sampleNum)
        self.b=0
        self.features=features
        self.labels=labels
        self.kernelMatrix=self.getKernelMatrix()
        self.alphaNot0NorC=[]
        # for every non-bound example, error cache stores it's Ei
        # for other examples, error cache stores
        # 两种方案，这里采取了空间换时间的方案
        # 省空间方案：在errorCache里仅存（index,Ei),并且errorCache按index排序
        # 这样插入，删除和修改都要logn，但是空间能节省（前期没节省多少，后期还是可的）
        # 省时间方案：errorCache和sampleNum登长，0表示非non-bound example(non-bound example不可能有g(xi)-yi=0)，
        # 这里采用省时间方案
        self.errorCache=(np.sum((self.alhpa*self.labels*self.kernelMatrix),axis=1)+self.b)-self.labels
        

    def predict(self, features: np.array):
        super().predict(features)

        sklearn.svm()

    def loadPara(self):
        super().loadPara()

    def __trainer(self):
        examineAll:bool=True
        changedNum=0
        while changedNum>0 or examineAll==True:
            changedNum=0
            if examineAll:
                for sampleIndex in range(self.sampleNum):
                    changedNum+=self.__exaime(sampleIndex)
            else:
                for listIndex,sampleIndex in enumerate(self.alphaNot0NorC):
                    changedNum+=self.__examine(sampleIndex, listIndex)

            examineAll=False if examineAll else True

    def __examine(self, sampleIndex:int, listIndex:int=-1)->int:
        y2=self.labels[sampleIndex]
        alpha2=self.alhpa[sampleIndex]
        E2=self.errorCache[sampleIndex]
        r2=E2*y2
        # KKT condition;
        # alphaI==0<=>yi(W*Xi+b)>=1<=>yi(W*Xi+b-yi)>=0
        # 0<alphaI<C<=>yi(W*Xi+b)==1<=>yi(W*Xi+b-yi)==0
        # alphaI==C<=>yi(W*Xi+b)><=1<=>yi(W*Xi+b-yi)<=0
        # examine whether the KKT conditions are satisfied:
        if (r2<-self.epsilon and alpha2<self.C) or (r2>self.epsilon and alpha2>0):
            #not satisfied

        else:
            #satisfied
            return 0