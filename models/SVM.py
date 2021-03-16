import numpy as np
import json
import sklearn
from models import ModelBaseClass
# refer to Statistical Learning Methods and
# paper Sequential Minimal Optimization:A Fast Algorithm for Training Support Vector Machines
# optimized the program according to a zhihu answer:
# https://www.zhihu.com/question/31211585/answer/640501555?utm_source=qq&utm_medium=social&utm_oi=710101447663972352
# non-bound examples: examples with alphas that are neither 0 nor C
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
        #hot data
        self.alphaNot0NorC=[]
        #the actural time comsuming computation
        # Ei=wx[i]+b-yi
        self.wx=np.zeros(self.sampleNum)



    def predict(self, features: np.array):
        super().predict(features)

    def loadPara(self):
        super().loadPara()

    def __trainer(self):
        examineAll:bool=True
        changedNum=0
        while changedNum>0 or examineAll==True:
            changedNum=0
            if examineAll:
                for sampleIndex in range(self.sampleNum):
                    changedNum+=self.__examime(sampleIndex)
            else:
                for listIndex,sampleIndex in enumerate(self.alphaNot0NorC):
                    changedNum+=self.__examine(sampleIndex, listIndex)

            examineAll=False if examineAll else True

    def __getI1Heuristically(self,E2):
        if E2>=0:
            return np.argmax(self.wx+self.b-self.labels)

    def __examine(self, sampleIndex:int, listIndex:int=-1)->int:
        y2=self.labels[sampleIndex]
        alpha2=self.alhpa[sampleIndex]
        E2=self.wx[sampleIndex]+self.b-self.labels[sampleIndex]
        r2=E2*y2
        # KKT condition;
        # alphaI==0<=>yi(W*Xi+b)>=1<=>yi(W*Xi+b-yi)>=0
        # 0<alphaI<C<=>yi(W*Xi+b)==1<=>yi(W*Xi+b-yi)==0
        # alphaI==C<=>yi(W*Xi+b)><=1<=>yi(W*Xi+b-yi)<=0
        # examine whether the KKT conditions are satisfied:
        if (r2<-self.epsilon and alpha2<self.C) or (r2>self.epsilon and alpha2>0):
            #not satisfied
            alphaNot0NorClength=len(self.alphaNot0NorC)
            if alphaNot0NorClength>1:
                alpha1Index=-1
                # because of the list of alphaNot0NorC, the program doesn't need to spend time computing
                # bound examples' E
                if E2>=0:
                    indexOfAlpha1Index=np.argmax((self.wx[self.alphaNot0NorC]+self.b-self.labels[self.alphaNot0NorC]))
                else:
                    indexOfAlpha1Index=np.argmin((self.wx[self.alphaNot0NorC]+self.b-self.labels[self.alphaNot0NorC]))

                alpha1Index = self.alphaNot0NorC[indexOfAlpha1Index]
                assert alpha1Index!=-1

                if self.__optimize(alpha1Index,sampleIndex):
                    return 1

            #SMO can't make positive progress using alpha1Index
            start=np.random.randint(low=0,high=alphaNot0NorClength)
            for delta in range(0,alphaNot0NorClength):
                indexOfAnotherIndex=(start+delta)%alphaNot0NorClength
                if indexOfAnotherIndex==indexOfAlpha1Index:
                    continue
                anotherIndex=self.alphaNot0NorC[indexOfAnotherIndex]
                if self.__optimize(anotherIndex,sampleIndex):
                    return 1

            # desperately search
            boundExamples=list(set(i for i in range(self.sampleNum))-set(self.alphaNot0NorC))
            start=np.random.randint(low=0,high=len(boundExamples))
            for delta in range(0,len(boundExamples)):
                indexOfDesperateIndex=(start+delta)%len(boundExamples)
                desperateIndex=boundExamples[indexOfDesperateIndex]
                if self.__optimize(desperateIndex,sampleIndex):
                    return 1
            # give up optimizing sampleIndex
            return 0
        else:
            #satisfied
            return 0

        def __optimize(self,alpha1Index:int,alpha2Index:int)->bool:
            return False