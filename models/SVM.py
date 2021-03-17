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
        self.rbfGamma: float = 0
        self.sampleNum: int = 0
        self.C: float = 1
        self.epsilon: float = 1e-3
        self.alpha: np.ndarray = None
        self.b: float = None
        self.features: np.ndarray = None
        self.labels: np.ndarray = None
        self.wx: np.ndarray = None
        self.kernelMatrix: np.ndarray = None
        self.alphaNot0NorC: list = None
        self.eps:float = 1e-7
        self.alphay:np.ndarray=None
        self.supportVector:np.ndarray=None

    def rbf(self, x1, x2):
        """
        以sklearn “auto” 方式确定gamma
        """
        return np.exp(-np.sum(np.power((x1 - x2), 2)) / self.rbfGamma)

    def __getKernelMatrix(self, kernel) -> np.ndarray:
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
        self.alpha = np.zeros(self.sampleNum)
        self.b = 0
        self.features = features
        self.labels = labels
        self.kernelMatrix = self.__getKernelMatrix()
        # hot data
        self.alphaNot0NorC = []
        # the actual time consuming computation
        # Ei=wx[i]+b-y[i]
        self.wx = np.zeros(self.sampleNum)
        self.__trainer()
        para={}
        supportVectorIndex=np.where(self.alpha>0+self.eps)
        para["alpha*y"]=(self.alpha[supportVectorIndex]*self.labels[supportVectorIndex]).tolist()
        para["supportVector"]=self.features[supportVectorIndex].tolist()
        para["b"]=self.b
        self.save(para)

    def predict(self, features: np.array):
        self.loadPara()
        result = []
        for aFeature in features:
            kernel=np.exp(-np.sum(np.power((self.supportVector - aFeature), 2),axis=1) / self.rbfGamma)
            linear=np.sum(self.alphay*kernel)+self.b
            result.append(-1 if linear < 0 else 1)
        return np.array(result)

    def loadPara(self):
        para = self.loadJson()
        self.b=para["b"]
        self.alphay=np.array(para["alpha*y"])
        self.supportVector=np.array(para["supportVector"])
        self.rbfGamma = self.supportVector.shape[1] if self.supportVector.shape[1] > 0 else 1

    def __trainer(self):
        examineAll: bool = True
        changedNum = 0
        while changedNum > 0 or examineAll == True:
            changedNum = 0
            if examineAll:
                for sampleIndex in range(self.sampleNum):
                    changedNum += self.__examime(sampleIndex)
            else:
                for listIndex, sampleIndex in enumerate(self.alphaNot0NorC):
                    changedNum += self.__examine(sampleIndex, listIndex)

            examineAll = False if examineAll else True

    def __getI1Heuristically(self, E2):
        if E2 >= 0:
            return np.argmax(self.wx + self.b - self.labels)

    def __examine(self, sampleIndex: int, listIndex: int = -1) -> int:
        y2 = self.labels[sampleIndex]
        alpha2 = self.alpha[sampleIndex]
        E2 = self.wx[sampleIndex] + self.b - self.labels[sampleIndex]
        r2 = E2 * y2
        # KKT condition;
        # alphaI==0<=>yi(W*Xi+b)>=1<=>yi(W*Xi+b-yi)>=0
        # 0<alphaI<C<=>yi(W*Xi+b)==1<=>yi(W*Xi+b-yi)==0
        # alphaI==C<=>yi(W*Xi+b)><=1<=>yi(W*Xi+b-yi)<=0
        # examine whether the KKT conditions are satisfied:
        if (r2 < -self.epsilon and alpha2 < self.C) or (r2 > self.epsilon and alpha2 > 0):
            # not satisfied
            alphaNot0NorClength = len(self.alphaNot0NorC)
            if alphaNot0NorClength > 1:
                alpha1Index = -1
                # because of the list of alphaNot0NorC, the program doesn't need to spend time computing
                # bound examples' E
                if E2 >= 0:
                    indexOfAlpha1Index = np.argmax(
                        (self.wx[self.alphaNot0NorC] + self.b - self.labels[self.alphaNot0NorC]))
                else:
                    indexOfAlpha1Index = np.argmin(
                        (self.wx[self.alphaNot0NorC] + self.b - self.labels[self.alphaNot0NorC]))

                alpha1Index = self.alphaNot0NorC[indexOfAlpha1Index]
                assert alpha1Index != -1

                if self.__optimize(alpha1Index, sampleIndex):
                    return 1

            # SMO can't make positive progress using alpha1Index
            start = np.random.randint(low=0, high=alphaNot0NorClength)
            for delta in range(0, alphaNot0NorClength):
                indexOfAnotherIndex = (start + delta) % alphaNot0NorClength
                if indexOfAnotherIndex == indexOfAlpha1Index:
                    continue
                anotherIndex = self.alphaNot0NorC[indexOfAnotherIndex]
                if self.__optimize(anotherIndex, sampleIndex):
                    return 1

            # desperately search
            boundExamples = list(set(i for i in range(self.sampleNum)) - set(self.alphaNot0NorC))
            start = np.random.randint(low=0, high=len(boundExamples))
            for delta in range(0, len(boundExamples)):
                indexOfDesperateIndex = (start + delta) % len(boundExamples)
                desperateIndex = boundExamples[indexOfDesperateIndex]
                if self.__optimize(desperateIndex, sampleIndex):
                    return 1
            # give up optimizing alpha[sampleIndex]
            return 0
        else:
            # satisfied
            return 0

    def __optimize(self, alpha1Index: int, alpha2Index: int) -> bool:
        if alpha2Index == alpha1Index:
            return False
        alpha1 = self.alpha[alpha1Index]
        y1 = self.labels[alpha1Index]
        y2 = self.labels[alpha2Index]
        E1 = self.wx[alpha1Index] + self.b - self.labels[alpha1Index]
        E2 = self.wx[alpha2Index] + self.b - self.labels[alpha2Index]
        s = y1 * y2
        alpha2 = self.alpha[alpha2Index]
        # y1 equals to y2
        if s == 1:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        # y1 does not equal to y2
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)

        if L == H:
            return False

        eta = self.kernelMatrix[alpha1Index, alpha1Index] + self.kernelMatrix[alpha2Index, alpha2Index] - 2 * \
              self.kernelMatrix[alpha1Index, alpha2Index]

        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2)
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            K11 = self.kernelMatrix[alpha1Index, alpha1Index]
            K12 = self.kernelMatrix[alpha1Index, alpha2Index]
            K22 = self.kernelMatrix[alpha2Index, alpha2Index]

            f1 = y1 * (E1 + self.b) - alpha1 * K11 - s * alpha2 * K12
            f2 = y2 * (E2 + self.b) - s * alpha1 * K12 - alpha2 * K22
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            psiL = L1 * f1 + L * f2 + 0.5 * L1 * L1 * K11 + 0.5 * L * L * K22 + s * L * L1 * K12
            psiH = H1 * f1 + H * f2 + 0.5 * H1 * H1 * K11 + 0.5 * H * H * K22 + s * H * H1 * K12
            if psiL < psiH - self.eps:
                a2 = L
            elif psiL > psiH + self.eps:
                a2 = H
            else:
                a2 = alpha2

        if np.abs(a2 - alpha2) < self.eps * (alpha2 + a2 + self.eps):
            return 0

        a1 = alpha1 + s * (alpha2 - a2)

        K11 = self.kernelMatrix[alpha1Index, alpha1Index]
        K12 = self.kernelMatrix[alpha1Index, alpha2Index]
        K22 = self.kernelMatrix[alpha2Index, alpha2Index]
        b1 = E1 + y1 * (a1 - alpha1) * K11 + y2 * (a2 - alpha2) * K12 + self.b
        b2 = E2 + y1 * (a1 - alpha1) * K11 + y2 * (a2 - alpha2) * K22 + self.b
        if 0 < a1 < self.C:
            self.alphaNot0NorC.append(alpha1Index)
            self.b = b1
        if 0 < a2 < self.C:
            self.alphaNot0NorC.append(alpha2Index)
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        self.alpha[alpha1Index] = a1
        self.alpha[alpha2Index] = a2
        # to reduce computation, use derivative*delta to compute the increment of wx.
        self.wx += (a1 - alpha1) * self.kernelMatrix[alpha1Index] * self.labels[alpha1Index] + \
                   (a2 - alpha2) * self.kernelMatrix[alpha2Index] * self.labels[alpha2Index]

        return 1
