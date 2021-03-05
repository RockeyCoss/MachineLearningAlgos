import numpy as np
from models import ModelBaseClass
from utilities import BFGSAlgo


# Remember that for logistic model, the label yi is either 0 or 1, not -1 or 1!
class Logistic(ModelBaseClass):
    def __init__(self):
        self.w = None

    def train(self, features, labels, *args, **dicts):
        def f(w):
            wxi = np.sum(w.reshape(-1, ) * features, axis=1)
            resultBlock = labels * wxi - np.log(1 + np.exp(wxi))
            assert len(resultBlock.shape)==1
            result = np.sum(resultBlock)
            return -1 * result

        def g(w):
            # refer to the note on page 117
            YiXij = labels.reshape(-1, 1) * features
            WjXij = w * features
            expWXi = np.exp(np.sum(WjXij, axis=1)).reshape(-1, 1)
            rightSide = (features * expWXi) / (1 + expWXi)
            beforeSummation = rightSide - YiXij  # for finding minimum, need to minus the result
            result = np.sum(beforeSummation, axis=0)
            #由于早期错误严重，梯度可能会非常大，因此需要截断
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
