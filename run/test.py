import numpy as np
import importlib
from utilities import loadData, loadMainConfig

# validate
def test():
    features, labels = loadData(loadMainConfig("modelName"), "test")
    module = importlib.import_module("models")
    modelClass = getattr(module, loadMainConfig("modelName"))
    model = modelClass()
    predictResult = model.predict(features)

    sklearnMetricModule = importlib.import_module("sklearn.metrics")
    indicator = getattr(sklearnMetricModule, loadMainConfig("testIndicator"))
    testScore = indicator(predictResult, labels)
    print(f"{loadMainConfig('testIndicator')} is {testScore}")


if __name__ == "__main__":
    test()
