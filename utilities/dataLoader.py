import cv2
import csv
import importlib
import numpy as np
from sklearn.model_selection import  train_test_split
from utilities import loadConfigWithName,loadMultipleConfigWithName
trainFeature=None
trainLabel=None
testFeature=None
testLabel=None
def loadData(modelName:str,mode:str):
    global trainLabel,trainFeature,testLabel,testFeature
    if mode=="test" and (testFeature!=None).any() and (testLabel!=None).any():
        return testFeature,testLabel
    features=None
    with open('../data/train.csv','r') as f:
        reader=np.array(list(csv.reader(f)))
        features=reader[1:901,1:].astype(np.uint8)
        labels=reader[1:901,0].astype(np.int)

    if (features==None).any():
        print("data reading error")
        return None


    module=importlib.import_module("utilities.transform")
    #transform
    labelTransform=loadMultipleConfigWithName(modelName+"Config","labelTransform")
    if labelTransform!=None:
        for oneLabelTransform in labelTransform:
            labelTransformMethod=getattr(module,oneLabelTransform)
            labels=labelTransformMethod(labels)

    featureTransform=loadMultipleConfigWithName(modelName+"Config","featureTransform")
    if featureTransform!=None:
        for oneFeatureTransform in featureTransform:
            featureTransformMethod = getattr(module, oneFeatureTransform)
            features = featureTransformMethod(features)


    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33)

    if mode=='train':
        trainFeature=train_features
        trainLabel=train_labels
        testFeature=test_features
        testLabel=test_labels
        return train_features,train_labels
    else:
        return test_features,test_labels

