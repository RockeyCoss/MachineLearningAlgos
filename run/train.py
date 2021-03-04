import numpy as np
import importlib
from utilities import loadMainConfig,loadData
#import for reflection
from models import *

#trainModel
def train():
    features, labels = loadData(loadMainConfig("modelName"),"train")
    module=importlib.import_module("models")
    modelClass=getattr(module,loadMainConfig("modelName"))
    model=modelClass()
    model.train(features, labels)

if __name__=="__main__":
    train()



