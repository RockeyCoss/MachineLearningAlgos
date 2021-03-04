import numpy as np
import json
from models import ModelBaseClass

class Logistic(ModelBaseClass):

    def train(self, features, labels, *args, **dicts):
        super().train(features, labels, *args, **dicts)

    def predict(self, features):
        super().predict(features)

    def save(self, para):
        super().save(para)

    def load(self):
        super().load()