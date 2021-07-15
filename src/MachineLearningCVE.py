from logging import debug
import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.utils import shuffle
import pandas as pd
import random
import Debug
from Dataset import Dataset
from sklearn import preprocessing


class MachineLearningCVE:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(dirname, "dataset", "MachineLearningCVE", "Wednesday-workingHours.pcap_ISCX.csv")
        self._dataset = Dataset(filepath, " Label", "BENIGN", "BENIGN", "EVIL")

    def Optimize(self):
        cValues = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        tolValues = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        lossValues = ["hinge", "squared_hinge"]
        max_iter = 50

        parameters = {'C': cValues, 'loss': lossValues, 'tol': tolValues}

        self._pac = PassiveAggressiveClassifier(early_stopping=True, max_iter=max_iter)
        clf = GridSearchCV(self._pac, parameters)
        clf.fit(self._dataset._trainData, self._dataset._trainLabels.values.ravel())
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(pd.concat([pd.DataFrame(clf.cv_results_["params"]), pd.DataFrame(
            clf.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1))


def main():
    Debug.EnableDebug()
    Debug.BeginScope("Main")
    svm = MachineLearningCVE()
    svm.Optimize()
    Debug.EndScope()


if __name__ == "__main__":
    main()
