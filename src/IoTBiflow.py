from logging import debug
import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.utils import shuffle
import pandas as pd
import random
import Debug
from sklearn import preprocessing

TRAIN_SAMPLE_COUNT = 2100


class IoTBiflow:
    def __init__(self):
        Debug.BeginScope("Data preprocessing")

        dirname = os.path.dirname(os.path.abspath(__file__))
        dataset = pd.read_csv(os.path.join(
            dirname, "dataset", "IoTBiflow", "biflow_mqtt_bruteforce.csv"))

        # Clear dirty rows.
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset = dataset.dropna()
        dataset = dataset.iloc[:, 2:]

        # Get good and evil labeled examples separately.
        evils = dataset[dataset["is_attack"] == 1]
        goods = dataset[dataset["is_attack"] == 0]

        # Randomize example order.
        evils = shuffle(evils)
        goods = shuffle(goods)

        # Prepare train data.
        evilTrain = evils.iloc[:TRAIN_SAMPLE_COUNT, :]
        goodTrain = goods.iloc[:TRAIN_SAMPLE_COUNT, :]
        trainRawData = pd.concat([evilTrain, goodTrain])
        trainRawData = shuffle(trainRawData)
        self._trainData = trainRawData.iloc[:, :-1]
        self._trainLabels = trainRawData.iloc[:, -1:]

        # Normalize train data.
        xTrain = self._trainData.values
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        xTrain_scaled = min_max_scaler.fit_transform(xTrain)
        self._trainData = pd.DataFrame(
            xTrain_scaled, columns=self._trainData.columns)

        # Prepare test data.
        evilTests = evils.iloc[TRAIN_SAMPLE_COUNT:, :]
        goodTests = goods.iloc[TRAIN_SAMPLE_COUNT:, :]
        testRawData = pd.concat([evilTests, goodTests])
        testRawData = shuffle(testRawData)
        self._testData = testRawData.iloc[:, :-1]
        self._testLabels = testRawData.iloc[:, -1:]

        # Normalize test data.
        xTest = self._testData.values
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        xTest_scaled = min_max_scaler.fit_transform(xTest)
        self._testData = pd.DataFrame(
            xTest_scaled, columns=self._testData.columns)

        # Create model instance.
        self._pac = PassiveAggressiveClassifier()

        Debug.EndScope()

    def Train(self):
        Debug.BeginScope("Train")

        self._pac.fit(self._trainData, self._trainLabels.values.ravel())

        Debug.EndScope()

    def Test(self):
        Debug.BeginScope("Test")

        score = self._pac.score(self._testData, self._testLabels)
        print("Score: " + str(score))

        Debug.EndScope()


def main():
    model = IoTBiflow()
    model.Train()
    model.Test()


if __name__ == "__main__":
    Debug.EnableDebug()
    Debug.BeginScope("Main")
    main()
    Debug.EndScope()
