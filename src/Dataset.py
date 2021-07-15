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
from sklearn import preprocessing

class Dataset:
    def __init__(self, path, labelColumnName, currentGoodLabelName, finalGoodLabelName, finalEvilLabelName):
        Debug.BeginScope("Data preprocessing")

        self._goodLabelName = finalGoodLabelName
        self._evilLabelName = finalEvilLabelName
        self._labelColumnName = labelColumnName

        # Read from file to get the dataset.
        dataset = pd.read_csv(path)

        # Clear dirty rows.
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset = dataset.dropna()
        dataset = dataset.iloc[:, 2:] # use ip addresses

        # Convert label names.
        dataset.loc[dataset[labelColumnName] == currentGoodLabelName, labelColumnName] = finalGoodLabelName
        dataset.loc[dataset[labelColumnName] != currentGoodLabelName, labelColumnName] = finalEvilLabelName

        # Split train and test set.
        self._size = len(dataset.index)
        splitSize = int(self._size / 2)
        self._trainData = dataset.iloc[:splitSize, :-1]
        self._trainLabels = dataset.iloc[:splitSize, -1:]
        self._testData = dataset.iloc[splitSize:, :-1]
        self._testLabels = dataset.iloc[splitSize:, -1:]

        # Normalize train data.
        xTrain = self._trainData.values
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        xTrain_scaled = min_max_scaler.fit_transform(xTrain)
        self._trainData = pd.DataFrame(
            xTrain_scaled, columns=self._trainData.columns)

        # Normalize test data.
        xTest = self._testData.values
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        xTest_scaled = min_max_scaler.fit_transform(xTest)
        self._testData = pd.DataFrame(
            xTest_scaled, columns=self._testData.columns)

        self.InspectDataset()

        Debug.EndScope()

    def InspectDataset(self):
        trainGoods = self._trainLabels[self._trainLabels[self._labelColumnName] == self._goodLabelName]
        trainEvils = self._trainLabels[self._trainLabels[self._labelColumnName] == self._evilLabelName]
        trainGoodsCount = len(trainGoods.index)
        trainEvilsCount = len(trainEvils.index)
        trainGoodPercent = (trainGoodsCount / float(trainGoodsCount + trainEvilsCount)) * 100.0

        testGoods = self._testLabels[self._testLabels[self._labelColumnName] == self._goodLabelName]
        testEvils = self._testLabels[self._testLabels[self._labelColumnName] == self._evilLabelName]
        testGoodsCount = len(testGoods.index)
        testEvilsCount = len(testEvils.index)
        testGoodPercent = (testGoodsCount / float(testGoodsCount + testEvilsCount)) * 100.0

        print("Dataset size is {}".format(self._size))
        print("Train set has {} percent good samples".format(trainGoodPercent))
        print("Test set has {} percent good samples".format(testGoodPercent))