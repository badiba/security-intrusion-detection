from logging import debug
import numpy as np
import os
import sys
from scipy.sparse import data
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
    def __init__(self, path, labelColumnName, currentGoodLabelName, finalGoodLabelName, finalEvilLabelName, trainSetSize):
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

        # Balance the dataset.
        evils = dataset[dataset[labelColumnName] == finalEvilLabelName]
        goods = dataset[dataset[labelColumnName] == finalGoodLabelName]
        evilCount = len(evils.index)
        goodCount = len(goods.index)

        if (evilCount > goodCount):
            evils = evils.iloc[:goodCount, :]
        elif (goodCount > evilCount):
            goods = goods.iloc[:evilCount, :]

        dataset = pd.concat([evils, goods])
        dataset = shuffle(dataset)

        # Keep a portion of the dataset as human-in-the-loop examples.
        self._humanExampleSize = 200000
        self._humanExamples = dataset.iloc[-self._humanExampleSize:, :]
        dataset = dataset.iloc[:-self._humanExampleSize, :]

        # Prepare human-in-the-loop examples.
        self._humanData = self._humanExamples.iloc[:, :-1]
        self._humanLabels = self._humanExamples.iloc[:, -1:]
        xHuman = self._humanData.values
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        xHuman_scaled = min_max_scaler.fit_transform(xHuman)
        self._humanData = pd.DataFrame(
            xHuman_scaled, columns=self._humanData.columns)

        # Split train and test set.
        self._size = len(dataset.index)
        self._trainData = dataset.iloc[:trainSetSize, :-1]
        self._trainLabels = dataset.iloc[:trainSetSize, -1:]
        self._testData = dataset.iloc[trainSetSize:, :-1]
        self._testLabels = dataset.iloc[trainSetSize:, -1:]

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
        print("Train set size is {}".format(len(self._trainData.index)))
        print("Test set size is {}".format(len(self._testData.index)))
        print("Train set has {0:0.2f} percent benign (not attack) samples".format(trainGoodPercent))
        print("Test set has {0:0.2f} percent benign (not attack) samples".format(testGoodPercent))
        print("{} samples will be used for online learning".format(self._humanExampleSize))