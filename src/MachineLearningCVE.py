import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import pandas as pd
import random
import Debug


class MachineLearningCVE:
    def __init__(self):
        Debug.BeginScope("Data preprocessing")

        dirname = os.path.dirname(os.path.abspath(__file__))
        dataset = pd.read_csv(os.path.join(
            dirname, "dataset", "MachineLearningCVE", "Wednesday-workingHours.pcap_ISCX.csv"))

        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset = dataset.dropna()

        evilsRaw = dataset[dataset[" Label"] != "BENIGN"]
        goods = dataset[dataset[" Label"] == "BENIGN"]
        evils = evilsRaw.copy()
        evils[" Label"] = "EVIL"

        evils = shuffle(evils)
        goods = shuffle(goods)

        evilTrain = evils.iloc[:5, :]
        goodTrain = goods.iloc[:5, :]

        evilTests = evils.iloc[5:400, :]
        goodTests = goods.iloc[5:400, :]
        testData = pd.concat([evilTests, goodTests])

        dataset = shuffle(pd.concat([evilTrain, goodTrain]))

        data = dataset.iloc[:, :-1]
        labels = dataset.iloc[:, -1:]

        Debug.EndScope()
        Debug.BeginScope("Train")

        svc = SVC(kernel='linear', C=0.1)
        svc.fit(data, labels.values.ravel())

        Debug.EndScope()
        Debug.BeginScope("Test")

        predictions = svc.predict(testData.iloc[:, :-1])

        Debug.EndScope()
        Debug.BeginScope("Getting test results")

        correctPredictionCount = 0
        totalPredictionCount = 0

        for index, row in testData.iterrows():
            totalPredictionCount += 1

            if (row[-1] == predictions[totalPredictionCount - 1]):
                correctPredictionCount += 1

        Debug.EndScope()

        print(correctPredictionCount)
        print(totalPredictionCount)


def main():
    Debug.BeginScope("svm")
    svm = MachineLearningCVE()
    Debug.EndScope()


if __name__ == "__main__":
    main()
