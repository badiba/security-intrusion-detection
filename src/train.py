import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import pandas as pd
import random


class SVM:
    def __init__(self):
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

        evilTrain = evils.iloc[:100, :]
        goodTrain = goods.iloc[:100, :]

        evilTests = evils.iloc[100:200, :]
        goodTests = goods.iloc[100:200, :]
        testData = pd.concat([evilTests, goodTests])

        dataset = shuffle(pd.concat([evilTrain, goodTrain]))

        data = dataset.iloc[:, :-1]
        labels = dataset.iloc[:, -1:]
        svc = SVC(kernel='linear', C=0.1)
        svc.fit(data, labels.values.ravel())

        predictions = svc.predict(testData.iloc[:, :-1])

        correctPredictionCount = 0
        totalPredictionCount = 0

        for index, row in testData.iterrows():
            totalPredictionCount += 1

            if (row[-1] == predictions[totalPredictionCount - 1]):
                correctPredictionCount += 1

        print(correctPredictionCount)
        print(totalPredictionCount)


def main():
    svm = SVM()


if __name__ == "__main__":
    main()
