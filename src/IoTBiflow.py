import numpy as np
import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import pandas as pd
import random
import Debug
from sklearn import preprocessing


class IoTBiflow:
    def __init__(self):
        Debug.BeginScope("Data preprocessing")

        dirname = os.path.dirname(os.path.abspath(__file__))
        dataset = pd.read_csv(os.path.join(
            dirname, "dataset", "IoTBiflow", "biflow_mqtt_bruteforce.csv"))

        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset = dataset.dropna()
        dataset = dataset.iloc[:, 2:]

        evils = dataset[dataset["is_attack"] == 1]
        goods = dataset[dataset["is_attack"] == 0]

        evils = shuffle(evils)
        goods = shuffle(goods)

        evilTrain = evils.iloc[:2152, :]
        goodTrain = goods.iloc[:2152, :]

        evilTests = evils.iloc[2152:4000, :]
        goodTests = goods.iloc[2152:2152, :]
        testData = pd.concat([evilTests, goodTests])

        dataset = shuffle(pd.concat([evilTrain, goodTrain]))

        data = dataset.iloc[:, :-1]
        labels = dataset.iloc[:, -1:]

        Debug.EndScope()
        Debug.BeginScope("Train")

        x = data.values
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_scaled = min_max_scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled, columns=data.columns)

        svc = SVC(kernel='linear', C=100)
        svc.fit(data, labels.values.ravel())

        Debug.EndScope()
        exit()
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

        print(100 * correctPredictionCount / float(totalPredictionCount))


def main():
    Debug.BeginScope("svm")
    svm = IoTBiflow()
    Debug.EndScope()


if __name__ == "__main__":
    main()
