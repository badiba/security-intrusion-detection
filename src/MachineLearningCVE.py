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
import threading
from threading import Thread
import time
import inspect
from multiprocessing.connection import Listener

def PrintSeparator():
    print("----------------")

def PrintBlank():
    print("                ")

lock = threading.Lock()

class MachineLearningCVE:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(dirname, "dataset", "MachineLearningCVE", "Wednesday-workingHours.pcap_ISCX.csv")
        self._dataset = Dataset(filepath, " Label", "BENIGN", "BENIGN", "EVIL", 100)

    def GetTestAccuracy(self, predictions):
        correctPredictions = 0
        trueNegative = 0
        truePositive = 0
        falseNegative = 0
        falsePositive = 0

        for i in range(len(predictions)):
            if (predictions[i] == self._dataset._testLabels.iloc[i][' Label']):
                if (predictions[i] == "BENIGN"):
                    trueNegative += 1
                else:
                    truePositive += 1

                correctPredictions += 1

            else:
                if (predictions[i] == "BENIGN"):
                    falseNegative += 1
                else:
                    falsePositive += 1

        accuracy = correctPredictions / len(predictions)
        return accuracy, trueNegative, truePositive, falseNegative, falsePositive

    def PrintConfusionMatrix(self):
        predictions = self._model.predict(self._dataset._testData)
        acc, tn, tp, fn, fp = self.GetTestAccuracy(predictions)
        precision = tp / float(tp + fp)
        specificity = tn / float(tn + fp)
        recall = tp / float(tp + fn)

        print("True Negative: " + str(tn))
        print("True Positive: " + str(tp))
        print("False Negative: " + str(fn))
        print("False Positive: " + str(fp))
        print("Specificity: {0:0.2f}".format(specificity))
        print("Recall: {0:0.2f}".format(recall))
        print("Precision: {0:0.2f}".format(precision))

    def Optimize(self, parameters):
        clf = GridSearchCV(self._model, parameters)
        clf.fit(self._dataset._trainData, self._dataset._trainLabels.values.ravel())
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        grid = pd.concat([pd.DataFrame(clf.cv_results_["params"]), pd.DataFrame(
            clf.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)

        print(grid)
        PrintSeparator()
        print("Parameters with max accuracy:")
        print(grid.loc[grid['Accuracy'].idxmax()])

    def OptimizePassiveAggressive(self):
        self._model = PassiveAggressiveClassifier(early_stopping=False, max_iter=1000)

        cValues = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        tolValues = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        lossValues = ["hinge", "squared_hinge"]

        parameters = {'C': cValues, 'loss': lossValues, 'tol': tolValues}
        self.Optimize(parameters)

    def OptimizePerceptron(self):
        self._model = Perceptron()

        penaltyValues = [None, "l1", "l2"]
        alphaValues = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        etaValues = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

        parameters = {'penalty': penaltyValues, 'alpha': alphaValues, 'eta0': etaValues}
        self.Optimize(parameters)


    def TrainPassiveAggressive(self, cValue):
        Debug.BeginScope("Train")

        self._model = PassiveAggressiveClassifier(early_stopping=False, max_iter=1000, C=cValue, loss="hinge", tol=0.00001)
        self._model.fit(self._dataset._trainData, self._dataset._trainLabels.values.ravel())

        Debug.EndScope()

    def TrainPerceptron(self):
        Debug.BeginScope("Train")

        self._model = Perceptron(alpha=0.001, eta0=0.05, penalty='l2')
        self._model.fit(self._dataset._trainData, self._dataset._trainLabels.values.ravel())

        Debug.EndScope()

    def Test(self, printDirectly = True):
        Debug.BeginScope("Test")

        score = self._model.score(self._dataset._testData, self._dataset._testLabels)

        if (printDirectly):
            print("Accuracy: {0:0.2f}".format(score))

        Debug.EndScope()
        return score

    def PartialFitAll(self):
        self._model.partial_fit(self._dataset._humanData, self._dataset._humanLabels.values.ravel(), ["BENIGN", "EVIL"])

    def PartialFit(self, example, label):
        self._model.partial_fit(example, label, ["BENIGN", "EVIL"])

    def PartialPredict(self, example):
        return self._model.predict(example)


def GetOptimizationResult():
    model = MachineLearningCVE()
    PrintSeparator()
    model.OptimizePassiveAggressive()

def AutoSimulate():
    model = MachineLearningCVE()
    PrintSeparator()

    model.TrainPassiveAggressive(0.01)
    model.Test()
    model.PrintConfusionMatrix()
    PrintSeparator()

    model.PartialFitAll()
    model.Test()
    model.PrintConfusionMatrix()


def SimulatePacketReceive(model):
    index = 0

    while (True):
        with lock:
            example = model._dataset._humanData.iloc[[index]]
            prediction = model.PartialPredict(example)
            label = model._dataset._humanLabels.iloc[[index]].values.ravel()
            isCorrect = prediction[0] == label[0]
            
            print("Id: {} - {} - Prediction: {} - Actual: {}".format(index, isCorrect, prediction, label))

            index += 1

        time.sleep(1)


def ListenUserCommands(model):
    address = ('localhost', 6000)
    listener = Listener(address, authkey=b'secret password')
    conn = listener.accept()

    while (True):
        msg = conn.recv()

        with lock:
            if (msg == "acc"):
                acc = model.Test(False)
                conn.send("Accuracy: {}".format(acc))
            elif (msg.split()[0] == "w" and len(msg.split()) == 2):
                id = msg.split()[1]
                example = model._dataset._humanData.iloc[[id]]
                label = model._dataset._humanLabels.iloc[[id]].values.ravel()
                model.PartialFit(example, label)
                conn.send("Using the packet with id {} for learning".format(id))


def ManualSimulate():
    model = MachineLearningCVE()
    PrintSeparator()

    model.TrainPassiveAggressive(0.0001)
    model.Test()
    #model.PrintConfusionMatrix()
    PrintSeparator()

    print("Entered manual simulation")
    PrintBlank()

    packetReceiveThread = Thread(target=SimulatePacketReceive, args=(model,))
    userInputThread = Thread(target=ListenUserCommands, args=(model,))

    packetReceiveThread.start()
    userInputThread.start()
    

def main():
    Debug.DisableDebug()
    Debug.BeginScope("Main")

    ManualSimulate()

    Debug.EndScope()


if __name__ == "__main__":
    main()
