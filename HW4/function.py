import numpy as np
from math import log, exp
import matplotlib.pyplot as plt

def LR_CalcObj(XTrain, yTrain, wHat):
    vexp = np.vectorize(exp)
    vlog = np.vectorize(log)
    XTrain = np.insert(XTrain, 0, 1, axis = 1)
    zHat = np.dot(XTrain, wHat)
    logProb = yTrain * zHat - vlog(1 + vexp(zHat))
    return np.sum(logProb)


def LR_CalcGrad(XTrain, yTrain, wHat):
    vexp = np.vectorize(exp)
    XTrain = np.insert(XTrain, 0, 1, axis = 1)
    zHat = np.dot(XTrain, wHat)
    yProb1 = vexp(zHat) / (1 + vexp(zHat))
    yDiff = yTrain - yProb1
    grad = np.dot(XTrain.T, yDiff)
    return grad

def LR_UpdateParams(wHat, grad, eta):
    wHat = wHat + eta * grad
    return wHat

def LR_CheckConvg(oldObj, newObj, tol):
    return abs(oldObj - newObj) < tol

def LR_GradientAscent(XTrain, yTrain, eta = 0.01, tol = 0.001):
    oldObjVal = 99999999
    objVals = np.array([])
    wSize = XTrain.shape[1] + 1
    wHat = np.array([0] * wSize).reshape(-1,1)
    oldObjVal = LR_CalcObj(XTrain, yTrain, wHat)
    objVals = np.append(objVals, oldObjVal)
    while True:
        grad = LR_CalcGrad(XTrain, yTrain, wHat)
        wHat = LR_UpdateParams(wHat, grad, eta)
        newObjVal = LR_CalcObj(XTrain, yTrain, wHat)
        objVals = np.append(objVals, newObjVal)
        if LR_CheckConvg(oldObjVal, newObjVal, tol):
            break
        oldObjVal = newObjVal
    return wHat, objVals

def LR_PredictLabels(XTest, yTest, wHat):
    vexp = np.vectorize(exp)
    XTest = np.insert(XTest, 0, 1, axis = 1)
    zHat = np.dot(XTest, wHat)
    yProb1 = vexp(zHat) / (1 + vexp(zHat))
    yHat = np.array([1 if prob > 0.5 else 0 for prob in yProb1]).reshape(-1, 1)
    numErrors = np.sum(yHat != yTest)
    return yHat, numErrors


def plotObjVals(objVals):
    numIters = np.arange(1,len(objVals) + 1)
    plt.plot(numIters, objVals)
    plt.axis([0, 90, -350, -50])
    plt.xlabel('Number of iterations')
    plt.ylabel('Objective function value')
    plt.show()


def randperm(n, k):
    subsetInds = np.random.permutation(n)[:k]
    return subsetInds

def LR_TrainErrorVSTestError(XTrain, yTrain, XTest, yTest):
    trainFullSize = XTrain.shape[0]
    trainSizeSet = np.arange(10, trainFullSize + 1, 10)
    testSize = XTest.shape[0]
    errorTrainRates, errorTestRates = np.array([]), np.array([])
    for trainSize in trainSizeSet:
        subsetInds = np.random.permutation(trainFullSize)[:trainSize]
        XTrainSubset = XTrain[subsetInds,:]
        yTrainSubset = yTrain[subsetInds]
        wHat, _ = LR_GradientAscent(XTrainSubset, yTrainSubset)
        _, numTestErrors = LR_PredictLabels(XTest, yTest, wHat)
        _, numTrainErrors = LR_PredictLabels(XTrainSubset, yTrainSubset, wHat)
        errorTestRates = np.append(errorTestRates, numTestErrors / testSize)
        errorTrainRates = np.append(errorTrainRates, numTrainErrors / trainSize)
    return trainSizeSet, errorTestRates, errorTrainRates

def plotOnePredictionError(XTrain, yTrain, XTest, yTest):
    trainSizeSet, errorTestRates, errorTrainRates = LR_TrainErrorVSTestError(XTrain, yTrain, XTest, yTest)
    plt.plot(trainSizeSet, errorTrainRates, 'b', label = 'Training Error')
    plt.plot(trainSizeSet, errorTestRates, 'r', label = 'Testing Error')
    plt.xlabel('Training Set Size')
    plt.ylabel('Prediction Error')
    plt.legend()
    plt.show()

def plotAvgPredictionError(XTrain, yTrain, XTest, yTest, times = 10):
    time = 0
    avgErrorTestRates, avgErrorTrainRates = np.array([]), np.array([])
    while time < times:
        trainSizeSet, errorTestRates, errorTrainRates = LR_TrainErrorVSTestError(XTrain, yTrain, XTest, yTest)
        if len(avgErrorTestRates) == 0 and len(avgErrorTrainRates) == 0:
            avgErrorTestRates, avgErrorTrainRates = errorTestRates, errorTrainRates
        else:
            avgErrorTestRates += errorTestRates
            avgErrorTrainRates += errorTrainRates
        time += 1
    avgErrorTestRates /= times
    avgErrorTrainRates /= times
    plt.plot(trainSizeSet, avgErrorTestRates, 'b', label = 'Training Error')
    plt.plot(trainSizeSet, avgErrorTrainRates, 'r', label = 'Testing Error')
    plt.xlabel('Training Set Size')
    plt.ylabel('Prediction Error')
    plt.legend()
    plt.show()

def findTwoLargestInds(wHat):
    index1 = index2 = -1
    w1 = w2 = 0
    for idx, w in enumerate(abs(wHat[1:])):
        if w >= w1 and w >= w2:
            w2, index2 = w1, index1
            w1, index1 = w, idx
        elif w2 <= w < w1:
            w2, index2 = w, idx
    return index1, index2

def plotDecisionBoundary(wHat, XTest, yTest):
    index1, index2 = findTwoLargestInds(wHat)
    X1 = XTest[:, index1]
    X2 = XTest[:, index2]
    colormap = np.array(['r', 'g'])
    plt.scatter(X1, X2, s = 50, c=colormap[yTest.flatten()])
    w0, w1, w2 = wHat.flatten()[0], wHat[1:].flatten()[index1], wHat[1:].flatten()[index2]
    delta = 0.5
    baseX = np.array([min(X1) - delta, max(X1) + delta])
    fun = lambda w0, w1, w2, x : (-w1 * x - w0) / w2
    plt.plot(baseX, fun(w0, w1, w2, baseX))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Decision Boundary')
    plt.show()
