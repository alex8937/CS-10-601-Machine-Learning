import numpy as np
from math import log, exp

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
    numIters
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
