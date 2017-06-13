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
    oldObj = 99999999
    wSize = XTrain.shape[1] + 1
    wHat = np.array([0] * wSize).reshape(-1,1)
    oldObj = LR_CalcObj(XTrain, yTrain, wHat)
    while True:
        grad = LR_CalcGrad(XTrain, yTrain, wHat)
        wHat = LR_UpdateParams(wHat, grad, eta)
        newObj = LR_CalcObj(XTrain, yTrain, wHat)
        if LR_CheckConvg(oldObj, newObj, tol):
            break
        oldObj = newObj
    return wHat, newObj
