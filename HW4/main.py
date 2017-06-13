import scipy.io as sio
from function import *
import matplotlib.pyplot as plt

mat_content = sio.loadmat('HW4Data.mat')


XTrain = mat_content['XTrain']
yTrain = mat_content['yTrain']
XTest = mat_content['XTest']
yTest = mat_content['yTest']



obj = LR_CalcObj(XTrain, yTrain, wHat)

grad = LR_CalcGrad(XTrain, yTrain, wHat)

wHat = LR_UpdateParams(wHat, grad, 1)

hasConverged = LR_CheckConvg(1, obj, 1)

wHat, objVals = LR_GradientAscent(XTrain, yTrain, eta = 0.01, tol = 0.001)

yHat, numErrors = LR_PredictLabels(XTest, yTest, wHat)

numErrors


plotObjVals(objVals)




plotOnePredictionError(XTrain, yTrain, XTest, yTest)
plotAvgPredictionError(XTrain, yTrain, XTest, yTest, times = 50)


index1 = index2 = -1
w1 = w2 = -999999999
for idx, w in enumerate(wHat[1:]):
    if w >= w1 and w >= w2:
        w2, index2 = w1, index1
        w1, index1 = w, idx
    else if w2 <= w < w:
        w2, index2 = w, idx
print(w1, w2)


wHat
w = wHat[1:]
w.argmax() + 1
w
