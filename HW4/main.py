import scipy.io as sio
from function import *


mat_content = sio.loadmat('HW4Data.mat')


XTrain = mat_content['XTrain']
yTrain = mat_content['yTrain']
XTest = mat_content['XTest']
yTest = mat_content['yTest']

wHat, objVals = LR_GradientAscent(XTrain, yTrain, eta = 0.01, tol = 0.001)

yHat, numErrors = LR_PredictLabels(XTest, yTest, wHat)

print(numErrors)

plotObjVals(objVals)

plotOnePredictionError(XTrain, yTrain, XTest, yTest)

plotAvgPredictionError(XTrain, yTrain, XTest, yTest, times = 50)

plotDecisionBoundary(wHat, XTest, yTest)
