import scipy.io as sio
from function import *
from math import log, exp

mat_content = sio.loadmat('HW4Data.mat')
mat_content


XTrain = mat_content['XTrain']
yTrain = mat_content['yTrain']
XTest = mat_content['XTest']
yTest = mat_content['yTest']

wHat = np.array([1] * 11).reshape(-1,1)
wHat[0]=0
wHat

obj = LR_CalcObj(XTrain, yTrain, wHat)

grad = LR_CalcGrad(XTrain, yTrain, wHat)

wHat = LR_UpdateParams(wHat, grad, 1)

hasConverged = LR_CheckConvg(1, obj, 1)

 LR_GradientAscent(XTrain, yTrain, eta = 0.01, tol = 0.001)
