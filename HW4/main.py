import scipy.io as sio
from function import *
from math import log, exp

mat_content = sio.loadmat('HW4Data.mat')
mat_content


XTrain = mat_content['XTrain']
yTrain = mat_content['yTrain']
XTest = mat_content['XTest']
yTest = mat_content['yTest']

print(type(XTrain))
def LR_CalcObj(XTrain, yTrain, wHat):
    vexp = np.vectorize(exp)
    vlog = np.vectorize(log)
    zHat = np.dot(XTrain, wHat[1:]) + wHat[0]
    logProb = yTrain * yHat - vlog(1 + vexp(yHat))
    return np.sum(logProb)

def sigmoid(Z)

def LR_CalcGrad(XTrain, yTrain, wHat):


LR_CalcObj(XTrain, yTrain, wHat)

vexp = np.vectorize(exp)
vlog = np.vectorize(log)

wHat = np.array([1] * 11).reshape(-1,1)
wHat[0]=1
wHat
wHat.shape
XTrain.shape

yHat = np.dot(XTrain, wHat[1:]) + wHat[0]
yHat
yHat.shape

a = vlog(1 + vexp(yHat))
a = yTrain * yHat - vlog(1 + vexp(yHat))
(a).shape
np.sum(a)

x1 = np.arange(9.0).reshape((-1,1))
x1
x1.shape
x2 = np.arange(3.0)
x2

XTrain
print(sum(XTrain[0, :]))
