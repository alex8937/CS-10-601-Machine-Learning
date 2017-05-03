import scipy.io as sio
import numpy as np
from math import log

mat_content = sio.loadmat('HW3Data.mat')
Vocabulary = mat_content['Vocabulary']
XTrain = mat_content['XTrain']
yTrain = mat_content['yTrain']
XTest = mat_content['XTest']
yTest = mat_content['yTest']
XTrainSmall = mat_content['XTrainSmall']
yTrainSmall = mat_content['yTrainSmall']

vlog = np.vectorize(log)

print(yTrain.size)
print(yTrain)
print(Vocabulary.size)
print(type(XTrain))
print(XTrain.shape)
print(XTrain)

XTrain[:,0]

XTrain[:,0].todense()

np.multiply(np.array(yTrain == 1), XTrain[:,0].todense()).sum()
list(range(3))
XTrain.shape[1]
np.array([np.multiply((yTrain == 1), XTrain[:,i].todense()).sum() for i in range(0,XTrain.shape[1])])

x = XTrain.tocoo()

print(x.col)
print(x.row)
a = list(zip(x.col, x.row))
(1, 5) in a


def logProd(x):
    """ A function takes as input a vector of numbers in logspace  (i.e. xi = log pi)
    and returns the product of those numbers in logspace—i.e., logProd(x) = log(prod(pi)) """
    return sum(x)

NB_XGivenY(XTrain, yTrain)
yTrain.size
NB_YPrior(yTrain)

def NB_XGivenY(XTrain, yTrain):
    alpha = 2
    beta = 1

    numYeq1, numYeq2 = (yTrain == 1).sum(), (yTrain == 2).sum()
    numYeq1 += alpha - 1 + beta - 1
    numYeq1 += alpha - 1 + beta - 1

    numXinYeq1 = np.array([np.multiply((yTrain == 1), XTrain[:,i].todense()).sum() for i in range(0,XTrain.shape[1])])
    numXinYeq1 += alpha - 1

    numXinYeq2 = np.array([np.multiply((yTrain == 2), XTrain[:,i].todense()).sum() for i in range(0,XTrain.shape[1])])
    numXinYeq2 += beta - 1

    XGivenYeq1 = numXinYeq1 / numYeq1
    XGivenYeq2 = numXinYeq2 / numYeq2

    return np.array([XGivenYeq1, XGivenYeq2])

def  NB_YPrior(yTrain):
    return (yTrain == 1).sum() / yTrain.size
