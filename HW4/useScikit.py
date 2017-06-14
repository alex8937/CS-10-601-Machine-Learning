import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

mat_content = sio.loadmat('HW4Data.mat')
XTrain = mat_content['XTrain']
yTrain = mat_content['yTrain']
XTest = mat_content['XTest']
yTest = mat_content['yTest']

XTrain = np.insert(XTrain, 0, 1, axis = 1)
XTest = np.insert(XTest, 0, 1, axis = 1)
yTest, yTrain = yTest.flatten(), yTrain.flatten()


model = LogisticRegression(tol = 0.001)

fit = model.fit(XTrain, yTrain)

yPred = model.predict(XTest)

accuracy_score(yTest, yPred)
confusion_matrix(yTest, yPred)
