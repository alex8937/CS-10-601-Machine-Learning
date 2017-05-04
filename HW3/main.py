import scipy.io as sio
from function import *

mat_content = sio.loadmat('HW3Data.mat')
Vocabulary = mat_content['Vocabulary']
XTrain = mat_content['XTrain'].toarray()
yTrain = mat_content['yTrain'].flatten()
XTest = mat_content['XTest'].toarray()
yTest = mat_content['yTest'].flatten()
XTrainSmall = mat_content['XTrainSmall'].toarray()
yTrainSmall = mat_content['yTrainSmall'].flatten()

D = NB_XGivenY(XTrain, yTrain)
p = NB_YPrior(yTrain)
yHatTrain = NB_Classify(D, p, XTrain)
yHatTest = NB_Classify(D, p, XTest)
trainError = ClassificationError(yHatTrain, yTrain);
testError = ClassificationError(yHatTest, yTest);
trainError
testError

D = NB_XGivenY(XTrainSmall, yTrainSmall)
p = NB_YPrior(yTrainSmall)
yHatTrainSmall = NB_Classify(D, p, XTrainSmall)
yHatTestSmall = NB_Classify(D, p, XTest)
trainErrorSmall = ClassificationError(yHatTrainSmall, yTrainSmall);
testErrorSmall = ClassificationError(yHatTestSmall, yTest);
trainErrorSmall
testErrorSmall


TopOccurence(XTrain, yTrain, Vocabulary, k = 5)
TopDiscriminate(XTrain, yTrain, Vocabulary, k = 5)
