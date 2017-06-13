import numpy as np
from math import log, exp

def LR_CalcObj(XTrain, yTrain, wHat):
    vexp = np.vectorize(exp)
    vlog = np.vectorize(log)
    zHat = np.dot(XTrain, wHat[1:]) + wHat[0]
    print(zHat)
    logProb = yTrain * yHat - vlog(1 + vexp(zHat))
    print(logProb)
    return np.sum(logProb)

LR_CalcObj(XTrain, yTrain, wHat)

def








def NB_Classify(D, p, X):
    """Complete the function [yHat] = NB Classify(D, p, X). The input X is an m × V matrix
    containing m feature vectors (stored as its rows). The output yHat is a m × 1 vector of predicted class
    labels, where yHat(i) is the predicted label for the ith row of X."""
    vlog = np.vectorize(log)
    Eco, Oni = 0, 1
    yHat = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        llGivenEco = X[i] * D[Eco] + (1 - X[i]) * (1 - D[Eco])
        llGivenOni = X[i] * D[Oni] + (1 - X[i]) * (1 - D[Oni])
        llGivenEco = np.append(llGivenEco, p)
        llGivenOni = np.append(llGivenOni, 1 - p)
        logPostEcoGivenX = logProd(vlog(llGivenEco))
        logPostOniGivenX = logProd(vlog(llGivenOni))
        yHat[i] = 1 if logPostEcoGivenX > logPostOniGivenX else 2
    return yHat

def NB_XGivenY(XTrain, yTrain):
    """Complete the function [D] = NB XGivenY(XTrain, yTrain). The output D is a 2 × V
    matrix, where for any word index w ∈ {1, . . . , V } and class index y ∈ {1, 2}, the entry D(y,w) is the
    MAP estimate of θyw = P(Xw = 1|Y = y) with a Beta(2,1) prior distribution. """
    Ecomomist = (yTrain == 1)
    Onion = (yTrain == 2)
    numEco, numOni = Ecomomist.sum() + 2, Onion.sum() + 2
    numXinEco = XTrain[Ecomomist,].sum(0) + 1
    numXinOni = XTrain[Onion,].sum(0) + 1
    return np.array([numXinEco / numEco, numXinOni / numOni])

def logProd(x):
    """ A function takes as input a vector of numbers in logspace  (i.e. xi = log pi)
    and returns the product of those numbers in logspace—i.e., logProd(x) = log(prod(pi)) """
    return sum(x)

def NB_YPrior(yTrain):
    """Complete the function [p] = NB YPrior(yTrain). The output p is the MLE for ρ = P(Y = 1)."""
    Ecomomist = (yTrain == 1)
    return Ecomomist.sum() / yTrain.size

def ClassificationError(yHat, yTruth):
    """Complete the function [error] = ClassificationError(yHat, yTruth), which takes two
    vectors of equal length and returns the proportion of entries that they disagree on."""
    return sum(yHat != yTruth) / len(yHat)


def TopOccurence(XTrain, yTrain, Vocabulary, k = 5):
    Ecomomist = (yTrain == 1)
    Onion = (yTrain == 2)
    Ecoindex = np.argsort(XTrain[Ecomomist,].sum(0))[-1 : -k - 1 : -1]
    Oniindex = np.argsort(XTrain[Onion,].sum(0))[-1 : -k - 1 : -1]
    return {'Ecomomist' : Vocabulary[Ecoindex], 'Onion' : Vocabulary[Oniindex]}

def TopDiscriminate(XTrain, yTrain, Vocabulary, k = 5):
    D = NB_XGivenY(XTrain, yTrain)
    Ecoindex = np.argsort(D[0] / D[1])[-1 : -k - 1 : -1]
    Oniindex = np.argsort(D[1] / D[0])[-1:-6:-1]
    return {'Ecomomist' : Vocabulary[Ecoindex], 'Onion' : Vocabulary[Oniindex]}
