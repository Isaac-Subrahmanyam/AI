#!/usr/bin/python

import random
from util import *

def extractWordFeatures(x):
    word_features=dict()
    split_words = str(x).split(' ')
    for i in split_words:
        if i not in word_features:
            word_features[i]=1
        else:
            word_features[i]+=1
    return word_features

def learnClassifier(trainExamples, testExamples, featureExtractor, numEpochs, eta):
    weights = {} 
    for i in range(numEpochs):
        for j in trainExamples:
            x1, x2 = j
            takeValue = x2 * dotProduct(weights, featureExtractor(x1))
            if takeValue < 1:
                increment(weights, -x2*-eta, featureExtractor(x1))        
        print("Train error: " + str(evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))))
        print("Test error: " + str(evaluatePredictor(testExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))))
    return weights

def generateDataset(numExamples, weights):
    random.seed(42)
    def generateExample():
        phi = {}
        for i in weights.keys():
            num = random.randint(-10, 10);
            if (num > 0):
                phi[i] = num
            if dotProduct(phi,weights) >= 0:
                x=1
            else:
                x=-1
        return (phi, x)
    return [generateExample() for _ in range(numExamples)]

def extractCharacterFeatures(n):
    def extract(x):
        x = "".join(x.split())
        ngrams = {}
        rang = range(0, len(x) + 1 - n)
        for i in rang:
            ngram = x[i:i + n]
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    return extract

if __name__ == "__main__":

    weights = {}
    # generate random weights 
    import string
    for _ in range(100):
        k = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
        v = random.uniform(-1, 1)
        weights[k] = v

    # generate train and test examples
    trainExamples = generateDataset(100, weights)
    testExamples = generateDataset(100, weights)
    for phi, y in trainExamples:
        if (dotProduct(phi, weights) >= 0 and y == -1) or (dotProduct(phi, weights) < 0 and y == 1):
             print("Examples are not correctoy generate!")

    for phi, y in testExamples:
        if (dotProduct(phi, weights) >= 0 and y == -1) or (dotProduct(phi, weights) < 0 and y == 1):
             print("Examples are not correctoy generate!")

    # Train and test the classifier on the synthetic examples
    featureExtractor = lambda x : x
    weights = learnClassifier(trainExamples, testExamples, featureExtractor, numEpochs=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(testExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(testExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))
        
    ############################################################
    # Train and test the classifier on the real data
    ############################################################
    trainExamples = readExamples('polarity.train')
    testExamples = readExamples('polarity.dev')
    featureExtractor = extractWordFeatures
    weights = learnClassifier(trainExamples, testExamples, featureExtractor, numEpochs=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(testExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(testExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

    ############################################################
    # Train and test the classifier on the feature extractor provided by extractCharacterFeatures
    ############################################################
    trainExamples = readExamples('polarity.train')
    testExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(7)
    weights = learnClassifier(trainExamples, testExamples, featureExtractor, numEpochs=20, eta=0.01)
    outputWeights(weights, 'weights_charachter')
    outputErrorAnalysis(testExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(testExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))


