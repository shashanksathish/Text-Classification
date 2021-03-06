import collections
import os
import re
import codecs
import numpy

Iteration = 10
ham = list()
spam = list()
countTrainHam = 0
countTrainSpam = 0
dictProbHam = dict()
dictProbSpam = dict()
learningRate = 0.001
regularization = 0.001

bias = 0

train_path = 'C:/Users/shash/Desktop/MS CE/Subject/Spring 2020/CS6375/CS-6375-Machine-Learning-master/Assignment 3/train'
directoryHam = train_path + '/ham'
directorySpam = train_path + '/spam'

test = 'C:/Users/shash/Desktop/MS CE/Subject/Spring 2020/CS6375/CS-6375-Machine-Learning-master/Assignment 3/test'
testHam = test + '/ham'
testSpam = test + '/spam'

# Regular expression to clean the data given in train ham and spam folder
regex = re.compile(r'[A-Za-z0-9\']')

def ReadFile(file,filepath):
    fileHandler = codecs.open(filepath+"\\" + file,'rU','latin-1')
    Findwords = re.findall('[A-Za-z0-9\']+', fileHandler.read())
    allwords = list()
    for word in Findwords:
        word = word.lower()
        allwords+=[word]
    fileHandler.close()    
    return allwords
    
def numberofFiles(filepath):
    wordList = list()
    NumberOfFiles = 0
    for files in os.listdir(filepath):
        if files.endswith(".txt"):
            wordList += ReadFile(files,filepath)
            NumberOfFiles+=1
    return wordList, NumberOfFiles

# iterating through train to get the list of ham words used to form combined bag of words
ham, countTrainHam = numberofFiles(directoryHam)
spam, countTrainSpam = numberofFiles(directorySpam)

# iterating through test to get the list of ham words used to form combined bag of words
hamTest, countTestHam = numberofFiles(testHam)
SpamTest, countTestSpam = numberofFiles(testSpam)

bagOfWords = ham + spam
dictBagOfWords = collections.Counter(bagOfWords)
listBagOfWords = list(dictBagOfWords.keys())
TargetList = list()  # final value of ham or spam, ham = 1 & spam = 0
totalFiles = countTrainHam + countTrainSpam

# correct it to testham/spam
testBagOfWords = hamTest + SpamTest
testDictBagOfWords = collections.Counter(testBagOfWords)
testListBagOfWords = list(testDictBagOfWords.keys())
testTargetList = list()  # final value of ham or spam, ham = 1 & spam = 0
totalTestFiles = countTestHam + countTestSpam

# initialize matrix to zero and use list comprehension to create this matrix
def initiliazeMatrix(row, column):
    featureMatrix = [0] * row
    for i in range(row):
        featureMatrix[i] = [0] * column
    return featureMatrix

trainFeatureMatrix = initiliazeMatrix(totalFiles, len(listBagOfWords))
testFeatureMatrix = initiliazeMatrix(totalTestFiles, len(testListBagOfWords))

rowMatrix = 0
testRowMatrix = 0

sigMoidList = list()  # for each row
for i in range(totalFiles):
    sigMoidList.append(-1)
    TargetList.append(-1)

for i in range(totalTestFiles):
    testTargetList.append(-1)

weightOfFeature = list()

for feature in range(len(listBagOfWords)):
    weightOfFeature.append(0)


def makeMatrix(featureMatrix, path, listBagOfWords, rowMatrix, classifier, TargetList):
    for fileName in os.listdir(path):
        words = ReadFile(fileName, path)
        temp = dict(collections.Counter(words))
        for key in temp:
            if key in listBagOfWords:
                column = listBagOfWords.index(key)
                featureMatrix[rowMatrix][column] = temp[key]
        if (classifier == "ham"):
            TargetList[rowMatrix] = 0
        elif (classifier == "spam"):
            TargetList[rowMatrix] = 1
        rowMatrix += 1
    return featureMatrix, rowMatrix, TargetList


#train matrix including ham and spam
trainFeatureMatrix, rowMatrix, TargetList = makeMatrix(trainFeatureMatrix, directoryHam, listBagOfWords, rowMatrix,
                                                       "ham", TargetList)
trainFeatureMatrix, rowMatrix, TargetList = makeMatrix(trainFeatureMatrix, directorySpam, listBagOfWords, rowMatrix,
                                                       "spam", TargetList)

testFeatureMatrix, testRowMatrix, testTargetList = makeMatrix(testFeatureMatrix, testHam, testListBagOfWords,
                                                              testRowMatrix, "ham", testTargetList)
testFeatureMatrix, testRowMatrix, testTargetList = makeMatrix(testFeatureMatrix, testSpam, testListBagOfWords,
                                                              testRowMatrix, "spam", testTargetList)


# for each column
def sigmoid(x):
    den = (1 + numpy.exp(-x))
    sigma = 1 / den
    return sigma


# Calculate for each file
def sigmoidFunction(totalFiles, totalFeatures, featureMatrix):
    global sigMoidList
    for files in range(totalFiles):
        summation = 1.0
        
        for features in range(totalFeatures):
            summation += featureMatrix[files][features] * weightOfFeature[features]
        sigMoidList[files] = sigmoid(summation)


def calculateWeightUpdate(totalFiles, numberOfFeature, featureMatrix, TargetList):
    global sigMoidList

    for feature in range(numberOfFeature):
        weight = bias
        for files in range(totalFiles):
            frequency = featureMatrix[files][feature]
            y = TargetList[files]
            sigmoidValue = sigMoidList[files]
            weight += frequency * (y - sigmoidValue)

        oldW = weightOfFeature[feature]
        weightOfFeature[feature] += ((weight * learningRate) - (learningRate * regularization * oldW))

    return weightOfFeature


def trainingFunction(totalFiles, numbeOffeatures, trainFeatureMatrix, TargetList):
    sigmoidFunction(totalFiles, numbeOffeatures, trainFeatureMatrix)
    calculateWeightUpdate(totalFiles, numbeOffeatures, trainFeatureMatrix, TargetList)


def classifyData():
    correctHam = 0
    incorrectHam = 0
    correctSpam = 0
    incorrectSpam = 0
    idx=0
    print("Classifyig the test files, this may take a while \n")
    for file in range(totalTestFiles):
        summation = 1.0
        print(file)
        for i in range(len(testListBagOfWords)):
            word = testListBagOfWords[i]

            if word in listBagOfWords:
                index = listBagOfWords.index(word)
                weight = weightOfFeature[index]
                wordcount = testFeatureMatrix[file][i]

                summation += weight * wordcount

        sigSum = sigmoid(summation)
        if (testTargetList[file] == 0):
            if sigSum < 0.5:
                correctHam += 1.0
            else:
                incorrectHam += 1.0
        else:
            if sigSum >= 0.5:
                correctSpam += 1.0
            else:
                incorrectSpam += 1.0
        idx += 1
    print("Classifyig the test files, this may take a while \n") 
    print("\nRegularization Parameter is: " + str(regularization))    
    print("\nLearning Rate is: " + str(learningRate))
    print("Accuracy on Ham:" + str((correctHam / (correctHam + incorrectHam)) * 100))
    print("Accuracy on Spam:" + str((correctSpam / (correctSpam + incorrectSpam)) * 100))
    print("Overall Accuracy :" + str(((correctHam+correctSpam) / (correctHam + incorrectHam+correctSpam + incorrectSpam)) * 100))

for i in range(Iteration):
    print("Iteration of training: " + str(i + 1))
    trainingFunction(totalFiles, len(listBagOfWords), trainFeatureMatrix, TargetList)


classifyData()

