import csv
import math
import random

##################################################################################
########## pick the data set and split in test and train sets randomly ###########
##################################################################################
def splitDataset (dataset, splitRatio):
  trainSize = int(len(dataset) * splitRatio)
  trainSet = []
  testSet = list(dataset)

  # loop in dataSet*splitRatio times and make the data set's #
  while len(trainSet) < trainSize:
    index = random.randrange(len(testSet))
    trainSet.append(testSet.pop(index))

  return [trainSet, testSet]


##################################################################################
#### split dataSet according to classes. EX: separated[0 or 1] = 2500 samples ####
##################################################################################
def separatedByClass(dataset):
  separated = {}

  # for each sample (EX: dataset[i] = [feature1, feature2, feature3, class]) #
  for i in range(len(dataset)):
    vector = dataset[i]

    # if vector class not exists in separated, create it, if not, include the row #
    if (vector[-1] not in separated):
      separated[vector[-1]] = []
    separated[vector[-1]].append(vector)
  return separated


##################################################################################
########################### simply calculate the mean ############################  
##################################################################################
def mean(numbers):
  return sum(numbers)/float(len(numbers))


##################################################################################
######################## calculate the standart deviation ########################  
##################################################################################
def stdev(numbers):
  avg=mean(numbers)
  variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
  return math.sqrt(variance)


##################################################################################
## make the mean and std for each descriptor and delete the result for classes ###
##################################################################################
def summarize(dataset):
  summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
  del summaries[-1]
  return summaries


##################################################################################
##################### make the mean and std for each feature #####################
##################################################################################
def summarizeByClass(dataset):
  # get rows for each class: vector[class] = list(sampleFeatures) #
  separated = separatedByClass(dataset)
  summaries = {}

  # do the mean and std for all features (one mean and std for each feat.) #
  for classValue, instances in separated.items():
    summaries[classValue] = summarize(instances)

  return summaries


########################################################################################################
# calculate the probability with mean and std from classX[featureX], based on featureX from one sample #
############################ e^-[(x-mean)^2/2*std^2] ###################################################
########################################################################################################
def calculateProbability(x, mean, stdev):
  exponent= math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
  return (1/(math.sqrt(2*math.pi)*stdev))*exponent


##################################################################################
########## Calculate the % of the sample to be in the class1, class2... ##########
##################################################################################
def calculateClassProbabilities(summaries, inputVector):
  probabilities = {}

  # classSummaries = [allMeansFeature1(get mean and std), allStdsFeature1(get mean and std), allMeansFeature2, ..., class 1] 
  # classSummaries = [allMeansFeature1(get mean and std), allStdsFeature1(get mean and std), allMeansFeature2, ..., class 2]
  # classValue = classValue (EX: 0 for guitarSamples) 
  # loops into each class row #
  for classValue, classSummaries in summaries.items():
    probabilities[classValue] = 1

    # loops into each feature #
    for i in range(len(classSummaries)):
      mean, stdev = classSummaries[i]
      
      # pick the value of one feature of sample #
      x = inputVector[i]

      # calculate % of sample be in the class1, class2... #
      probabilities[classValue] *= calculateProbability(x, mean, stdev)

  return probabilities


##################################################################################
########### pick the best class probability based on a sample features ###########
##################################################################################
def predict(summaries, inputVector):

  # calculate probabilities of a sample be in class1, class2, ... #
  probabilities = calculateClassProbabilities(summaries, inputVector)
  bestLabel, bestProb = None, -1
  for classValue, probability in probabilities.items():

    # pick best class probability from a sample #
    if bestLabel is None or probability > bestProb:
      bestProb = probability
      bestLabel = classValue

  return bestLabel


##################################################################################
########## loops into all testSet and pick best class probabilities ##############
##################################################################################
def getPredictions(summaries, testSet):
  predictions = []

  # get best class probability for each sample #
  for i in range (len(testSet)):
    result = predict(summaries, testSet[i])
    predictions.append(result)
  return predictions


##################################################################################
# based on predicts compare all last column in testSet, to validate the classes  #
##################################################################################
def getAccuracy(testSet,predictions):
  correct = 0

  confusionMatrix = [[0,0],[0,0]]
  confusionMatrix[0][1] = 0
  confusionMatrix[0][0] = 0
  confusionMatrix[1][0] = 0
  confusionMatrix[1][1] = 0

  for x in range(len(testSet)):

    # compare prediction result based on testSet[i][classColumn] #
    # make the confusion matrix #
    if testSet[x][-1] == predictions[x]:
      if(predictions[x] == 0): 
        confusionMatrix[0][0] += 1
      else:
        confusionMatrix[1][1] += 1
      correct += 1
    else:
      if(predictions[x] == 0): 
        confusionMatrix[0][1] += 1
      else:
        confusionMatrix[1][0] += 1

  #print(confusionMatrix)
  return (correct/float(len(testSet)))*100.0


##################################################################################
#################### pick 3 classifiers and do voting system  ####################
##################################################################################
def emsemble(testSet,predictions1,predictions2,predictions3):
  correct = 0

  # make the confusion matrix #
  confusionMatrix = [[0,0],[0,0]]
  confusionMatrix[0][1] = 0
  confusionMatrix[0][0] = 0
  confusionMatrix[1][0] = 0
  confusionMatrix[1][1] = 0

  # for each sample, pick the result of classification and classifies based in major votes #
  for x in range(len(testSet)):
    class1 = 0
    class2 = 0
    if(predictions1[x]==0):
      class1 += 1
    else:
      class2 += 1
    if(predictions2[x]==0):
      class1 += 1
    else:
      class2 += 1
    if(predictions3[x]==0):
      class1 += 1
    else:
      class2 += 1

    # if vote for class 1 is the major #
    if(class1 > class2):
      if(testSet[x][-1] == 0):
        confusionMatrix[0][0] += 1
        correct += 1
      else:
        confusionMatrix[0][1] += 1
    else:
      if(testSet[x][-1] == 1):
        confusionMatrix[1][1] += 1
        correct += 1
      else:
        confusionMatrix[1][0] += 1

  #print(confusionMatrix)
  return (correct/float(len(testSet)))*100.0
