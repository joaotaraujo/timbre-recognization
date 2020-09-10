# Audio, maths and plot functions #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from os import listdir
from os.path import isfile, join
import naiveBayes as nb
import ast

# Sklearn libraries #
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# define festures list with classes #
feature_set = []

# define festures list without classes #
feature_setNOCLASS = []

# open file and put preprocessed data in lists #
with open('dataProcessed.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()

    for line in filecontents:
        # remove linebreak which is the last character (/n) of the string #
        features = line[:-1]

        # add item to the features list #
        feature_set.append( ast.literal_eval(features))
        feature_setNOCLASS.append( ast.literal_eval(features))

# convert feature_set to use .corr() function to make correlation features matrix #
dfObj = pd.DataFrame(feature_set) 

# make the correlation matrix and plot it #
plt.figure(figsize=(30,10))
cor = dfObj.corr()
ax = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

# make the list with percentage of relevance of features #
# Selecting highly correlated features #
cor_target = abs(cor[8]) 
relevant_features = cor_target[cor_target>0.0]
print(relevant_features)

# create a vector with sample classes #
y = []
for j in range(len(feature_set)):
    if(feature_set[j][8]==0):
        y.append(0);
    else:
        y.append(1);

# remove classes from feature_setNOCLASS #
[r.pop(8) for r in feature_setNOCLASS]



print("###########################################")

# percentage of features to use in f-test #
featuresPercent = 0

# loops in range(x) #
for rep in range(5):
  print("Rep:", rep)
  print("----------------------------------")

  # do cross validation #
  kf = KFold(5, shuffle=True, random_state=rep)

  # for each loop, use 20% more of features #
  featuresPercent += 20
  fs = feature_selection.SelectPercentile(feature_selection.f_classif, percentile=featuresPercent)
  feature_set_selected = fs.fit_transform(feature_setNOCLASS, y)
  x = pd.DataFrame(feature_set_selected)
  print(x.shape)


  for linhas_treino, linhas_valid in kf.split(feature_set_selected):
    #print("Treino:", linhas_treino.shape[0])
    #print("Valid:", linhas_valid.shape[0])

    # pick train and test data from cross validation #
    X_train = [feature_set_selected[i] for i in linhas_treino]
    X_test = [feature_set_selected[i] for i in linhas_valid]
    y_train = [y[i] for i in linhas_treino] 
    y_test = [y[i] for i in linhas_valid]
    #X_train, X_test, y_train, y_test = train_test_split(feature_setNOCLASS, y, stratify=y, test_size=0.4)

    # Naive Bayes (Sklearn) #
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("NB: {0:0.2f}%".format(acc*100))

    # SVC (Sklearn) #
    classifier = SVC(kernel="linear", C=0.025)
    classifier.fit(X_train, y_train)
    y_pred_SVC = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred_SVC)
    print("SVC: {0:0.2f}%".format(acc*100))


    # Decision Tree (Sklearn) #
    classifier = tree.DecisionTreeClassifier(criterion='entropy',max_depth=6, min_samples_leaf=5)
    classifier.fit(X_train, y_train)
    y_pred_tree = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred_tree)
    print("DecTree: {0:0.2f}%".format(acc*100))

    # pick train and test with classes to apply the implemented Naive Bayes #
    train = np.c_[X_train,y_train]
    test = np.c_[X_test,y_test]

    # pick the mean and std for each feature #
    summarie = nb.summarizeByClass(train)

    # pick best class probabilities from all testSet #
    predictionNB = nb.getPredictions(summarie, test)

    # compare predictions with testSet classes #
    accuracy = nb.getAccuracy(test, predictionNB)
    print('NBImp: {0:0.2f}%'.format(accuracy))

    # compare classifiers predictions with testSet classes to do voting system (emsemble) #
    accuracy = nb.emsemble(test, y_pred_tree,predictionNB,y_pred_SVC)
    print('Emsemb: {0:0.2f}%'.format(accuracy))
  rep += 1
  print("###########################################")
