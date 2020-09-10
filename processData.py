import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from os import listdir
from os.path import isfile, join
import naiveBayes as nb

# pick sample files #
audiodir = 'baixoXguitarra/'
filenames = [f for f in listdir(audiodir) if isfile(join(audiodir, f))]
filenames = sorted(filenames)

feature_set =[]

# apply audio descritors to extract audio features #
for i in range(len(filenames)):

  # pick audio values and the sample rate #
  y, sr = librosa.load(audiodir+filenames[i])

  # apply short-time Fourier transform and pick module of positive and negative numbers #
  S = np.abs(librosa.stft(y))

  # pick mean and standard deviation from centroid descriptor output #
  centroid = librosa.feature.spectral_centroid(S=S)
  centroid_mean = np.mean(centroid)
  centroid_std = np.std(centroid)
  
  # pick mean and standard deviation from flatness descriptor output #
  flatness = librosa.feature.spectral_flatness(S=S)
  flatness_mean = np.mean(flatness)
  flatness_std = np.std(flatness)
 
  # pick mean and standard deviation from rolloff descriptor output # 
  rolloff = librosa.feature.spectral_rolloff(S=S)
  rolloff_mean = np.mean(rolloff)
  rolloff_std = np.std(rolloff)
  
  # pick mean and standard deviation from mfcc descriptor output #
  mfcc = librosa.feature.mfcc(S=S)
  mfcc_mean = np.mean(mfcc)
  mfcc_std = np.std(mfcc)
  

  # mount the list of features with collected values and putting the respective class of each sample #
  if(filenames[i][0]=="b"):
    features = [centroid_mean, centroid_std, flatness_mean, flatness_std, rolloff_mean, rolloff_std, mfcc_mean, mfcc_std,0]
  else:
    features = [centroid_mean, centroid_std, flatness_mean, flatness_std, rolloff_mean, rolloff_std, mfcc_mean, mfcc_std,1]
  
  feature_set.append(features)

with open('dataProcessed.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % features for features in feature_set)
############################################################

print(feature_set);
# define empty list
feature_set = []

# open file and read the content in a list
with open('dataProcessed.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()

    for line in filecontents:
        # remove linebreak which is the last character of the string
        features = line[:-1]

        # add item to the list
        feature_set.append(features)

#print(feature_set);




