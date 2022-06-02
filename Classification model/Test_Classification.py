import pandas as pd
import time
from Pre_processing import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn import metrics
from sklearn.svm import SVC
import category_encoders as ce
import pickle
import math

#loading necessary data
SVC_model_poly = pickle.load(open('SVC_model1.pkl', 'rb'))
SVC_model_Kernel_ovo = pickle.load(open('SVC_model2.pkl', 'rb'))
SVC_model_kernel_ovr = pickle.load(open('SVC_model3.pkl', 'rb'))
TopFeatures = pickle.load(open('Features.pkl', 'rb'))
encoder = pickle.load(open('Yencoder.pkl', 'rb'))
rbf = pickle.load(open('rbf.pkl', 'rb'))

#reading files
data = pd.read_csv('player-test-samples.csv')
y_test = data['PlayerLevel']
data = data[TopFeatures]
#filling missing values with mean
for i in data.columns:
    if(math.isnan(data[i].mean())):
        data[i].fillna(value = 0, inplace=True)
        continue
        data[i].fillna((data[i].mean()), inplace=True)



X = data[TopFeatures]
X = featureScaling(X, 0, 1)

y_model1 = SVC_model_poly.predict(X)
y_model2 = SVC_model_Kernel_ovo.predict(X)
y_model3 = SVC_model_kernel_ovr.predict(X)
y_model4 = rbf.predict(X)

newY = encoder.transform(y_test)

#z = encoder.transform(X["wage"])
print("Accuracy of poly svc:", metrics.accuracy_score(newY, y_model1))
print("Accuracy Of SVM Svc linear onevsone:", metrics.accuracy_score(newY, y_model2))
print("Accuracy Of svm linear onevsrest:", metrics.accuracy_score(newY, y_model3))
#weak model
print("Accuracy Of RBF SVC:", metrics.accuracy_score(newY, y_model4))

