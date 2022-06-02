import numpy as np
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



#load data from file
data = pd.read_csv('player-classification.csv')
#drop Features that have more than 10% missing values:
col_Min_nonmissing = len(data) * 0.90  #len(data) = number of rows
data.dropna(axis=1, thresh=col_Min_nonmissing, inplace=True)

#drop rows that have any missing value:
data.dropna(axis=0, how='any', inplace=True)

#print(data)

#label encoding Work Rate:
# create object of Ordinalencoding
# Sum all work rates: High = 3, medium = 2, low = 1
encoder = ce.OrdinalEncoder(cols=['work_rate'], return_df=True,
                           mapping=[{'col': 'work_rate',
'mapping': {'High/ High': 6, 'High/ Medium': 5, 'High/ Low': 4,
              'Medium/ High': 5, 'Medium/ Medium': 4, 'Medium/ Low': 3,
               'Low/ High': 4, 'Low/ Medium': 3, 'Low/ Low': 2}}])
data['work_rate'] = encoder.fit_transform(data['work_rate'])

encoder = ce.OrdinalEncoder(cols=['PlayerLevel'], return_df=True, mapping=[{'col': 'PlayerLevel',
                      'mapping': {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}}]
                            )
data['PlayerLevel'] = encoder.fit_transform(data['PlayerLevel'])

label_encoding = encoder;

#Encoding the rest of features:
cols = ('nationality', 'preferred_foot', 'body_type', 'club_team', 'club_position')
data = Feature_Encoder(data, cols)
#one hot encoding for positions column:
data = pd.concat([data.drop('positions', 1), data['positions'].str.get_dummies(sep=",")], 1)
#print("data fter one hot encoding:",data)

#Select features
X = data.iloc[:, 4:72] #Features
del X['PlayerLevel']
Y = data['PlayerLevel']


#correlation:
corr = data.corr()
top_feature = corr.index[abs(corr['PlayerLevel']) > 0.5]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
X = X[top_feature]
Dictionary = top_feature

#Scaling Data
X = featureScaling(X, 0, 1)

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=120)

C = 0.001
start1 = time.time()
svm_kernel_ovo = OneVsOneClassifier(SVC(kernel='linear')).fit(X_train, y_train)
stop1 = time.time()

start2 = time.time()
svm_kernel_ovr = OneVsRestClassifier(SVC(kernel='linear')).fit(X_train, y_train)
stop2 = time.time()

start3 = time.time()
svm_poly_ovr = OneVsOneClassifier(SVC(kernel='poly', degree=3)).fit(X_train, y_train)
stop3 = time.time()

rbf_svc1 = OneVsOneClassifier(SVC(kernel='rbf', gamma=0.8, C=90)).fit(X_train, y_train)
rbf_svc2 = OneVsOneClassifier(SVC(kernel='rbf', gamma=0.8, C=5)).fit(X_train, y_train)
rbf_svc3 = OneVsOneClassifier(SVC(kernel='rbf', gamma=0.8, C=0.00001)).fit(X_train, y_train)
rbf_svc4 = OneVsOneClassifier(SVC(kernel='rbf', gamma=9, C=C)).fit(X_train, y_train)
rbf_svc5 = OneVsOneClassifier(SVC(kernel='rbf', gamma=3, C=C)).fit(X_train, y_train)
rbf_svc6 = OneVsOneClassifier(SVC(kernel='rbf', gamma=0.0001, C=C)).fit(X_train, y_train)



#poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
TestStart1 = time.time()
y_predict1 = svm_kernel_ovo.predict(X_test)
TestStop1 = time.time()

TestStart2 = time.time()
y_predict2 = svm_kernel_ovr.predict(X_test)
TestStop2 = time.time()

TestStart3 = time.time()
y_predict3 = svm_poly_ovr.predict(X_test)
TestStop3 = time.time()

y_predict4 = rbf_svc1.predict(X_test)
y_predict5 = rbf_svc2.predict(X_test)
y_predict6 = rbf_svc3.predict(X_test)
y_predict7 = rbf_svc4.predict(X_test)
y_predict8 = rbf_svc5.predict(X_test)
y_predict9 = rbf_svc6.predict(X_test)

Linear1Acc = metrics.accuracy_score(y_test, y_predict1);
Linear2Acc = metrics.accuracy_score(y_test, y_predict2);
PolyAcc = metrics.accuracy_score(y_test, y_predict3);
TrainTime1 = stop1 - start1
TrainTime2 = stop2 - start2
TrainTime3 = stop3 - start3
TestTime1 = TestStop1 - TestStart1
TestTime2 = TestStop2 - TestStart2
TestTime3 = TestStop3 - TestStart3

print("Training Time of SVM Svc Linear onevsone:", stop1 - start1)
print("Accuracy Of SVM Svc linear onevsone:", metrics.accuracy_score(y_test, y_predict1))
print("Testing Time of SVM Svc Linear onevsone:", TestStop1 - TestStart1)
print("-------------------------------------------")

print("Training Time of SVM Svc Linear onevsrest:", stop2 - start2)
print("Accuracy Of svm linear onevsrest:", metrics.accuracy_score(y_test, y_predict2))
print("Testing Time of SVM Svc Linear onevsrest:", TestStop2 - TestStart2)
print("-------------------------------------------")

print("Training Time of Poly svc:", stop3 - start3)
print("Accuracy Of poly svc:", metrics.accuracy_score(y_test, y_predict3))
print("Testing Time of Poly svc:", TestStop3 - TestStart3)
print("--------------------------------------------")

print("Accuracy Of rbf svc with C = 90:", metrics.accuracy_score(y_test, y_predict4))
print("Accuracy Of rbf svc with C = 5:", metrics.accuracy_score(y_test, y_predict5))
print("Accuracy Of rbf svc with C = 0.00001:", metrics.accuracy_score(y_test, y_predict6))
print("Accuracy Of rbf svc with gemma = 10:", metrics.accuracy_score(y_test, y_predict7))
print("Accuracy Of rbf svc with gemma = 5:", metrics.accuracy_score(y_test, y_predict8))
print("Accuracy Of rbf svc with gemma = 0.0001:", metrics.accuracy_score(y_test, y_predict9))

#Visualization Accuracy
Data ={"Linear 1vs1": Linear1Acc, "Linear 1vsAll": Linear2Acc, "Poly": PolyAcc}
KernalFuns = list(Data.keys())
Accuracy = list(Data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(KernalFuns, Accuracy, color ='blue',
        width = 0.4)
#plt.xlabel("Courses offered")
#plt.ylabel("No. of students enrolled")
plt.title("Accuraccy Differences in Kernel Functions")
plt.show()

#Visualising Training Time
Data ={"Linear 1vs1": TrainTime1, "Linear 1vsAll": TrainTime2, "Poly": TrainTime3}
KernalFuns = list(Data.keys())
Accuracy = list(Data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(KernalFuns, Accuracy, color ='blue',
        width = 0.4)
plt.title("Training Time Differences in Kernel Functions")
plt.show()

#Visualising Training Time
Data ={"Linear 1vs1": TestTime1, "Linear 1vsAll": TestTime2, "Poly": TestTime3}
KernalFuns = list(Data.keys())
Accuracy = list(Data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(KernalFuns, Accuracy, color ='blue',
        width = 0.4)
plt.title("Test Time Differences in Kernel Functions")
plt.show()

#pickle.dump(svm_poly_ovr, open('SVC_model1.pkl', 'wb'))
#pickle.dump(svm_kernel_ovo, open('SVC_model2.pkl', 'wb'))
#pickle.dump(svm_kernel_ovr, open('SVC_model3.pkl', 'wb'))
#pickle.dump(Dictionary, open('Features.pkl', 'wb'))
pickle.dump(label_encoding, open('Yencoder.pkl', 'wb'))
#pickle.dump(rbf_svc6, open('rbf.pkl', 'wb'))







