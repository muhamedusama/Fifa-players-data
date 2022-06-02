import pandas as pd
import time
from Pre_processing import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn import linear_model
import category_encoders as ce
#from LinearRegression_Model import *
import sklearn.metrics as sm
import pickle
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


#loading necessary data
pickled_model = pickle.load(open('linear_model.pkl', 'rb'))
PolyFeatures = pickle.load(open('PolyFeatures.pkl', 'rb'))
PolyModel = pickle.load(open('PolyModel.pkl', 'rb'))
TopFeatures = pickle.load(open('Features.pkl', 'rb'))

#reading file
data = pd.read_csv('player-test-samples.csv')

y_test = data['value']
data = data[TopFeatures]

#filling missing rows with mean or zero is colum empty
for i in data.columns:
    if(math.isnan(data[i].mean())):
        data[i].fillna(value = 0, inplace=True)
        continue
        data[i].fillna((data[i].mean()), inplace=True)



X = data[TopFeatures]

#feature scaling
X = featureScaling(X, 0, 1)




#Linear Regression

linearmodel = pickled_model.predict(X)
rms = mean_squared_error(y_test, linearmodel, squared=False)




print("Mean using Linear Regression:")
print('Mean Square Error', metrics.mean_squared_error(y_test,linearmodel))
print('Model Accuracy with linear Regression ', sm.r2_score(y_test,linearmodel))
print("Model Accuracy with RMSE: ", rms)


#PolynomialRegression
X_poly = PolyFeatures.transform(X)
prediction = PolyModel.predict(X_poly)
rms = mean_squared_error(y_test, prediction, squared=False)

print("Mean using Polynomial Regression:")
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
print('Model Accuracy with Polynomial Regression ', sm.r2_score(y_test, prediction))
print("Model Accuracy with RMSE: ", rms)



