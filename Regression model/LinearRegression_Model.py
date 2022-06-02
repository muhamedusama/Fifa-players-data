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
import sklearn.metrics as sm
import pickle
#load data from file
data = pd.read_csv('player-value-prediction.csv')

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


#Encoding the rest of features:
cols = ('nationality', 'preferred_foot', 'body_type', 'club_team', 'club_position')
data = Feature_Encoder(data, cols)
#one hot encoding for positions column:
data = pd.concat([data.drop('positions', 1), data['positions'].str.get_dummies(sep=",")], 1)
#print("data fter one hot encoding:",data)

#Select features
X = data.iloc[:, 4:72] #Features
del X['value']
Y = data['value']


#correlation:
corr = data.corr()
top_feature = corr.index[abs(corr['value']) > 0.5]
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


#Apply Linear Regression on the selected features
cls = linear_model.LinearRegression()
start = time.time()
cls.fit(X_train, y_train)
stop = time.time()
prediction = cls.predict(X_test)

print('Mean Square Error of linear regression:', metrics.mean_squared_error(np.asarray(y_test), prediction))
print('Model Accuracy with Linear Regression ', sm.r2_score(y_test, prediction))
print("Time in linear regression = ", stop - start)


#polynomial Regression:
poly_features = PolynomialFeatures(degree=3)
# transforms the existing features to higher degree features.

X_train_poly = poly_features.fit_transform(X_train)
# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
start2 = time.time()
poly_model.fit(X_train_poly, y_train)
stop2 = time.time()

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

ypred = poly_model.predict(poly_features.transform(X_test))
# predicting on test data-set
Poly_prediction = poly_model.predict(poly_features.fit_transform(X_test))
print("Mean using Polynomial Regression:")
print('Mean Square Error', metrics.mean_squared_error(y_test, Poly_prediction))
print('Model Accuracy with Polynomial Regression ', sm.r2_score(y_test, Poly_prediction))
print("Time in polynomial = ", stop2 - start2)
#pickle.dump(cls, open('linear_model.pkl', 'wb'))
#pickle.dump(Dictionary, open('Features.pkl', 'wb'))
#pickle.dump(poly_features, open('PolyFeatures.pkl', 'wb'))
#pickle.dump(poly_model, open('PolyModel.pkl', 'wb'))




