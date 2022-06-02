import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
def Feature_Encoder(X, cols):
    for c in cols:
       lbl = LabelEncoder()
       lbl.fit(list(X[c].values))
       X[c] = lbl.transform(list(X[c].values))
    return X

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in (range(X.shape[1])):
        if (((max(X[:, i]) - min(X[:, i]))) * (b - a) + a == 0 or ((max(X[:, i]) - min(X[:, i]))) * (b - a) + a == ""):
            Normalized_X[:, i] = 0
            continue
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X