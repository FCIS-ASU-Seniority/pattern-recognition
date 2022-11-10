from sklearn.preprocessing import LabelEncoder
import numpy as np

def features_encoder(X, cols):
    for c in cols:
        lbl_enc = LabelEncoder()
        lbl_enc.fit(list(X[c].values))
        X[c] = lbl_enc.transform(list(X[c].values))
    return X

def feature_scaling(X, a, b):
    X = np.array(X)
    X_normalized = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        X_normalized[:, i] = ((X[:, i]-min(X[:, i])/(max(X[:, i])-min(X[:, i]))))*(b-a)+a
    return X_normalized