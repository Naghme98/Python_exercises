from sklearn.cluster import MeanShift
import pandas as pd
from sklearn import metrics
import numpy as np


dataset_test = pd.read_csv('features_test.csv')
labels_test = pd.read_csv('Labels_test.csv')
X_test = dataset_test.iloc[:, :]
Label_test = labels_test.iloc[:, 0]


def mean_shift(X):
    return MeanShift(bandwidth=15).fit(X)


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


predict_test = mean_shift(X_test).predict(X_test)
print('gini: ',metrics.adjusted_rand_score(Label_test, predict_test)*100)
print('Purity: ',purity_score(Label_test,predict_test)*100)

print('-----------------------in ghesmate kheyli run kardanesh tool mikeshe-----------------------')
dataset_valid = pd.read_csv('features.csv')
labels_valid = pd.read_csv('Labels.csv')
X_valid = dataset_valid.iloc[:, :]
Label_valid = labels_valid.iloc[:, 0]
predict_valid = mean_shift(X_valid).predict(X_valid)
print('gini:',metrics.adjusted_rand_score(Label_valid, predict_valid)*100)
print('Purity: ',purity_score(Label_valid,predict_valid)*100)
