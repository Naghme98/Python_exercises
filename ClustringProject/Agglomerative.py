from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn import metrics
import numpy as np


# dataset_test = pd.read_csv('features_test.csv')
# labels_test = pd.read_csv('Labels_test.csv')
# X_test = dataset_test.iloc[:, :]
# Label_test = labels_test.iloc[:, 0]

def agglomerative_clustering(X, link):
    return AgglomerativeClustering(n_clusters=20, linkage=link).fit_predict(X)


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# for x in ['single', 'complete', 'average', 'ward']:
#     print('-------------- ', x, ' -----------------')
#     predict_test = agglomerative_clustering(X_test,x )
#     print('gini: ', metrics.adjusted_rand_score(Label_test, predict_test)*100)
#     print('purity: ', purity_score(Label_test, predict_test) * 100)


print('----------------------------------------------')

dataset_valid = pd.read_csv('features.csv')
labels_valid = pd.read_csv('Labels.csv')
X_valid = dataset_valid.iloc[:, :]
Label_valid = labels_valid.iloc[:, 0]
for x in ['complete', 'ward']:
    print('-------------- ', x, ' -----------------')
    predict_valid = agglomerative_clustering(X_valid,x )
    print('gini: ', metrics.adjusted_rand_score(Label_valid, predict_valid) * 100)
    print('purity: ', purity_score(Label_valid, predict_valid) * 100)
