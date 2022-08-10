from sklearn.cluster import MeanShift, KMeans
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


def k_mean(X, centers):
    km = KMeans(n_clusters=len(centers),
                init=centers,
                max_iter=300,
                n_init=1,
                algorithm='elkan',
                tol=0.001)
    return km.fit(X)

mean_shifted = mean_shift(X_test)
centers = mean_shifted.cluster_centers_
predict_test = k_mean(X_test,centers).predict(X_test)
print('Gini: ',metrics.adjusted_rand_score(Label_test,predict_test)*100)
print('Purity: ',purity_score(Label_test,predict_test)*100)

print('----------------------------------------------')
dataset_valid = pd.read_csv('features.csv')
labels_valid = pd.read_csv('Labels.csv')
X_valid = dataset_valid.iloc[:,:]
Label_valid = labels_valid.iloc[:,0]
mean_shifted2 = mean_shift(X_test)
centers2 = mean_shifted.cluster_centers_
predict_valid = k_mean(X_valid,centers2).predict(X_valid)
print('gini: ',metrics.adjusted_rand_score(Label_valid,predict_valid)*100)
print('purity: ',purity_score(Label_valid,predict_valid)*100)