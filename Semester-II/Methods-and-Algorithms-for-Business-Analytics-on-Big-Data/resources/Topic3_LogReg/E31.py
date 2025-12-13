# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:44:39 2023

@author: Ivan
"""

# Examp 3.1 pima-indians-diabetes.csv data set
from numpy import loadtxt
# download the file
raw_data = 'pima-indians-diabetes.csv'
# load the CSV file as a numpy matrix
datasetD = loadtxt(raw_data, delimiter=',')
# separate the data from the target attributes
X = datasetD[:,0:8]
y = datasetD[:,8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# LogisticRegression





from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=5000)
#  ‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’ 
#logreg = LogisticRegression(max_iter=1000)
# 10000 НЕ СА ДОСТАТЪЧНИ
logreg.fit(X_train,y_train)
print('test set score :',logreg.score(X_test, y_test))




from sklearn import metrics
# make predictions
expected = y
predicted = logreg.predict(X)
# summarize the fit of the model
print(metrics.confusion_matrix(expected, predicted))
print(metrics.classification_report(expected, predicted))
# 
#from sklearn import metrics
# make predictions
expected = y_test
predicted = logreg.predict(X_test)
# summarize the fit of the model
print(metrics.confusion_matrix(expected, predicted))
print(metrics.classification_report(expected, predicted))



