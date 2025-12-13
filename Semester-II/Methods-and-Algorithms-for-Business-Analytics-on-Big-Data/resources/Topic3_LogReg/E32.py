# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 10:22:36 2023

@author: Ivan
"""


# Examp 3.2 pima-indians-diabetes.csv data set
# KFold approach
import pandas
#load the data
url = "pima-indians-diabetes.csv"
#Load the data into python
dataset = pandas.read_csv(url,header=None)
#print(dataset)
# separate the data from the target attributes
X = dataset.values[:,0:8]
y = dataset.values[:,8]

import collections
counter = collections.Counter(y)
counter  #Counter({1: 268, 0: 500})


# LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=5000)


from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle=True)
sm=0
for train_index, test_index in kf.split(X):
    #print('k=',k)
    logreg.fit(X[train_index],y[train_index])
    score_test = logreg.score(X[test_index], y[test_index])
    print (score_test)
    if sm < score_test :
        #print('k=',k)
        sm=score_test
        train_minindex = train_index
        test_minindex = test_index
    print() 
    
logreg.fit(X[train_minindex ],y[train_minindex ])
score_test = logreg.score(X[test_minindex], y[test_minindex])
print (score_test)


X_test = X[test_minindex]
y_test = y[test_minindex]
from sklearn.metrics import confusion_matrix, classification_report
# make predictions
y_pred= logreg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('confusion_matrix for test set')
print(cm)
print
print('classification_report for test set')
print (classification_report(y_test, y_pred))










