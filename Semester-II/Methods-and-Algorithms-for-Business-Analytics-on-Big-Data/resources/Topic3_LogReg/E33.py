# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:49:42 2024

@author: Ivan
"""

# Examp 3.3 pima-indians-diabetes.csv data set
# KFold approach
import pandas
#load the data
#Load the data into python
dataset = pandas.read_csv("pima-indians-diabetes.csv", header=None)
#print(dataset)
# separate the data from the target attributes
X = dataset.values[:,0:8]
y = dataset.values[:,8]
# LogisticRegression


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=5000)

from sklearn.model_selection import KFold
def k_fold_split(X, y):
    kf = KFold(n_splits=5, shuffle=True,random_state=42)# 42
    k=0
    sm=0
    for train_index, test_index in kf.split(X, y):
         logreg .fit(X[train_index],y[train_index])
         score_test = logreg.score(X[test_index], y[test_index])
         if sm < score_test :
             sm=score_test
             train_minindex = train_index
             test_minindex = test_index
    
         k+=1
         
    return X[train_minindex], X[test_minindex], y[train_minindex], y[test_minindex]

X_train, X_test, y_train, y_test = k_fold_split(X, y)

logreg.fit(X_train,y_train)
score_test = logreg.score(X_train,y_train)
print (score_test)

from sklearn.metrics import confusion_matrix, classification_report
# make predictions
y_pred= logreg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('confusion_matrix for test set')
print(cm)
print
print('classification_report for test set')
print (classification_report(y_test, y_pred))

"""
y_pred= logreg.predict(X)
from sklearn.metrics import confusion_matrix
print ('\n confusion matrix for the test set :\n',
       confusion_matrix(y,y_pred))
from sklearn.metrics import classification_report
print ('\n classification report for the test set :\n',
       classification_report(y, y_pred)) 
"""