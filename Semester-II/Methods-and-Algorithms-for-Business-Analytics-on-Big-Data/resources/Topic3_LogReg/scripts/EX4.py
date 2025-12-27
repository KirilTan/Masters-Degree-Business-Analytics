# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:35:41 2024

@author: Ivan
"""


import pandas as pd
import numpy as np

data = pd.read_csv("../data/pima-indians-diabetes.csv", header=None)

#print(dataset)
# separate the data from the target attributes
Y = data.values[:,0:8]
z = data.values[:,8]

#Y = data.iloc[:,:8]
#z = data.iloc[:,8]

# Shuffle data:
np.random.seed(99)
permuted_indices = np.random.permutation(len(Y))

X=Y[permuted_indices]
y=z[permuted_indices]

import collections
counter = collections.Counter(y)
counter  #Counter({1: 268, 0: 500})


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs', max_iter=5000) 


from sklearn.model_selection import KFold
#дефинира процедура 
def k_fold_split(X, y):
    kf = KFold(n_splits=5, shuffle = True, random_state=42)
    k=0
    sm=0
    for train_index, test_index in kf.split(X, y):
         clf.fit(X[train_index],y[train_index])
         score_test = clf.score(X[test_index], y[test_index])
         print(score_test)
         if sm < score_test :
             sm=score_test
             train_minindex = train_index
             test_minindex = test_index
    
         k+=1
         print()
         
    return X[train_minindex], X[test_minindex], y[train_minindex], y[test_minindex]
    
X_train, X_test, y_train, y_test = k_fold_split(X, y)
clf.fit(X_train, y_train)



print('SCORE on train set: ', clf.score(X_train, y_train))
print('SCORE on test set: ', clf.score(X_test, y_test))
scTrain=clf.score(X_train, y_train)
scTest=clf.score(X_test, y_test)
print('Error: ', abs(scTrain-scTest) )

y_pred= clf.predict(X_test)
from sklearn.metrics import confusion_matrix
print ('\n confusion matrix for the test set :\n',
       confusion_matrix(y_test,y_pred))
from sklearn.metrics import classification_report
print ('\n classification report for the test set :\n',
       classification_report(y_test, y_pred)) 

"""
y_pred= clf.predict(X)
from sklearn.metrics import confusion_matrix
print ('\n confusion matrix for the test set :\n',
       confusion_matrix(y,y_pred))
from sklearn.metrics import classification_report
print ('\n classification report for the test set :\n',
       classification_report(y, y_pred)) 
"""
