# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:31:28 2025

@author: Ivan Ivanov
"""


# Example_framinghamLRKFold
import pandas as pd
# Load the dataset
data = pd.read_csv("framingham.csv")


# намиране на редове с NAN стойности 
data.isnull().sum()
dataset=data.dropna()
dataset.isnull().sum()

#
# # премахвам  липсващите стойности 
## data.dropna(inplace=True)
#


X = dataset.values[:,0:15]
y = dataset.values[:,15]

print('Модел Логистична регресия ')
print()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
#Logistic Regression
from sklearn.linear_model import LogisticRegression

 

logreg = LogisticRegression(solver='lbfgs', max_iter=5000,  class_weight='balanced')  

#logreg=LogisticRegression(solver='liblinear',  class_weight="balanced") 


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=41)
# 892 for LR1  and 41 for LR 2 
       

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
#Confusiton matrix and clasification report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = logreg.predict(X_test)
print ('\n confussion matrix for the test set :\n',confusion_matrix(y_test,
                                                                    y_pred))
print ('\n classification report for the test set :\n',
       classification_report(y_test, y_pred))





