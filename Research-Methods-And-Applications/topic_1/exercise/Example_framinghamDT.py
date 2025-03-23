# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 16:27:24 2023

@author: Ivan
"""


import pandas as pd
# Load the dataset
data = pd.read_csv("framingham.csv")


# намиране на редове с NAN стойности 
data.isnull().sum()
dataset=data.dropna()

X = dataset.values[:,0:15]
y = dataset.values[:,15]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=122)

print('Модел Дърво на решенията (Decition Tree) ')
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
#
# https://scikit-learn.org/stable/modules/tree.html 
# 
print()
from sklearn.tree import DecisionTreeClassifier
logreg = DecisionTreeClassifier(random_state=22)

#logreg = DecisionTreeClassifier( class_weight='balanced', random_state=366 )

#print('DecisionTree :')
logreg.fit(X_train,y_train)
print('test set score :',logreg.score(X_test, y_test))
#Confusiton matrix and clasification report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = logreg.predict(X_test)
print ('\n confussion matrix for the test set :\n',confusion_matrix(y_test,
                                                                    y_pred))
print ('\n classification report for the test set :\n',
       classification_report(y_test, y_pred))