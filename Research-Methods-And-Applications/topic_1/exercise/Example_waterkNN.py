# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:50:26 2023

@author: Ivan
"""


# # Example_waterkNN
import pandas as pd
dataset = pd.read_csv('water_potability.csv')
#Заменям празните стойности
dataset["ph"].fillna(value=dataset["ph"].mean(), inplace=True)
dataset["Sulfate"].fillna(value=dataset["Sulfate"].mean(), inplace=True)
dataset["Trihalomethanes"].fillna(value=dataset["Trihalomethanes"].mean(), inplace=True)
# dataset['Potability'].value_counts()
X = dataset.values[:, :9]
y = dataset.values[:, 9]


#Метод на най-близкия съсед 
#разделяне на множеството от данни на случаен принцип чрез train_test_split
print('Модел на най-близките съседи')
print()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
knn.fit(X_train,y_train)

train_score = knn.score(X_train, y_train)
test_score  = knn.score(X_test, y_test)
print
print('knn','train_score',train_score,'test_score',test_score)
print

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = knn.predict(X_test)
print ('\n confussion matrix for the test set :\n',confusion_matrix(y_test,
                                                                    y_pred))
print ('\n classification report for the test set :\n',classification_report
       (y_test, y_pred))

