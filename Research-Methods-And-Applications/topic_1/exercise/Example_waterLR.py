# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:47:09 2023

@author: Ivan
"""

# # Example_waterLR
import pandas as pd
dataset = pd.read_csv('water_potability.csv')
#Заменям празните стойности
dataset["ph"].fillna(value=dataset["ph"].mean(), inplace=True)
dataset["Sulfate"].fillna(value=dataset["Sulfate"].mean(), inplace=True)
dataset["Trihalomethanes"].fillna(value=dataset["Trihalomethanes"].mean(), inplace=True)
# dataset['Potability'].value_counts()
X = dataset.values[:, :9]
y = dataset.values[:, 9]


print('Модел Логистична регресия ')
print()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=1000) 
logreg = LogisticRegression(solver='lbfgs', max_iter=1000,  class_weight='balanced')  

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


