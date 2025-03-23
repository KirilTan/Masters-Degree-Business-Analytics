# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:36:43 2025

@author: Ivan
"""

# Example_framinghamRF.py


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

print('Random Forest ')
from sklearn.ensemble import RandomForestClassifier
logreg = RandomForestClassifier(random_state=122)
        
              
logreg=RandomForestClassifier(n_estimators=100, max_depth=3, 
                              class_weight ='balanced', random_state=762) #762



logreg.fit(X_train,y_train)
print('Train set score :',logreg.score(X_train, y_train))
print('Test  set score :',logreg.score(X_test, y_test))

#Confusiton matrix and clasification report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = logreg.predict(X_test)
print ('\n confussion matrix for the test set :\n',confusion_matrix(y_test,
                                                                    y_pred))
print ('\n classification report for the test set :\n',
       classification_report(y_test, y_pred))

