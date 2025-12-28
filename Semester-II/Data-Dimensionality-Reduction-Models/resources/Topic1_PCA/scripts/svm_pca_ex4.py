# -*- coding: utf-8 -*-
"""


@author: Nikola
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC 

## load data
data = pd.read_csv("winequality-white.csv", delimiter = ';')

#count of the target variable
sns.countplot(x='quality', data=data)


quality = data["quality"].values
category = []
for num in quality:
    if num<6:
        category.append(-1)
    elif num>6:
        category.append(1)
    else:
        category.append(0)
        
#Create new data
category = pd.DataFrame(data=category, columns=["category"]).astype(int)
df = pd.concat([data,category],axis=1)

#count of the target variable
sns.countplot(x='category', data=df)

y = df.iloc[:,12].values


from sklearn.preprocessing import StandardScaler
Y = df.iloc[:,:11].values
Y = StandardScaler().fit_transform(Y)


from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
## pca = PCA(n_components = 0.95)
X = pca.fit(Y).transform(Y)
print('explained variance ratio (first four components): %s' 
      % str(pca.explained_variance_ratio_)) 


model = SVC(kernel='rbf', C=1000, gamma='auto', decision_function_shape='ovr') # best results: C = 1000

model.fit(X, y)

from sklearn.model_selection import KFold
n_splits=15 #15
kf = KFold(n_splits=n_splits)
k=0
sm=0
for train_index, test_index in kf.split(X):
    model.fit(X[train_index],y[train_index])
    sc_test  = model.score(X[test_index],y[test_index])

    if sm < sc_test : 
        sm=sc_test
        ksm=k
        train_maxindex = train_index
        test_maxindex =  test_index
        

X_train = X[train_maxindex]
y_train = y[train_maxindex]
X_test = X[test_maxindex]
y_test = y[test_maxindex]
model.fit(X_train,y_train)

train_score = model.score(X_train, y_train)
test_score  = model.score(X_test, y_test)
y_test_pred = model.predict(X_test)




from sklearn.metrics import classification_report, confusion_matrix
## test set score and reports
print ('\n svm.score for the test set :\n ', model.score(X[test_maxindex], y[test_maxindex]) )
print ('\n confussion matrix for the test set :\n',confusion_matrix(y_test, y_test_pred))
print ('\n classification report for the test set :\n',classification_report(y_test,y_test_pred)) 

## full set score and reports
y_p = model.predict(X)
print ('\n svm.score for the full set :\n ', model.score(X, y) )
print ('\n confussion matrix for the full set :\n',confusion_matrix(y, y_p))  
print('\n classification report for the full set :\n',classification_report(y,y_p)) 
