# -*- coding: utf-8 -*-
"""


@author: Nikola
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

data = load_boston()
boston = pd.DataFrame(data.data, columns=data.feature_names)
boston['MEDV']=data.target

y = boston['MEDV']
X = boston.iloc[:, :13]

# scale
X = StandardScaler().fit_transform(X)

# PCA
pca = PCA()
x_pca = pca.fit_transform(X)
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()

pca = PCA(n_components=6)
X = pca.fit_transform(X)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
print('explained variance ratio (first six components): %s' 
      % str(sum(pca.explained_variance_ratio_))) 

fig = plt.figure(figsize=(12,6))
#fig.add_subplot(1,2,1)
plt.bar(np.arange(pca.n_components_), 100 * pca.explained_variance_ratio_)
plt.title('Relative information content of PCA components')
plt.xlabel("PCA component number")
plt.ylabel("PCA component variance %")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

# fit the model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# model evaluation for training set
y_train_predict = linreg.predict(X_train)
score = linreg.score(X_train,y_train)
mse = ((mean_squared_error(y_train, y_train_predict)))

print("The model performance for training set")
print("--------------------------------------")
print('Score is {}'.format(score))
print('MSE is {}'.format(mse))
print("\n")

# model evaluation for testing set
score = linreg.score(X_test,y_test)
y_test_predict = linreg.predict(X_test)
mse = ((mean_squared_error(y_test, y_test_predict)))

print("The model performance for testing set")
print("--------------------------------------")
print('Score is {}'.format(score))
print('MSE is {}'.format(mse))

