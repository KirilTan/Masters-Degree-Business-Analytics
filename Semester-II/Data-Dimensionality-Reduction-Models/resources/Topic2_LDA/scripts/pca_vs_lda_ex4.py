# -*- coding: utf-8 -*-
"""


@author: Nikola
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
wine = pd.read_csv('winequalityred.csv')
wine.head()

wine['quality2'] = np.where(wine['quality']<=4,1, np.where(wine['quality']<=6,2,3))

X = wine.drop(columns=['Unnamed: 0','quality','quality2'])
y = wine['quality2']

scaler = StandardScaler()    #####################
X = scaler.fit_transform(X)

target_names = np.unique(y)
target_names
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components of PCA): %s'
 % str(pca.explained_variance_ratio_))
print('explained variance ratio (first two components of LDA): %s'
 % str(lda.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, target_names, target_names):
 plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
 label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of WINE dataset')
plt.figure()
for color, i, target_name in zip(colors, target_names, target_names):
 plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
 label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of WINE dataset')
plt.show()