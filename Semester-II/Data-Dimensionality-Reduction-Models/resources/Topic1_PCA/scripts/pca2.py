# -*- coding: utf-8 -*-
"""


@author: User
"""

#Load libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Load data
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="target")

# Show the shape of the dataset
print(X.shape)

#Scale data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Apply PCA without selecting the number of components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)


#Calculate the cumulative explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

#Visualize the results with a Scree plot
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance)+1), explained_variance, 'o-', label='Variance per component')
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 's--', label='Cumulative variance')
plt.xlabel('Number of principal components')
plt.ylabel('Explained variance')
plt.title('PCA on Wine Dataset')
plt.xticks(range(1, len(explained_variance)+1))
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()
