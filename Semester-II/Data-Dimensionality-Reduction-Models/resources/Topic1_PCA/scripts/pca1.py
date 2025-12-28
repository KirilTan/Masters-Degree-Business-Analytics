# -*- coding: utf-8 -*-
"""

@author: Nikola
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

#Apply PCA with two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#Create a dataframe woth the PCA results
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['target'] = y

#Visualize the results
plt.figure(figsize=(8,6))
for label, color in zip([0, 1, 2], ['red', 'green', 'blue']):
    subset = pca_df[pca_df['target'] == label]
    plt.scatter(subset['PC1'], subset['PC2'], c=color, label=wine.target_names[label], alpha=0.6)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% explained variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% explained variance)")
plt.title("PCA on Wine Dataset (2 components)")
plt.legend()
plt.show()

#Sum of explained variance
print("Sum of explained variance of the components:", pca.explained_variance_ratio_.sum())

