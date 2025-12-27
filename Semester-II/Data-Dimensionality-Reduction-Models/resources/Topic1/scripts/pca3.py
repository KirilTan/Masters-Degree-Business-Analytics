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

#Show the shape of the dataset
print(X.shape)


#Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Apply PCA and keep 95% of the explained variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Number of components: {pca.n_components_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

#Create a new dataframe with the principal components
columns = [f"PC{i+1}" for i in range(pca.n_components_)]
pca_df = pd.DataFrame(X_pca, columns=columns)
pca_df["target"] = y


#Scree plot
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_variance)+1), explained_variance, 'o-', label='Variance per component')
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 's--', label='Cumulative variance')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold') 

plt.xlabel('# PC')
plt.ylabel('Explained variance')
plt.title('Scree Plot – PCA (95% explained variance)')
plt.xticks(range(1, len(explained_variance)+1))
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

#List of the explained variance
for i, var in enumerate(explained_variance, start=1):
    print(f"PC{i}: {var*100:.2f}%  (Cumulative: {cumulative_variance[i-1]*100:.2f}%)")
