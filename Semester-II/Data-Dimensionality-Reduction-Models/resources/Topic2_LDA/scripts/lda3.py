# -*- coding: utf-8 -*-
"""

@author: Nikola
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# 2. Load data
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2.1. Create a dataframe
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = [target_names[i] for i in y]


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. LDA model fit
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# 6. LDA (LD1 vs LD2)
plt.figure(figsize=(8, 6))
for label, color, name in zip([0, 1, 2], ['red', 'green', 'blue'], target_names):
    plt.scatter(X_train_lda[y_train == label, 0], 
                X_train_lda[y_train == label, 1], 
                alpha=0.7, label=name, color=color, edgecolor='k')
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDA Projection of Iris Dataset (Train Set)")
plt.legend()
plt.grid(True)
plt.show()

# 7. Model accuracy
y_pred = lda.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=target_names))
