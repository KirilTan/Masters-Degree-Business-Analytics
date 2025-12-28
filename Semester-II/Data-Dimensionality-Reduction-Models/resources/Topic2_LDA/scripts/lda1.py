# -*- coding: utf-8 -*-
"""


@author: User
"""


# 1. Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# 2. Define the dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=10,
    n_redundant=0,
    n_classes=2,
    random_state=7
)


# 3. Train/test split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 4. StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Define and train the model
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# 6. Test accuracy
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.3f}")

# 7. Cross validation
cv_scores = cross_val_score(model, X, y, cv=10)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 8. Clasify new observation
new_row = [0.12777556, -3.64400522, -2.23268854, -1.82114386, 1.75466361,
           0.1243966, 1.03397657, 2.35822076, 1.01001752, 0.56768485]
new_row_scaled = scaler.transform([new_row])
yhat = model.predict(new_row_scaled)
print(f"Predicted Class for new row: {yhat[0]}")


# 9. LDA viz
X_lda = model.transform(X_train)

plt.figure(figsize=(8, 6))
for label in np.unique(y_train):
    plt.hist(X_lda[y_train == label, 0], bins=30, alpha=0.6, label=f"Class {label}")
plt.xlabel("LD1")
plt.title("LDA Projection (1D) – Two-Class Case")
plt.legend()
plt.grid(True)
plt.show()