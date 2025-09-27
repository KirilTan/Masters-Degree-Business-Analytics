# Machine Learning Classification: kNN, Logistic Regression, and Decision Trees

## 1. Introduction
This lecture covers fundamental machine learning classification techniques using Python and **scikit-learn**. We will explore:
- k-Nearest Neighbors (kNN)
- Logistic Regression
- Decision Trees
- Data preprocessing and handling imbalanced data
- Model evaluation techniques

## 2. Libraries Used
The main libraries used in the scripts are:
- `pandas`: For data manipulation
- `sklearn.model_selection.train_test_split`: To split data into training and testing sets
- `sklearn.neighbors.KNeighborsClassifier`: k-Nearest Neighbors algorithm
- `sklearn.linear_model.LogisticRegression`: Logistic Regression model
- `sklearn.tree.DecisionTreeClassifier`: Decision Tree model
- `sklearn.metrics.confusion_matrix`: Confusion matrix for performance evaluation
- `sklearn.metrics.classification_report`: Detailed classification metrics

---

## 3. Data Processing
### 3.1 Loading and Cleaning Data
Each script loads a dataset (CSV file) and processes it:
- **Handling missing values**:
  ```python
  dataset.fillna(value=dataset["column"].mean(), inplace=True)
  ```
- **Splitting data into features (X) and target (y)**:
  ```python
  X = dataset.values[:, :9]  # First 9 columns (features)
  y = dataset.values[:, 9]   # Last column (target)
  ```
- **Splitting into training and testing sets**:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
  ```
  - `test_size=0.3` means 30% of data is used for testing.

---

## 4. Machine Learning Models

### 4.1 k-Nearest Neighbors (kNN)
kNN is a **non-parametric** algorithm that classifies data points based on their closest neighbors.
#### **Implementation**:
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```
- `n_neighbors=5`: Classifies based on 5 nearest neighbors.
- `fit(X_train, y_train)`: Trains the model.

### 4.2 Logistic Regression
Logistic Regression is a **linear classification algorithm** for binary classification problems.
#### **Implementation**:
```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)
```
- `solver='lbfgs'`: Optimization algorithm for training.
- `max_iter=1000`: Ensures convergence.
- `class_weight='balanced'`: Adjusts weights for imbalanced data.

### 4.3 Decision Tree Classifier
Decision Trees recursively split data based on the most informative features.
#### **Implementation**:
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
```
- Uses default settings.

---

## 5. Model Evaluation
### 5.1 Accuracy Score
```python
print('Test set score:', model.score(X_test, y_test))
```
### 5.2 Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
```
### 5.3 Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```
Provides **precision, recall, and F1-score**.

---

## 6. Handling Imbalanced Data
### 6.1 Identifying Class Imbalance
```python
dM1 = dM[dM['TenYearCHD'] == 0]  # Majority class
dM2 = dM[dM['TenYearCHD'] == 1]  # Minority class
```
### 6.2 Undersampling the Majority Class
```python
dMSmall_train, dMSmall_test = train_test_split(dMSmall, test_size=0.40)
```
Ensures equal class distribution in training.

---

## 7. Summary
1. **Load and preprocess data** (handle missing values, split into train/test sets).
2. **Train models**: kNN, Logistic Regression, Decision Tree.
3. **Evaluate models**: Accuracy, confusion matrix, classification report.
4. **Handle imbalanced data**: Balance class distribution.

Would you like to see an example dataset or expand on any section? 🚀

