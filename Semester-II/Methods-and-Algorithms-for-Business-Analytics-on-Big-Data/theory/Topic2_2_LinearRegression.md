# Topic 2 – Linear Regression (Part 2: Validation, Generalization & Cross-Validation)

This document extends **Topic 2 – Part 1** by focusing on **model validation and generalization**.

The goal is to understand **how to evaluate linear regression models correctly** and **why validation is essential** before using a model for business decision-making.

---

## 1. Supervised Learning Context

Linear regression belongs to the class of **supervised learning methods**.

In supervised learning:
- the dataset contains **input variables** (`X`)
- and a known **target variable** (`y`)

The model learns a relationship between `X` and `y` using labeled data and is then expected to generalize to unseen observations.

---

## 2. Why Validation Is Necessary

Evaluating a model only on the data used for training can lead to **overly optimistic results**.

A model may:
- fit the training data very well
- but perform poorly on new, unseen data

This phenomenon is known as **overfitting**.

Validation techniques are used to estimate how well a model will perform in practice.

---

## 3. Train/Test Split

### 3.1 Concept

The most common validation approach is to split the dataset into two disjoint subsets:

- **Training set** – used to train the model
- **Test set** – used only for evaluation

The model is trained on the training set and evaluated on the test set.

---

### 3.2 Purpose

The train/test split allows us to:
- simulate performance on unseen data
- detect overfitting
- compare different models objectively

A good model should perform well on both training and test data.

---

## 4. Overfitting and Underfitting

Two common modeling problems are:

- **Overfitting** – the model learns noise and peculiarities of the training data
- **Underfitting** – the model is too simple to capture the underlying relationship

Indicators:
- high training score + low test score → overfitting
- low training score + low test score → underfitting

---

## 5. Evaluation Metrics in Regression

For linear regression, the most commonly used evaluation metric is **R²**.

R² on the **test set** measures:
- how much variance in the target variable is explained by the model on unseen data

Test-set R² is a more reliable indicator of real-world performance than training R².

---

## 6. Limitations of a Single Train/Test Split

A single train/test split has limitations:
- results depend on how the data is split
- different splits may produce different scores

This variability motivates the use of **cross-validation**.

---

## 7. Cross-Validation

Cross-validation is a validation technique that uses **multiple train/test splits**.

The most common approach is **K-Fold Cross-Validation**.

---

## 8. K-Fold Cross-Validation

### 8.1 Concept

In K-Fold Cross-Validation:
- the dataset is divided into `K` equal parts (folds)
- the model is trained `K` times
- each fold is used once as a test set

The final performance is computed as the **average score** across all folds.

---

### 8.2 Purpose

K-Fold Cross-Validation:
- reduces dependence on a single split
- provides a more stable estimate of performance
- uses data more efficiently

It is especially useful when datasets are not very large.

---

## 9. Model Selection Using Cross-Validation

Cross-validation allows comparison between:
- different models
- different model parameters

The preferred model is typically the one with:
- higher average validation score
- lower variability across folds

This supports informed and defensible model selection.

---

## 10. Relation to Business Analytics

In business analytics, validation is critical because:
- decisions are made based on model outputs
- poor generalization can lead to costly mistakes

Validation techniques ensure that models are:
- reliable
- robust
- suitable for decision support

---

## 11. Summary

- Linear regression is a supervised learning method
- Training performance alone is insufficient
- Train/test split estimates generalization performance
- Overfitting and underfitting must be avoided
- Cross-validation provides a more robust evaluation
- Validation is essential for trustworthy business analytics

---

➡️ **Next Step:** Applying linear regression with proper validation in Jupyter notebooks and scripts.

