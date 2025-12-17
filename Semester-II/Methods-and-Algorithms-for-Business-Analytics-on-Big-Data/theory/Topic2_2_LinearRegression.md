# Topic 2 – Linear Regression (Part 2: Validation, Generalization & Cross-Validation)

---

## 1️⃣ Supervised Learning Context

Linear regression is part of **supervised machine learning**.

In supervised learning:

* input variables (`X`) are known
* the target variable (`y`) is known

The goal is to learn a relationship between `X` and `y` that can be applied to **new, unseen data**.

---

## 2️⃣ Why Validation Is Necessary

Evaluating a model only on the data used for training can be misleading.

A model may:

* fit the training data very well
* but perform poorly on unseen data

This problem is known as **overfitting**.

Validation techniques are used to estimate how well a model generalizes beyond the training data.

---

## 3️⃣ Train/Test Split

### Concept

A common validation approach is to split the dataset into two disjoint subsets:

* **Training set** – used to train the model
* **Test set** – used only for evaluation

The model is trained on the training set and evaluated on the test set.

---

## 4️⃣ Overfitting and Underfitting

Two typical modeling problems are:

* **Overfitting** – the model learns noise and specific patterns of the training data
* **Underfitting** – the model is too simple to capture the underlying relationship

Indicators:

* high training score + low test score → overfitting
* low training score + low test score → underfitting

---

## 5️⃣ Evaluation Metrics in Regression

For linear regression, the most commonly used evaluation metric is **R² (coefficient of determination)**.

When computed on the **test set**, R² measures:

* how much variance in `y` is explained by the model on unseen data

Test-set R² is more reliable than training-set R² for assessing real-world performance.

---

## 6️⃣ Limitations of a Single Train/Test Split

A single train/test split has limitations:

* results depend on the chosen split
* different splits may lead to different evaluation scores

This variability motivates the use of **cross-validation**.

---

## 7️⃣ Cross-Validation

Cross-validation is a validation technique based on **multiple train/test splits**.

Instead of evaluating the model once, the model is evaluated repeatedly on different subsets of the data.

---

## 8️⃣ K-Fold Cross-Validation

In **K-Fold Cross-Validation**:

* the dataset is divided into `K` approximately equal folds
* the model is trained `K` times
* each fold is used once as a test set

The final performance is computed as the **average score across all folds**.

---

## 9️⃣ Model Selection Using Cross-Validation

Cross-validation allows comparison between:

* different models
* different model parameters

The preferred model is typically the one with:

* higher average validation score
* more stable performance across folds

This leads to more reliable model selection.

---

## 1️⃣0️⃣ Relation to Business Analytics

In business analytics, validation is critical because:

* models support real decisions
* poor generalization can lead to incorrect or costly decisions

Validation ensures that analytical models are:

* reliable
* robust
* suitable for decision support

---

## 🔑 One-Sentence Explanation

> Validation techniques estimate how well a regression model will perform on unseen data and help prevent overfitting.

---

## 🔗 References & Further Reading
- 📘 [scikit-learn Documentation – Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- 📘 [GeeksForGeeks – Supervised Machine Learning](https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/)
- 📘 [GeeksForGeeks – K-Fold Cross-Validation](https://www.geeksforgeeks.org/machine-learning/k-fold-cross-validation-in-machine-learning/)
