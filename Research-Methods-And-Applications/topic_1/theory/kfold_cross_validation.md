
# 🔁 K-Fold Cross-Validation – Theory & Application

**K-Fold Cross-Validation** is a model evaluation technique used to assess how well your machine learning model generalizes to unseen data.

Rather than relying on a single train-test split, it splits the data into **K equal parts (folds)** and performs **K rounds** of training and validation.

---

## 🧠 How It Works

1. Split the dataset into **K equal-sized folds** (e.g., K=5)
2. For each round (fold):
   - Use **K-1 folds** for training
   - Use **1 fold** for testing (rotates each time)
3. Average the performance metrics (accuracy, F1, etc.) across all K rounds

```
K = 5 example:

Fold 1:  [TEST]  | Train on Folds 2-5
Fold 2:  [TEST]  | Train on Folds 1,3-5
Fold 3:  [TEST]  | Train on Folds 1-2,4-5
Fold 4:  [TEST]  | Train on Folds 1-3,5
Fold 5:  [TEST]  | Train on Folds 1-4

Final score = average of all 5 test scores
```

---

## ✅ Strengths

- Makes **better use of limited data**
- Provides **more reliable model evaluation**
- Reduces variance from lucky/unlucky single splits
- Useful when tuning hyperparameters or comparing models

---

## ⚠️ Weaknesses

- More computationally expensive (train K models instead of 1)
- Needs careful setup to **preserve class balance** (use `StratifiedKFold` for classification)

---

## 📌 Example

- Logistic Regression is trained on 5 different train/test splits
- The best-performing model is selected based on test scores
- Balancing (`class_weight='balanced'`) is used to address class imbalance

---

## 🧠 Pro Tip

For classification tasks with imbalanced classes, use:

```python
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 🔗 Further Reading

- 📘 [Scikit-learn Docs – KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- 📘 [Stratified K-Fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

