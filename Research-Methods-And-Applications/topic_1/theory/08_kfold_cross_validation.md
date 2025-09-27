# 🧠 08 K-Fold Cross-Validation

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

## 📌 Example Snippet (sklearn)

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

X, y = ...  # features and labels

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(class_weight='balanced', max_iter=1000)

scores = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
print("F1 (macro) per fold:", scores)
print("Mean F1 (macro):", scores.mean())
```

---

## 🧠 Pro Tips

- For classification with imbalance: **use `StratifiedKFold`**
- Report **mean ± std** of metrics across folds
- Keep a **hold-out test set** if you will tune hyperparameters heavily

---

## 🔗 Further Reading

- 📘 [Scikit-learn: KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- 📘 [Scikit-learn: StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- 📘 [GFG: K-Fold CV](https://www.geeksforgeeks.org/machine-learning/k-fold-cross-validation-in-machine-learning/)
