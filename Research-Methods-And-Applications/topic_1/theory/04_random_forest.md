
# 🌲 Random Forest – Theory & Application

**Random Forest** is an ensemble machine learning algorithm that builds multiple decision trees and merges their results to improve performance and reduce overfitting.

It is widely used for **classification** and **regression**, especially on structured/tabular data.

---

## 🧠 How It Works

Random Forest combines **many Decision Trees** using two key techniques:

### 1️⃣ Bootstrapping
- For each tree, it samples the training data **with replacement**.
- Each tree is trained on a **random subset of rows**, so it sees slightly different data.

### 2️⃣ Random Feature Selection
- At each split in the tree, only a **random subset of features** is considered.
- This ensures **decorrelation** among trees.

### 🔁 Final Step: Voting or Averaging
- **Classification**: each tree votes; majority vote wins.
- **Regression**: average of all tree predictions.

---

## 📊 Visual Summary

```
Full Dataset
    └── Bootstrap Samples (random rows)
            ├── Tree 1 (random features)
            ├── Tree 2 (random features)
            └── Tree N (random features)
                     ↓
              Majority Voting
                     ↓
             Final Prediction
```

---

## ✅ Strengths

- Reduces overfitting (compared to a single tree)
- Handles large datasets and high-dimensional data well
- Works well even with missing data or noisy inputs
- Can handle imbalanced data with `class_weight='balanced'`

## ⚠️ Weaknesses

- Less interpretable than a single decision tree
- Can be computationally expensive (many trees)
- Slower to predict than simpler models

---

## 📌 Applications

- Medical diagnostics (e.g., disease classification)
- Credit scoring & fraud detection
- Customer churn prediction
- Water quality prediction

---

## 🔗 Further Reading

- 📘 [GFG: Random Forest in ML](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/)
- 📘 [GFG: Random Forest in Scikit-learn](https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/)
- 📘 [Analytics Vidhya: Random Forest Guide](https://www.analyticsvidhya.com/blog/2021/08/decision-tree-algorithm/)
