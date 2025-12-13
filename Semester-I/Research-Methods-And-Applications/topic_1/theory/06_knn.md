# 🧠 06 k-Nearest Neighbors (k-NN)

# 📍 k-Nearest Neighbors – Theory & Application

**k-NN** is a simple and intuitive **supervised learning** algorithm for **classification** and **regression**.
It is **instance-based (lazy learning)**: it stores the training data and makes predictions at query time.

---

## 🔍 How It Works

- For a new point **x**, find the **k closest** training points using a distance metric.
- **Classification:** assign the **majority class** among the k neighbors.
- **Regression:** predict the **average** (or weighted average) of the neighbors’ targets.

**Common distances:** Euclidean, Manhattan, Minkowski.  
**Tip:** Scale features (e.g., `StandardScaler`) so distances are meaningful.

---

## ⚙️ Key Choices

- **k (neighbors):**  
  - Small k → low bias, high variance (can overfit)  
  - Large k → higher bias, lower variance (can underfit)
- **Distance metric:** Match to data geometry; try Euclidean first.
- **Weighting:** `weights='distance'` gives closer neighbors more influence.
- **Tie-breaking:** Ensure deterministic behavior with fixed `random_state` (for CV splits).

---

## ✅ Strengths

- Very **simple** and **non-parametric**
- Naturally handles **non-linear** decision boundaries
- Works well with **small** datasets and **well-separated** classes

## ⚠️ Weaknesses

- **Slow at prediction** (needs all data)
- **Sensitive to feature scaling** and **irrelevant features**
- **Struggles in high dimensions** (curse of dimensionality)

---

## 📊 Practical Workflow

1. **Scale** features (standardize or normalize).  
2. **Choose k** via cross-validation (grid search over, e.g., 1–31 odd numbers).  
3. Try `weights=['uniform','distance']`.  
4. Evaluate with **F1/Recall** if classes are imbalanced (not just accuracy).  
5. Use **StratifiedKFold** for classification CV.

---

## 📌 Use Cases

- Image/handwriting recognition  
- Recommendation & similarity search  
- Tabular classification (e.g., lab results, water quality)

---

## 🔗 Further Reading

- 📘 [GFG: k-NN Algorithm](https://www.geeksforgeeks.org/k-nearest-neighbours/)
- 📘 [Scikit-learn: KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
