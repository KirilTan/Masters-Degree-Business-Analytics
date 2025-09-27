# ⚖️ Class Imbalance – Theory & Techniques

**Class imbalance** occurs when the number of samples in one class greatly outweighs the number in another.

Example: In a medical dataset, 95% of patients might be healthy (class 0), while only 5% have a disease (class 1). A
model trained on this data might just learn to predict everyone as healthy and still achieve 95% accuracy - but it
completely fails to detect the minority class.

---

## 🚨 Why It's a Problem

- Standard accuracy becomes **misleading** (e.g., predicting only the majority class still gives high accuracy).
- Models tend to **ignore the minority class**, which is often the one we care about (e.g. fraud, cancer, etc.).
- Evaluation should be done with **precision, recall, F1**, not just accuracy.

---

## 🛠️ Techniques to Handle It

| Strategy                    | Description                                                             |
|-----------------------------|-------------------------------------------------------------------------|
| **Downsampling**            | Randomly remove samples from the majority class                         |
| **Oversampling**            | Duplicate or synthetically generate minority class samples (e.g. SMOTE) |
| **class_weight='balanced'** | Tell the model to penalize mistakes on the minority class more heavily  |
| **Equal Sample Training**   | Construct training sets with an equal number of samples from each class |


- In Logistic Regression and Decision Tree models, `class_weight='balanced'` can be used to automatically address
  imbalance.

---

## ✅ Best Practices

- Always check `value_counts()` on your target variable.
- Evaluate models using **recall**, **precision**, and **F1-score**, especially for the minority class.
- Use stratified sampling or balancing techniques **only on the training set** (not on the test set).

---

## 🔗 Further Reading

- 📘 [GFG: Dealing with Imbalanced Data](https://www.geeksforgeeks.org/how-to-handle-imbalanced-classes-in-machine-learning/)

