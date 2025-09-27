
# 🌳 Decision Tree – Theory & Application

A **Decision Tree** is a supervised machine learning algorithm used for both **classification** and **regression**, though it is more commonly used for classification tasks.

It models decisions as a **tree-like structure**, where each internal node represents a **question/condition**, each branch is an **answer (yes/no)**, and each leaf node represents an **outcome or class label**.

---

## 🧠 How It Works

The tree is built by asking a series of **if/else** questions based on feature values. For example:

```
Is Age > 30?
├── Yes → Is Salary > 50k?
│   ├── Yes → Approve Loan
│   └── No  → Reject Loan
└── No  → Reject Loan
```

The goal is to split the data in such a way that each branch increases **purity** (data becomes more homogenous in class).

---

## ⚙️ Key Concepts

- **Root Node**: The top-most decision node.
- **Leaf Node**: Final decision/class.
- **Splitting**: Dividing the dataset based on a feature.
- **Impurity Measures**:
  - **Gini Index**
  - **Entropy & Information Gain**
- **Stopping Criteria**: Tree stops growing when:
  - Max depth is reached
  - Minimum samples per node is met
  - Nodes are "pure" (only one class)

---

## ✅ Strengths

- Easy to understand and visualize
- Handles both numerical and categorical data
- Requires little data preprocessing (no need for normalization or scaling)
- Works well with rule-based problems

## ⚠️ Weaknesses

- **Overfitting**: Deep trees can memorize training data
- **Instability**: Small changes in data can lead to different trees
- Not good with continuous decision boundaries (no "grey area")
- Biased toward features with more levels

---

## 📊 Use Cases

- Loan approval systems
- Medical diagnosis (yes/no decisions)
- Risk assessment tools
- Rule-based classification systems

---

## 🔗 Further Reading

- 📘 [GFG: Decision Tree Introduction](https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/)
- 📘 [GFG: Implementation in Python](https://www.geeksforgeeks.org/machine-learning/decision-tree-implementation-python/)
- 📘 [Analytics Vidhya: Decision Tree Algorithm](https://www.analyticsvidhya.com/blog/2021/08/decision-tree-algorithm/)
