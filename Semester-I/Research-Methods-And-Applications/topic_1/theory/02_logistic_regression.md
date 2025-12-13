# 🧠 02 Logistic Regression

# 📈 Logistic Regression – Theory & Application

**Logistic Regression** is a supervised learning algorithm used for **binary classification** problems. It predicts the
**probability** of the target variable belonging to a particular class.

---

## 🔢 Logistic vs Linear Regression

| Feature         | Linear Regression               | Logistic Regression                   |
|-----------------|---------------------------------|---------------------------------------|
| Output          | Continuous value                | Probability (0 to 1)                  |
| Target Variable | Any real number                 | Binary (0 or 1)                       |
| Function Used   | Linear function                 | Sigmoid function                      |
| Goal            | Minimize squared error          | Maximize likelihood                   |
| Use Case        | Predicting prices, trends, etc. | Classification: spam/not spam, yes/no |

---

## 🧮 Logistic Regression Formula

- **Sigmoid Function (Logit):**  
  σ(z) = 1 / (1 + e^(-z))

- **Prediction Formula:**  
  P(y = 1 | x) = 1 / (1 + e^-(b₀ + b₁x₁ + ... + bₙxₙ))

The output is a **probability** between 0 and 1, typically converted to class 0 or 1 using a **threshold** (commonly
0.5).

---

## 📊 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Correctness of predicted positives
- **Recall**: How well positives are detected
- **F1 Score**: Balance between precision and recall
- **Confusion Matrix**: Visual breakdown of TP, FP, TN, FN

---

## ⚖️ Class Imbalance & Solutions

Logistic regression can be sensitive to imbalanced datasets.

**Solutions:**

- Use `class_weight='balanced'` in scikit-learn
- Downsample or upsample classes
- Use resampling or SMOTE

---

## 🧠 Assumptions of Logistic Regression

- The dependent variable is binary (0 or 1)
- Features are **linearly separable** in the log-odds space
- No extreme multicollinearity among features
- Large sample size is preferred

---

## 🧪 Applications

- Email spam detection
- Medical diagnosis (e.g. predicting disease presence)
- Customer churn prediction
- Loan default prediction

---

## 🔗 Further Reading

- 📘 [GFG: Understanding Logistic Regression](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)
- 📘 [GFG: Linear vs Logistic Regression](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression-vs-logistic-regression/)
- 📘 [GFG: Linear Regression Overview](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/)
