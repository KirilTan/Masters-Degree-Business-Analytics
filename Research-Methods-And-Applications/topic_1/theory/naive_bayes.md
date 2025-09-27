
# 🤖 Naive Bayes – TL;DR + Core ML Concepts

## 📌 What is Naive Bayes?

Naive Bayes is a **probabilistic classification algorithm** based on **Bayes’ Theorem**, assuming all features are **independent** of each other given the class.

It predicts the **most probable class** based on observed features.

---

## 🧠 Bayes’ Theorem

P(Class | Features) = [P(Features | Class) × P(Class)] / P(Features)

Since P(Features) is constant across all classes, the algorithm focuses on:

P(Features | Class) × P(Class)

---

## ✅ Strengths

- Very fast and efficient, even on large datasets
- Performs well on **text data** and **high-dimensional spaces**
- Requires little training data

## ⚠️ Weaknesses

- Assumes **feature independence** (which is rarely true)
- Performs poorly with **correlated features**
- Needs preprocessing for numerical data (e.g. normalization or Gaussian NB)

---

## 💡 Real-world Examples

- Spam detection
- Text classification (e.g. topic or sentiment)
- Medical diagnosis with symptoms

---

## 🔍 Most Important ML Concepts Related to Naive Bayes

### 1️⃣ Bayes’ Theorem
- Core of the algorithm: combines prior knowledge with evidence.

### 2️⃣ Conditional Probability
- Predicts classes based on the likelihood of features given a class.

### 3️⃣ Prior & Likelihood
- **Prior**: P(Class) — base rate of the class in the data.
- **Likelihood**: P(Feature | Class) — how often a feature appears within a class.

### 4️⃣ Independence Assumption
- Assumes all features are **conditionally independent**.
- Rare in real-world data, but the algorithm still works well.

### 5️⃣ Variants of Naive Bayes

| Variant         | Best For                     | Example Use                      |
|------------------|------------------------------|----------------------------------|
| Multinomial NB   | Word counts, discrete data   | Document classification          |
| Bernoulli NB     | Binary data                  | Spam detection                   |
| Gaussian NB      | Continuous features          | Medical diagnosis (age, BMI...)  |

### 6️⃣ Laplace Smoothing (Add-1)
- Prevents zero probability for unseen feature/class combinations.

### 7️⃣ Log Probabilities
- Used to avoid floating-point underflow when multiplying many small probabilities.

### 8️⃣ Feature Engineering
- Often necessary for **continuous or correlated features**.

### 9️⃣ Scalability
- Extremely scalable. No gradient descent needed — just counting.

---

## 🔗 Further Reading

- 📘 [GFG: Bayes’ Theorem Explained](https://www.geeksforgeeks.org/bayes-theorem/)
- 📘 [GFG: Naive Bayes Classifier](https://www.geeksforgeeks.org/naive-bayes-classifiers/)