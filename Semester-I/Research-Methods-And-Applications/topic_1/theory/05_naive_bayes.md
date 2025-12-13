# 🧠 05 Naive Bayes

# 🤖 Naive Bayes – TL;DR + Core ML Concepts

## 📌 What is Naive Bayes?

Naive Bayes is a **probabilistic classification algorithm** based on **Bayes’ Theorem**, assuming all features are *
*independent** of each other given the class.

It predicts the **most probable class** based on observed features.

---

## 🧠 Bayes’ Theorem (intuition)

P(Class | Features) ∝ P(Features | Class) × P(Class)

Since P(Features) is constant across classes, prediction compares:
P(Features | Class) × P(Class) for each class and picks the largest.

---

## ✅ Strengths

- Very fast and efficient, even on large datasets
- Performs well on **text** / **high-dimensional** data
- Requires little training data and no iterative optimization

## ⚠️ Weaknesses

- Assumes **feature independence** (often violated)
- Struggles with **correlated** features
- Continuous features need assumptions (e.g., Gaussian) or binning

---

## 💡 Real-world Examples

- Spam detection
- Topic/sentiment classification
- Simple medical diagnosis based on symptoms

---

## 🔍 Most Important ML Concepts Related to Naive Bayes

### 1️⃣ Bayes’ Theorem

Combines **prior** beliefs with **evidence** from data.

### 2️⃣ Conditional Probability

Likelihoods like P(word="offer" | spam).

### 3️⃣ Prior & Likelihood

- **Prior**: P(Class) — base rate of each class.
- **Likelihood**: P(Feature | Class) — how typical the feature is for that class.

### 4️⃣ Independence Assumption

Features are **conditionally independent** given the class — simplifies computation.

### 5️⃣ Variants of Naive Bayes

| Variant        | Best For                | Example Use                   |
|----------------|-------------------------|-------------------------------|
| Multinomial NB | Discrete counts         | Text (word counts)            |
| Bernoulli NB   | Binary presence/absence | Spam detection                |
| Gaussian NB    | Continuous features     | Medical data (age, BMI, etc.) |

### 6️⃣ Laplace Smoothing (Add-1)

Avoids zero probabilities for unseen feature–class combinations.

### 7️⃣ Log Probabilities

Use log-space to avoid underflow when multiplying many probabilities.

### 8️⃣ Feature Engineering

Binning/normalization for continuous features; careful with correlated inputs.

### 9️⃣ Scalability

Linear in number of samples × features — extremely fast.

---

## 🔗 Further Reading

- 📘 [GFG: Bayes’ Theorem Explained](https://www.geeksforgeeks.org/bayes-theorem/)
- 📘 [GFG: Naive Bayes Classifier](https://www.geeksforgeeks.org/naive-bayes-classifiers/)
