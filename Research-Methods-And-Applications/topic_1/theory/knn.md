
# 📍 k-Nearest Neighbors (k-NN) – Theory & Application

**k-NN** is a simple and intuitive **supervised machine learning algorithm** used for both **classification** and **regression** tasks.

It belongs to the family of **instance-based** (or **lazy learning**) algorithms.

---

## 🔍 How It Works

- Given a new data point, k-NN **searches for the k nearest data points** in the training set (based on a distance metric like Euclidean distance).
- For **classification**, it assigns the label most common among those neighbors (majority vote).
- For **regression**, it predicts the **average value** of the neighbors.

---

## ⚙️ Key Concepts

- **k**: Number of neighbors to consider (e.g., k = 3)
- **Distance metrics**: Euclidean, Manhattan, Minkowski
- **Decision boundary**: Can be jagged and non-linear
- **Lazy learning**: No training phase; all computation happens at prediction time

---

## ✅ Strengths

- Simple to understand and implement
- No assumptions about data distribution
- Works well with small datasets and well-separated classes

## ⚠️ Weaknesses

- **Computationally expensive** at prediction time (especially for large datasets)
- **Sensitive to irrelevant features** and feature scaling
- **Can overfit** if k is too small; underfit if k is too large
- Doesn’t perform well on **high-dimensional** or **noisy** data

---

## 📊 Use Cases

- Image and handwriting recognition
- Recommender systems (e.g., movies, products)
- Water quality or medical classification (e.g., based on lab results)
- Anomaly detection

---

## 🔗 Further Reading

- 📘 [GFG: k-NN Algorithm](https://www.geeksforgeeks.org/k-nearest-neighbours/)
