# Topic 0 – Introduction and Overview (Lecture 1)

---

## 1️⃣ Where DDRM Fits: AI → ML → Deep Learning

In this course, we study **Dimensionality Reduction** as a core tool inside **Machine Learning (ML)**, which itself is part of **Artificial Intelligence (AI)**.

- **Artificial Intelligence (AI)** = the broad goal of automating cognitive/intellectual tasks humans normally do. It includes many approaches, not only “learning” ones.
- **Machine Learning (ML)** = instead of hand-writing rules, we let the system **learn rules from data**.
- **Deep Learning (DL)** = a subfield of ML that learns **successive layers of representations** (typically via neural networks), often dozens or more.

**Key takeaway for DDRM:** dimensionality reduction is mostly about **finding better representations** of data-exactly the central idea behind ML (and DL).

---

## 2️⃣ ML as a “New Programming Paradigm”

A very practical distinction:

### 🔹 Classical programming
```text
                 ┌───────────────────────┐ 
Rules  ───────►  │      Classical        │ ───────►  Answers
Data   ───────►  │      programming      │
                 └───────────────────────┘
```

### 🔹 Machine learning
```text
                  ┌───────────────────────┐ 
Data    ───────►  │      Machine          │ ───────►  Rules
Answers ───────►  │      learning         │
                  └───────────────────────┘
```

This is why we say we *train* a model instead of *programming* it: the _"rules"_ are extracted from examples.

---

## 3️⃣ What “Learning” Means: Generalization

Training is only useful if the model can **generalize**:
- It shouldn’t just perform well on the training examples
- It must perform well on **new, unseen data**

That **ability to work on unknown data** is the whole point of ML training.

---

## 4️⃣ The Three Learning Setups (Big Picture)

### ✅ 4.1 Supervised Learning (controlled learning)
We have **labeled** data: each example includes the correct output (target).
- typical tasks: **classification** (category) and **regression** (number)

Example: The dog is guided because it has a reference (what steak smells like), just like labeled examples guide the model.

---

### ✅ 4.2 Unsupervised Learning
We have **no labels**. The goal is to find structure:
- grouping similar items (**clustering**)
- finding hidden patterns
- discovering useful representations

Example: Labels are “lost,” so the dog must group items by similarity (smell), which mirrors clustering.

Chollet connects unsupervised learning directly to **visualization, compression, denoising**, and highlights **dimensionality reduction** as a classic unsupervised category.

➡️ **This is the main home of DDRM**: we reduce dimensions to understand data, visualize it, compress it, and remove redundancy/noise.

---

### ✅ 4.3 Reinforcement Learning (RL)
Instead of a fixed dataset, an **agent** interacts with an **environment** and learns from **rewards/penalties**.

Chollet summarizes RLwith an agent that chooses actions to maximize reward, with games as famous examples.

---

## 5️⃣ Representations: The Bridge to Dimensionality Reduction

A powerful way to think about ML is:

> ML searches for **transformations** of data that make the task easier.

Chollet gives an intuition: a change of coordinates can turn a hard classification problem into a simple rule

These transformations can include:
- coordinate changes  
- **linear projections** (often information-reducing!)  
- nonlinear transformations  

That “linear projection” idea is basically the intuition behind many dimensionality reduction methods (PCA later in the lecture).

---

## 6️⃣ Why Deep Learning Matters (and why it’s not always needed)

Deep learning became dominant not only because of performance, but because it **automates feature engineering**: instead of humans designing good features manually, a deep model learns multiple layers of representations jointly.

At the same time, the lecture warns that **most business ML today isn’t necessarily deep learning**—sometimes you don’t have enough data, or the problem is better solved with simpler methods.

**Where DDRM ties in again:** dimensionality reduction is often part of the “make the data easier” step—whether you later use classical ML or deep learning.

---

## 9️⃣ Exam-Oriented Summary

- AI is the broad field; ML is learning rules from data; DL is ML that learns **layers of representations**.
- ML replaces “handwritten rules” with “learned rules,” trained from examples.
- The key ML requirement is **generalization** to unseen data.
- Three learning setups:
  - supervised = labeled targets  
  - unsupervised = no labels; structure discovery (includes **dimensionality reduction**)  
  - reinforcement = agent + environment + rewards  
- Dimensionality reduction is fundamentally about learning **better representations** (often via projections/transformations).

---

## 🔑 One-Sentence Explanation

> DDRM studies how to transform data into a smaller, more informative representation that’s easier to visualize, understand, and model.

---

## 🔗 References

- Lecture 1
- François Chollet, *Deep Learning with Python* 
