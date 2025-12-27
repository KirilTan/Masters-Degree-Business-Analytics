# Topic 1 – Principal Component Analysis (PCA) (Lecture 1)

---

## 1️⃣ Why PCA Exists (the business + ML reason)

Real datasets often have:
- **many features** (high dimensionality)
- **redundant information** (features overlap / correlate)
- **noise** that makes models less stable

A core DDRM goal is to keep the *useful structure* but work in **fewer dimensions**.

PCA is the classic answer when:
- you want **compression + visualization** (e.g., 2D/3D plots)
- you want to reduce **multicollinearity** (highly correlated predictors)
- you want a **cleaner input** for downstream models (classification/regression)

Textbook intuition: ML is essentially about finding **useful transformations / representations** of data. These transformations can include **linear projections that may destroy information**-that’s exactly what PCA is.

> Garzon summarizes PCA’s intent as extracting features that retain the most **variance/covariance**, flattening data into fewer dimensions (often 2D/3D) for understanding and analysis.

---

## 2️⃣ The Core PCA Idea in One Sentence

> **PCA finds a new coordinate system (axes) where the first axis captures the most variance, the second captures the next most, etc., and then keeps only the top axes.**

These axes are the **principal components (PCs)**.

---

## 3️⃣ Geometric Intuition (what PCA is “doing”)

Imagine a cloud of points in 2D/3D:
- In the original axes, the cloud might look “tilted”
- PCA **rotates** the axes to align with the direction where the cloud spreads out the most

### ✅ PC1 (First Principal Component)
- Direction that maximizes the variance of the projected data  
- “Where the data varies the most”

### ✅ PC2, PC3, …
- Next directions of variance
- Must be **orthogonal** (perpendicular) to previous components

So PCA is a **linear** transformation:
- rotation (and possibly scaling depending on preprocessing)
- followed by truncation (dropping some axes)

---

## 4️⃣ The Two Main PCA Perspectives (same result, different intuition)

### 4.1 Variance Maximization
PCA chooses directions so that the **projected data keeps as much variance as possible**.  
The variance captured by each PC is linked directly to **eigenvalues** of the covariance matrix. 

### 4.2 Reconstruction Error Minimization
Another equivalent view:
- compress → reconstruct  
- PCA is the best **linear** compression (in least-squares sense)

This “autoencoder-like” viewpoint is common in modern ML thinking:
- a linear encoder/decoder with squared loss becomes equivalent to PCA 

---

## 5️⃣ PCA Algorithm in Practice (the steps you actually do)

The “standard” workflow (and what scikit-learn assumes you mean by PCA):

### ✅ Step 1: Center the data
Subtract the mean so each feature has mean 0. 

### ✅ Step 2: Standardize (usually)
Divide by standard deviation so features become unit-free and comparable. 

📌 Why this matters: If one feature is measured in big units (e.g., “income in EUR”) it can dominate variance and “hijack” PCA.

### ✅ Step 3: Compute covariance matrix + eigendecomposition (or SVD)
PCA relies on eigenvectors/eigenvalues of the covariance matrix (or an SVD-based equivalent). 

### ✅ Step 4: Sort components by explained variance
Largest eigenvalue → PC1, then PC2, etc.

### ✅ Step 5: Project onto the top *k* components
Your new reduced features are the coordinates in the PC basis. 

---

## 6️⃣ Explained Variance (how many PCs should we keep?)

Each component has:
- **explained_variance_ratio** = “% of total variance captured by this PC”

Common selection rules:
- **Scree plot**: look for the “elbow”
- **Cumulative variance threshold**: keep enough PCs to reach e.g. 90–95%

### ⚠️ Important nuance
Explained variance is not the same as “useful for prediction.”
Sometimes:
- a low-variance direction can still be predictive for *y*
- PCA is unsupervised, so it doesn’t “know” the target

---

## 7️⃣ PCA as a Preprocessing Step in ML Pipelines

Lecture 1 demonstrates PCA combined with supervised learning through a **pipeline**:
- PCA reduces features to `n_components`
- then a classifier/regressor is trained on the reduced space
- evaluation is done with cross-validation

Example shown in the lecture: **PCA + Logistic Regression** inside a pipeline, with `n_components=10`.

📌 Why pipelines matter:
- scaling + PCA + model must be validated **together**
- otherwise you can get “data leakage” (PCA fitted using information from validation folds)

---

## 8️⃣ How to Interpret PCA Outputs (exam-friendly)

### ✅ Principal components (PCs)
- the new axes / new features (orthogonal directions)

### ✅ Loadings
- how strongly each original feature contributes to a component
- large absolute loading ⇒ that original feature is influential for that PC

### ✅ Scores
- your data points expressed in the PC coordinate system (the transformed data)

---

## 9️⃣ Limitations & When PCA Is a Bad Fit

PCA works best when:
- relationships are mostly **linear**
- variance is a good proxy for “information”

It can struggle when:
- the true structure is **nonlinear** (manifold-like)
- features are not standardized and have mismatched units
- interpretability is required in terms of original variables (PCs are mixtures)

Also note the computational aspect:
- naive eigendecomposition of a D×D covariance matrix scales poorly in very high dimensions
- using SVD and low-rank approximations is the typical solution 

---

## 🔟 Exam-Oriented Summary

- PCA is an **unsupervised linear** dimensionality reduction method.
- It creates **orthogonal components** ordered by **explained variance**. 
- Practical steps: **center → (usually) standardize → eigendecompose/SVD → project**. 
- Choose number of PCs using **scree plot** or **cumulative variance**. 
- PCA is commonly used in ML pipelines (Lecture: PCA + Logistic Regression example). 
- PCA optimizes variance (and equivalently minimizes reconstruction error in a linear sense). 

---

## 🔑 One-Sentence Explanation

> **PCA replaces many correlated features with a few orthogonal components that preserve as much variance (information) as possible.** 

---

## 🔗 References

- Lecture 1 (PCA examples + pipelines)
- Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* (PCA steps, variance, SVD link) 
- Garzon et al. — *Dimensionality Reduction in Data Science* (PCA definition + intuition) 
- François Chollet — *Deep Learning with Python* (representations + linear projections intuition) 
