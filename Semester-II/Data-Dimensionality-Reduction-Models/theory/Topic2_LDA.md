# Topic 2 - Linear Discriminant Analysis (LDA) (Lecture 2)

---

## 1️⃣ Why LDA Exists (business + ML reason)

PCA reduces dimensionality by preserving **maximum variance** in the input features **X**, but PCA does **not** use
labels.

When we *do* have labels (classes), we often want dimensionality reduction that preserves what matters for *
*classification**:

- clearer separation between classes
- reduced redundancy/noise **without** destroying discriminative information
- faster training / simpler models (fewer dimensions)

**LDA is supervised dimensionality reduction**: it learns a projection using **X and y** so that classes become as
separable as possible, using **linear** boundaries in the projected space.

---

## 2️⃣ PCA vs LDA (high-yield comparison)

| Aspect          | PCA                                       | LDA                                                           |
|-----------------|-------------------------------------------|---------------------------------------------------------------|
| Learning type   | Unsupervised                              | **Supervised**                                                |
| Uses labels `y` | ❌                                         | ✅                                                             |
| Goal            | Preserve variance                         | **Preserve class separability**                               |
| Directions      | Orthogonal PCs                            | Discriminant directions (not necessarily orthogonal)          |
| Max #components | ≤ min(n_features, n_samples)              | **≤ (C − 1)** where C = #classes                              |
| Typical use     | visualization, compression, preprocessing | **classification-oriented reduction**, visualization by class |

**Key exam idea:**

- PCA asks: “Where does the data vary most?”
- LDA asks: “Where do the classes separate most?”

---

## 3️⃣ The Core LDA Idea: Fisher’s Criterion

LDA tries to find directions where:

- **class means are far apart** (large between-class separation)
- **each class is tight** (small within-class spread)

A classic two-class objective (Fisher criterion) is:

$$
J(w) = \frac{w^T S_B w}{w^T S_W w}
$$

- $w$ = projection direction (a vector of weights)
- $S_B$ = between-class scatter
- $S_W$ = within-class scatter

Intuition:

- big numerator → projected class centers are far apart
- small denominator → projected classes are compact

---

## 4️⃣ The Key Objects (means + scatter matrices)

Let:

- $x_i$ be a sample
- it belongs to class $c$
- $\mu_c$ be the mean of class $c$
- $\mu$ be the overall mean
- $n_c$ be the number of samples in class $c$

### ✅ Within-class scatter (how much each class spreads)

$$
S_W = \sum_{c=1}^{C} \sum_{x_i \in c} (x_i - \mu_c)(x_i - \mu_c)^T
$$

### ✅ Between-class scatter (how far class means are)

$$
S_B = \sum_{c=1}^{C} n_c (\mu_c - \mu)(\mu_c - \mu)^T
$$

LDA searches for directions where between-class scatter is large *relative* to within-class scatter.

---

## 5️⃣ How LDA finds the directions (eigenproblem intuition)

For multiclass LDA, the discriminant directions can be derived from the generalized eigenvalue problem:

$$
S_W^{-1} S_B w = \lambda w
$$

- eigenvectors $w$ give the discriminant axes (LD1, LD2, …)
- eigenvalues $\lambda$ relate to how much separation each axis provides

---

## 6️⃣ Why the max is **C − 1** components

If there are **C classes**, their means can only be separated in at most **C − 1** independent directions.

So:

$$
n\_{components} \le C - 1
$$

Examples:

- C = 2 → max 1 dimension (LD1)
- C = 3 → max 2 dimensions (LD1, LD2)

This is a major difference from PCA.

---

## 7️⃣ Two ways to use LDA in practice

### ✅ 7.1 LDA as a classifier

You fit LDA and directly predict:

- `fit(X, y)`
- `predict(X)`
- `score(X, y)`

Use-case: quick baseline classifier with linear boundaries.

### ✅ 7.2 LDA as a dimensionality reduction step (transformer)

You use LDA to project into a smaller space, then train another classifier:

- `fit(X, y)` learns discriminant directions using labels
- `transform(X)` returns LD scores (new coordinates)
- downstream model trains on LD scores

This mirrors “PCA → model” pipelines, but **LDA uses y during fitting**.

---

## 8️⃣ Scaling + leakage (very important)

### Why scaling matters

LDA uses covariance structure; if features are in different units, some can dominate.

Standardization:

$$
z = \frac{x - \mu}{\sigma}
$$

- $\mu$ and $\sigma$ must be learned from training data only

### Why pipelines matter (avoid data leakage)

If you do cross-validation, any step that “learns from data” must be inside the pipeline:

- `StandardScaler()`
- `LDA()`
- model (e.g., Logistic Regression)

Otherwise you accidentally let validation data influence the learned transformation.

Rule of thumb:
> If you call `fit(...)`, it must happen inside the CV fold.

---

## 9️⃣ How to interpret LDA outputs (scikit-learn)

Useful concepts:

- **LD scores**: the transformed coordinates in discriminant space (output of `transform`)
- **Discriminant directions**: weight vectors that define LD axes

Useful attributes (when available):

- `means_` → class mean vectors
- `priors_` → class prior probabilities
- `scalings_` → projection weights (directions)
- `explained_variance_ratio_` → how much discriminative variance each LD axis explains (multiclass)

📌 Documentation:

- https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

---

## 🔟 When LDA works well (and when it doesn’t)

### Works best when

- separation is mostly **linear**
- class distributions are reasonably “Gaussian-like”
- class covariances are not wildly different (classical LDA assumption)

### Can struggle when

- separation is **nonlinear** (kernels / trees / neural nets may be better)
- covariances differ a lot between classes (QDA may be better)
- minority class is small → accuracy may look fine but recall/F1 can be poor

---

## 1️⃣1️⃣ Common mistakes (quick checklist)

- ❌ Using LDA like PCA (forgetting it needs labels during `fit`)
- ❌ Not scaling when features have different units
- ❌ Fitting scaler/LDA outside cross-validation (data leakage)
- ❌ Ignoring class imbalance (use F1/recall, not only accuracy)
- ❌ Choosing `n_components` > (C − 1)

---

## ✅ Exam-Oriented Summary

- LDA is **supervised** dimensionality reduction (uses **X and y**).
- It maximizes class separability: big **between-class** separation and small **within-class** spread.
- The Fisher criterion:

$$
J(w) = \frac{w^T S_B w}{w^T S_W w}
$$

- Maximum number of discriminant components is:

$$
n\_{components} \le C - 1
$$

- Use pipelines to prevent leakage (scaler → LDA → model).

---

## 🔑 One-Sentence Explanation

> **LDA projects data into fewer dimensions while making the classes as separable as possible.**

---

## 🔗 References

- Lecture 2 (course slides)
- Garzon et al., *Dimensionality Reduction in Data Science*
- Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* (linear algebra background)
- scikit-learn docs: LinearDiscriminantAnalysis
