# Topic 2 – Polynomial Regression

---

## 1️⃣ Why Polynomial Regression Exists

Linear regression assumes a **straight-line** relationship between predictors and the target:

$$
y = \beta_0 + \beta_1 x
$$

In real business data, relationships are often **nonlinear**:
- demand may grow faster after a threshold
- energy consumption may increase disproportionately in freezing cold/heat
- costs may rise faster with scale (nonlinear operational complexity)

When a linear model cannot capture such curvature, polynomial regression is a common next step.

---

## 2️⃣ Key Idea

Polynomial regression extends linear regression by adding **powers of features**:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d
$$

Where:
- $d$ = degree of the polynomial
- higher degree → more flexibility

---

## 3️⃣ Important Concept: Still Linear in the Coefficients

Even though the model is **nonlinear in $x$**, it is still **linear in the parameters** $\beta$:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2
$$

The unknowns ($\beta_0, \beta_1, \beta_2$) appear linearly, so we still solve it using standard **linear regression methods**.

In practice:
- we **transform features** (add $x^2, x^3, ...$)
- then fit `LinearRegression()` on the transformed dataset

---

## 4️⃣ Polynomial Regression with Multiple Features

For multiple predictors $X_1, X_2, X_3$, polynomial expansion includes:

### Degree 2 examples
- squares: $X_1^2, X_2^2, X_3^2$
- interactions: $X_1 X_2, X_1 X_3, X_2 X_3$

### Degree 3 adds even more terms
- $X_1^3$, $X_1^2 X_2$, $X_1 X_2 X_3$, etc.

This is why polynomial regression can dramatically improve performance if the true relationship contains interactions or nonlinear effects.

---

## 5️⃣ When to Use Polynomial Regression

Polynomial regression is useful when:

### ✅ Visual signs
- scatter plots show a curved pattern
- residual plots show systematic structure (curve/funnel pattern)
- linear regression underfits

### ✅ Practical signs
- linear model score (R²) is good but not “good enough”
- the relationship is known to be nonlinear from domain knowledge
- the dataset is large enough to support extra model complexity

---

## 6️⃣ Risk: Overfitting

Higher-degree polynomials can fit noise, not just signal.

Typical symptoms:
- training R² increases strongly
- test R² stops improving or drops
- residuals become small on training but unstable on new data

To control overfitting:
- use train/test split
- prefer cross-validation (K-Fold)
- keep degree low unless justified (often 2–3 is enough)

---

## 7️⃣ Model Evaluation

Polynomial regression is evaluated the same way as linear regression:

- split into train/test
- train on train set
- evaluate on test set
- report R²

In scikit-learn:
- `.score(X, y)` returns R² for regression

---

## 8️⃣ Practical Implementation in Python (Conceptual)

Workflow:

1. Load and clean data  
2. Choose predictors $X$ and target $y$  
3. Split into train/test  
4. Expand features using polynomial terms  
5. Fit `LinearRegression()`  
6. Predict and evaluate R²  
7. Compare degrees and choose best

---

## 9️⃣ Exam-Oriented Summary

- Polynomial regression is used when linear regression underfits nonlinear relationships
- It is still linear regression, applied to transformed features
- Degree controls flexibility and overfitting risk
- With multiple variables, polynomial regression also creates interaction terms
- Degree must be chosen using validation (test set or cross-validation)

---

## 🔑 One-Sentence Explanation

> Polynomial regression is linear regression on polynomially expanded features, used to model nonlinear relationships while still being solvable with standard linear regression methods.

---

## 🔗 References & Further Reading

- 📘 [GeeksforGeeks – Python Implementation of Polynomial Regression](https://www.geeksforgeeks.org/machine-learning/python-implementation-of-polynomial-regression/)
- 📘 [GeeksforGeeks – Linear vs Polynomial Regression](https://www.geeksforgeeks.org/machine-learning/linear-vs-polynomial-regression-understanding-the-differences/#what-is-polynomial-regression)
- 📘 [scikit-learn – PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- 📘 [scikit-learn – LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
