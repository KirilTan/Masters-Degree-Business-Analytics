# Topic 2 – Linear Regression (Part 1: Concepts & Simple Linear Regression)

---

## 1️⃣ Regression vs Classification

In supervised machine learning, problems are divided into **regression** and **classification** tasks.

* **Regression** → the dependent variable is **continuous**

  * examples: sales, revenue, cost, demand

* **Classification** → the dependent variable takes **discrete values (classes)**

  * examples: churn / no churn, approve / reject

In this topic, we focus on **regression problems**.

---

## 2️⃣ Structure of the Data

Data used for regression is assumed to be **structured in tabular form**:

* **Rows** → observations
* **Columns** → variables (features)

There is:

* one **dependent variable** (`y`)
* one or more **independent variables** (`X`)

The goal of regression is to model the relationship between `X` and `y`.

---

## 3️⃣ Key Concepts and Terminology

The course distinguishes between the following concepts:

* **Model** – abstract mathematical representation of a real process
* **Modeling** – process of constructing a model
* **Method** – general procedure for solving a model
* **Algorithm** – concrete step-by-step implementation of a method
* **Methodology** – sequence of methods covering the full analytical process

In practice, Python libraries provide algorithms that implement methods through a standard interface.

---

## 4️⃣ General Methodology for Data Analysis

Throughout the course, the following methodology is applied:

1. Reading the data
2. Data preprocessing (missing values, scaling, etc.)
3. Choice of model and parameters
4. Splitting data into training and test sets
5. Model evaluation

Linear regression is one concrete realization of this methodology.

---

## 5️⃣ Simple Linear Regression

### Model Definition

In **simple linear regression**, the dependent variable `y` is modeled as a linear function of a single independent variable `x`:

```
y = a·x + b
```

where:

* `a` is the **regression coefficient (slope)**
* `b` is the **intercept**

The objective is to estimate `a` and `b` from observed data.

---

## 6️⃣ Training and Prediction

Training a linear regression model means estimating the parameters `a` and `b` using data.

In scikit-learn:

* `fit(X, y)` → trains the model
* `predict(X)` → produces predicted values `y_pred`

After training, the model parameters are fixed and can be used for prediction.

---

## 7️⃣ Model Parameters in scikit-learn

After training a `LinearRegression` model, the parameters are available as:

* `coef_` → regression coefficients
* `intercept_` → intercept term

For simple linear regression:

* `coef_` contains one value (the slope)
* `intercept_` contains the constant term

Together, these parameters define the learned linear equation.

---

## 8️⃣ Visual Interpretation

Plotting helps evaluate whether a linear model is appropriate:

* **Scatter plot** → observed data points
* **Regression line** → model predictions

The vertical distance between points and the line represents the prediction error.

---

## 9️⃣ Model Evaluation: R²

The quality of a linear regression model is commonly evaluated using **R² (coefficient of determination)**.

R² measures:

* the proportion of variance in `y` explained by the model

Interpretation:

* R² = 1 → perfect fit
* R² = 0 → no better than predicting the mean
* R² < 0 → worse than predicting the mean

In scikit-learn, R² can be computed using:

* `model.score(X, y)`
* `r2_score(y, y_pred)`

---

## 🔑 One-Sentence Explanation

> Linear regression models the relationship between a continuous dependent variable and one or more independent variables using a linear function.

---

## 🔗 References & Further Reading

- 📘 [scikit-learn Documentation – Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) <br>
- 📘 [GeeksForGeeks – Linear Regression](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/) <br>

