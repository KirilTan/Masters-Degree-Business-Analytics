# Topic 2 – Linear Regression (Part 1: Concepts & Simple Linear Regression)

This document introduces **linear regression** as it is used in the course *Methods and Algorithms for Business Analytics on Big Data*.

The focus is on **concepts, interpretation, and methodology**, not on mathematical derivations.

---

## 1. Regression vs Classification

In supervised machine learning, tasks are divided into two main categories:

- **Regression** – the dependent variable (target) is **continuous**
  - examples: sales, revenue, cost, demand, price

- **Classification** – the dependent variable takes **discrete values (classes)**
  - examples: churn / no churn, approve / reject, class A / B / C

In this topic, we focus exclusively on **regression problems**.

---

## 2. Structure of the Data

Data used for regression is assumed to be **structured in tabular form**:

- **Rows** → observations
- **Columns** → variables (features)

There is:
- **one dependent variable** (usually denoted by `y`)
- **one or more independent variables** (denoted by `X`)

The goal of regression is to model the relationship between `X` and `y`.

---

## 3. Key Concepts and Terminology

The course distinguishes clearly between the following concepts:

- **Model** – an abstract mathematical representation of a real process
- **Modeling** – the process of constructing a model
- **Method** – a general procedure for solving a model
- **Algorithm** – a concrete, step-by-step implementation of a method
- **Methodology** – a sequence of methods covering the entire analytical process

In practice, Python libraries (such as scikit-learn) provide **algorithms** that implement well-known **methods** through a standard interface.

---

## 4. General Methodology for Data Analysis

Throughout the course, the following general methodology is applied:

1. Reading the data
2. Preliminary data preprocessing (missing values, scaling, etc.)
3. Choice of model and its parameters
4. Splitting the data into training and test subsets
5. Model evaluation

Linear regression is one concrete realization of this general methodology.

---

## 5. Simple Linear Regression

### 5.1 Model Definition

In **simple linear regression**, the dependent variable `y` is modeled as a linear function of a single independent variable `x`:


y = a·x + b

where:
- `a` is the **regression coefficient (slope)**
- `b` is the **intercept**

The objective of the model is to determine the values of `a` and `b` that best describe the observed data.

---

### 5.2 Training the Model

Training a linear regression model means **estimating the coefficients** `a` and `b` using observed data.

In scikit-learn, this is done using the method:

```
fit(X, y)
```

After training:
- the model parameters are fixed
- the model can be used for prediction

---

### 5.3 Prediction

Once trained, the model can be applied to known or new values of `x`:

```
predict(X)
```

This produces:
- `y_pred` – predicted values of the dependent variable

These predictions are computed using the learned linear equation.

---

## 6. Model Parameters in scikit-learn

After training a `LinearRegression` model, its parameters are available as:

- `coef_` → regression coefficients
- `intercept_` → intercept term

For simple linear regression:
- `coef_` contains a single value (the slope)
- `intercept_` contains the constant term

These parameters fully define the learned linear model.

---

## 7. Visual Interpretation

Plotting plays an important role in understanding linear regression:

- **Scatter plot** → shows the observed data points
- **Regression line** → shows the model predictions

The vertical distance between points and the regression line represents the **prediction error**.

Visual inspection helps determine whether a linear model is appropriate for the data.

---

## 8. Model Evaluation: R² (Coefficient of Determination)

The quality of a linear regression model is commonly evaluated using **R²**.

R² measures:
- the proportion of variance in `y` explained by the model

Interpretation:
- R² = 1 → perfect fit
- R² = 0 → no better than predicting the mean of `y`
- R² < 0 → worse than predicting the mean

In scikit-learn, R² can be computed using:

- `model.score(X, y)`
- `r2_score(y, y_pred)`

---

## 9. Limitations of Training Evaluation

Evaluating a model on the same data used for training can be misleading.

A high R² on training data does **not guarantee** good performance on unseen data.

This motivates the use of:
- train/test split
- cross-validation

These topics are covered in **Part 2**.

---

## 10. Transition to Part 2

In the next document, we extend linear regression by introducing:

- train/test split
- cross-validation (K-Fold)
- model selection and comparison

➡️ **Topic 2 – Linear Regression (Part 2: Validation & Generalization)**

