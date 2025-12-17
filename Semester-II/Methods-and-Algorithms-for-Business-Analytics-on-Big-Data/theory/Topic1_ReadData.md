# Topic 1 – Reading Data

---

## 1️⃣ What Is a Dataset?

A **dataset** is a collection of observations collected for analysis.

In business analytics, a dataset is usually represented as a **table**:

* **Rows** → observations (records, cases)
* **Columns** → variables (features, attributes)

Each observation describes one entity (e.g. customer, transaction, employee) using multiple variables.

---

## 2️⃣ Structured Data

This course mainly works with **structured data**.

### Characteristics of structured data:

* tabular format
* fixed schema (same columns for all rows)
* easy to store, query, and analyze

Typical examples:

* CSV files
* Excel spreadsheets
* relational database tables

Structured data is the natural input for:

* statistical analysis
* regression models
* classification and clustering

---

## 3️⃣ CSV Files (Comma-Separated Values)

A **CSV file** is a text file that represents a table:

* each line corresponds to one observation
* values are separated by a delimiter (usually a comma)
* the first row often contains column names

### Why CSV is widely used:

* simple and lightweight
* human-readable
* supported by almost all tools (Excel, Python, R, databases)

### Conceptual mapping:

```
CSV file ↔ Table ↔ Pandas DataFrame
```

In Python analytics, CSV files are most often loaded into **pandas DataFrames**.

---

## 4️⃣ Missing Values

In real datasets, information is often incomplete.

### Common reasons for missing values:

* data was not collected
* data was lost during transfer
* not applicable for a given observation

In pandas, missing values are represented as:

* **NaN** (Not a Number)

Handling missing values correctly is essential before any modeling step.

---

## 5️⃣ Reading Data as Part of the Big Data Lifecycle

Reading data corresponds to the **Data Acquisition** stage in the Big Data lifecycle.

At this stage, the main goals are:

* load data correctly
* understand its structure
* identify missing or problematic values

No modeling or prediction is performed here.

---

## 6️⃣ Data Before Analytics

A key principle of business analytics:

> Incorrectly read or misunderstood data leads to incorrect conclusions.

Before applying regression or machine learning algorithms, we must:

* inspect the dataset
* understand variables and observations
* ensure data quality

This topic lays the foundation for all analytical methods that follow.

---

## 6️⃣ Pandas and NumPy in Data Reading

In Python-based data analytics, two libraries play a fundamental role when working with datasets: **NumPy** and **pandas**.

### NumPy

**NumPy** is a library for efficient numerical computation.
It provides:

* fast numerical arrays
* efficient mathematical operations on large datasets

NumPy forms the **numerical foundation** of many scientific and analytical libraries in Python.

### Pandas

**Pandas** is a high-level data analysis library built on top of NumPy.
It is designed specifically for working with **structured data**.

Pandas allows us to:

* read data from CSV and Excel files
* represent datasets in tabular form
* handle missing values
* inspect and explore data before analysis

The central pandas object is the **DataFrame**, which represents a dataset as a table with rows and columns.

### Relationship Between NumPy and Pandas

```
NumPy   → numerical foundation
Pandas → structured data handling and analysis
```

In this course, the typical workflow is:

1. load data using pandas
2. store it in a DataFrame
3. prepare it for statistical or machine learning analysis

---

## 7️⃣ Exam-Oriented Summary

* A dataset is a collection of observations organized in rows and columns
* Structured data is the main focus of this course
* CSV files are the standard format for analytical work
* Missing values are common and represented as NaN
* Reading data is part of the data acquisition stage

---

## 🔑 One-Sentence Explanation

> Reading data means loading structured datasets correctly, understanding their structure, and preparing them for further analysis.

---

## 🔗 References & Further Reading

- 📘 [pandas Documentation - Overview](https://pandas.pydata.org/docs/user_guide/index.html) <br>
- 📘 [pandas Documentation - IO Tools (CSV & TXT)](https://pandas.pydata.org/docs/user_guide/io.html) <br>
- 📘 [pandas Documentation - Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html) <br>
- 📘 [NumPy Documentation - Overview  ](https://numpy.org/doc/stable/user/whatisnumpy.html) <br>
