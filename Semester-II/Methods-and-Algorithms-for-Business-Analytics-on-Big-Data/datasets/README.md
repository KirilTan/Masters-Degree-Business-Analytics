# Datasets

This folder contains all datasets used in the course **Methods and Algorithms for Business Analytics on Big Data**.

The purpose of this folder is to provide a **clear, reproducible, and topic-agnostic structure** for working with data throughout the course, without requiring frequent updates to documentation.

---

## 📂 Folder Structure

```text
datasets/
├── raw/          # Original datasets (unchanged)
├── processed/    # Cleaned or transformed datasets
└── README.md     # This file
```

---

## 📁 raw/

The **raw** folder contains datasets exactly as they were provided from external sources such as:

* lecture materials
* exercises and homework
* public datasets
* course examples

⚠️ **Important rule:**

> Files in `raw/` must **never be modified**.

They serve as:

* immutable reference data
* inputs for notebooks and scripts
* reproducible starting points for analysis

Any interpretation, cleaning, or transformation must be done **outside** this folder.

---

## 📁 processed/

The **processed** folder contains datasets that result from applying transformations to raw data.

Typical operations include:

* removing or handling missing values
* filtering observations
* converting data types
* feature preparation for modeling

Processed datasets should:

* be derived only from data in `raw/`
* be reproducible via notebooks or scripts
* reflect analytical decisions made during the workflow

---

## 🔁 Typical Data Workflow

1. Load data from `datasets/raw/`
2. Inspect structure and data quality
3. Apply cleaning or transformations
4. Save resulting datasets to `datasets/processed/`

This workflow ensures:

* reproducibility
* traceability of decisions
* clear separation between data and analysis logic

---

## 📘 Relation to the Course

* Early topics focus on **reading and understanding** raw data
* Later topics apply **statistical and machine learning methods** to processed data

The same dataset may be reused across multiple topics at different stages of processing.

---

> *Raw data represents reality. Processed data represents analytical intent.*
