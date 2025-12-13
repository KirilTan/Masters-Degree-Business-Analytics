# Topic 0 - Introduction and Big Data Applications

## 1️⃣ What Are Big Data?
Big Data refers to datasets whose **size, complexity, and rate of generation** exceed the capabilities of traditional data-processing tools.

A commonly used definition describes Big Data as:
> Data that cannot be efficiently stored, processed, or analyzed using conventional technologies.

Big Data usually comes from:
- sensors and IoT devices
- social networks
- transactional systems
- mobile devices
- logs, text, images, audio, video

These data are often **unstructured or semi-structured**.

---

## 2️⃣ Characteristics of Big Data (V-models)

### 🔹 The 3V Model (Gartner)
The classical definition describes Big Data using three characteristics:

- **Volume** - extremely large amounts of data
- **Velocity** - high speed of data generation and processing
- **Variety** - multiple data formats (structured, semi-structured, unstructured)

This model explains *why* traditional tools struggle.

---

### 🔹 The 4V Model
An additional characteristic is often added:

- **Value** - the useful information extracted from data

This highlights that **data itself has no value without analysis**.

---

### 🔹 Extended V-models (5V → 9V)
Further research expands the concept with additional characteristics:

- **Veracity** - accuracy and reliability of data
- **Variability** - meaning of data may change over time
- **Validity** - suitability of data for a specific purpose
- **Volatility** - how long data should be stored
- **Visualization** - ability to present results clearly

📌 Key idea: Big Data is **large, fast, diverse, noisy, and often messy**.

---

## 3️⃣ From Raw Data to Structured Data

The primary goal of Big Data analytics is to:

> Transform unstructured data into **structured datasets** suitable for analysis.

Structured data can be represented as a **table**:
- rows → observations
- columns → features (variables)

Missing information is represented as **missing values (NaN)**.

Once structured, data can be:
- analyzed statistically
- clustered
- used to build predictive models

---

## 4️⃣ Big Data Lifecycle

Big Data follows a well-defined lifecycle:

### 1. Data Generation
Data is produced by multiple sources (devices, users, systems).

### 2. Data Acquisition
Includes:
- data selection
- preprocessing
- cleaning and filtering

### 3. Data Storage
Persistent storage using appropriate technologies.

### 4. Data Analytics
Main analytical stages:
- **Data transformation** - preparing data for analysis
- **Data analysis** - applying statistical and machine learning algorithms

### 5. Data Visualization
- evaluation of results
- interpretation
- presentation for decision-making

📌 Lifecycle summary:

**Raw Data → Information → Knowledge → Business Action**

---

## 5️⃣ Big Data Analytics

Traditional analytics tools are insufficient for Big Data because:
- data volumes are too large
- data types are heterogeneous
- data may arrive in real time

This leads to the emergence of **Big Data Analytics**:

> Advanced methods and technologies designed to extract insights from large-scale, heterogeneous data.

Big Data Analytics enhances the classical analytics process by enabling:
- scalability
- automation
- advanced modeling

---

## 6️⃣ Types of Analytics

### 🔹 Descriptive Analytics
Answers the question:
> **What happened?**

Techniques include:
- statistical summaries
- correlations
- sampling
- clustering (e.g. K-means)

---

### 🔹 Predictive Analytics
Answers the question:
> **What is likely to happen?**

Uses:
- regression models
- classification models
- decision trees

Focus: **future outcomes**.

---

### 🔹 Prescriptive Analytics
Answers the question:
> **What should we do?**

It evaluates multiple possible outcomes and suggests optimal actions.

---

## 7️⃣ Business Applications of Big Data

### 🏢 Business Process Management (BPM)
Big Data enhances BPM by integrating new data sources into processes.

Benefits include:
- real-time monitoring
- process optimization
- predictive decision-making

Two interaction modes:
- **PULL** - process triggers analytics
- **PUSH** - analytics triggers process actions

---

### 👥 Human Resources Management (HRM)
Big Data improves HR decisions in:

- **Recruitment** - better candidate matching
- **Training** - personalized learning paths
- **Career Management** - retention and engagement strategies

Considerations:
- ethical issues
- data privacy

---

### 📡 Telecommunications
Big Data Analytics enables:

- churn prediction
- upselling and cross-selling
- network optimization
- data monetization

Challenges:
- data security
- real-time processing
- complex infrastructure

---

## 8️⃣ Role of Machine Learning and Generative AI

Machine Learning is the **core analytical engine** of Big Data:
- regression
- classification
- clustering

Recently, **Generative AI** has emerged as an intermediary between:
- users
- analytical algorithms

Generative AI can produce:
- text
- code
- media

📌 Big Data → Machine Learning → Generative AI

---

## 9️⃣ Exam-Oriented Summary

- Big Data is defined by multiple V-characteristics
- The goal is transforming raw data into structured datasets
- Analytics follows a lifecycle from generation to visualization
- Three main analytics types: descriptive, predictive, prescriptive
- Business value is the ultimate objective
- Machine Learning is central to Big Data Analytics

---

## 🔑 One-Sentence Explanation

> Big Data Analytics transforms large, fast, and diverse data into actionable business decisions using statistical and machine learning methods.

