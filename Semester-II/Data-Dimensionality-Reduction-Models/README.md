# Data Dimensionality Reduction Models (DDRM)

## 📘 Course Overview
This repository contains theory notes, notebooks, and program-mode scripts for the master's-level course **"Data Dimensionality Reduction Models (DDRM)"**, part of the program *Modeling Big Data in Business and Finance* at Sofia University *St. Kliment Ohridski*.

The course focuses on:
- understanding **high-dimensional data** and the *curse of dimensionality*
- reducing dimensions for **visualization, compression, denoising, and modeling**
- applying classic and modern **dimensionality reduction techniques**
- evaluating trade-offs between **information retention, interpretability, and performance**

The repository is designed to be:
- ✅ exam-oriented  
- ✅ learning-oriented  
- ✅ reusable as a professional portfolio  

---

## 🎯 Learning Objectives
By working through this repository, the goal is to:
- Understand *why* dimensionality reduction is needed in business analytics
- Distinguish **feature selection** vs **feature extraction**
- Apply linear methods (e.g., PCA/SVD) and nonlinear methods (e.g., manifold learning)
- Use DR for:
  - visualization and exploratory analysis
  - preprocessing before clustering/classification/regression
  - noise reduction and compression
- Evaluate DR methods using:
  - explained variance / reconstruction error
  - neighborhood preservation (for manifold methods)
  - downstream model performance (task-based evaluation)
- Communicate results clearly: *what was reduced, what was preserved, and what was lost*

---

## 🧠 Course Context & Methodology
Dimensionality reduction is not just “making data smaller” — it’s about finding **useful structure** in high-dimensional spaces.

The course is approached as a practical pipeline:
1. Understand the **data geometry** (scales, correlations, sparsity)
2. Choose a DR objective:
   - preserve variance (PCA-style)
   - preserve distances (MDS-style)
   - preserve neighborhoods (t-SNE/UMAP-style)
   - preserve information relevant to a target (supervised DR / feature selection)
3. Apply the method correctly (scaling, hyperparameters, diagnostics)
4. Evaluate both:
   - **mathematical quality** (reconstruction / structure)
   - **business usefulness** (interpretability / predictive value)
5. Document results in a reusable, lecture-by-lecture format

---

## 🛠️ Tools & Environment
The repository uses a reproducible Python environment.

Typical libraries used:
- numpy
- pandas
- scipy
- matplotlib
- scikit-learn

Optional (when needed):
- umap-learn (UMAP)
- tensorflow / keras or pytorch (autoencoders / deep DR)

---

## 📂 Repository Structure
```text
Data-Dimensionality-Reduction-Models/
│
├── datasets/ # Raw and processed datasets + documentation
│ ├── raw/
│ ├── processed/
│ └── README.md
│
├── notebooks/ # Jupyter notebooks for exploration & intuition
│
├── resources/ # Official lecture slides, PDFs, homework descriptions
│
├── scripts/ # Python scripts (program mode)
│ ├── exercises/ # Practice scripts
│ ├── homework/ # Assignment scripts
│ └── README.md
│
├── theory/ # Markdown notes with structured explanations
│ ├── _cheatsheet.md
│ ├── _glossary.md
│ └── README.md
│
└── README.md # This file
```

---

## 🧪 Notebooks vs Scripts
- **Notebooks** are used to _understand_, _explain_, and _visualize_ dimensionality reduction:
  - step-by-step intuition
  - plots (2D/3D embeddings, explained variance, reconstruction error)
  - parameter sensitivity (e.g., perplexity in t-SNE)

- **Scripts** are used to _implement_, _practice_, and _reinforce_ in program mode:
  - small focused exercises (e.g., PCA from scratch)
  - homework-ready code (clean, reproducible, runnable)

Both formats are used intentionally and serve different purposes.

---

## 📁 Datasets
Datasets are separated into:
- **raw/** → original, immutable data
- **processed/** → cleaned, scaled, encoded, or transformed data

This separation ensures reproducibility and clarity of analytical decisions.
Each dataset should be documented inside `datasets/README.md` (source, features, preprocessing, usage).

---

## 📘 How to Use This Repository
Recommended workflow:
1. Start with **theory/** for structured understanding
2. Use **notebooks/** to see methods applied visually and step-by-step
3. Use **scripts/** to practice implementations in program mode
4. Store official materials and assignments in **resources/**
5. Track datasets and transformations in **datasets/**

This mirrors real analytics work:
**understanding → experimentation → implementation → evaluation → decision support**

---

## 📌 Notes
- `.venv/`, `.idea/`, and notebook checkpoint files are intentionally excluded from version control
- Large datasets are not committed; instructions and references are provided instead
- The repository evolves **lecture by lecture**, with consistent structure and documentation

---

## 👤 Author
Graduate student in **Business Analytics / Data Analytics**, using this repository both for academic mastery and professional development.

---

> *"High dimensions hide structure; dimensionality reduction reveals what matters."*
