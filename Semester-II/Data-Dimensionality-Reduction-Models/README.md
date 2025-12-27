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
