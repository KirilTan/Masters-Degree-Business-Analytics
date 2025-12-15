# Scripts

This folder contains **Python `.py` files executed in program mode**, primarily using **Spyder**.

It complements the use of Jupyter notebooks by focusing on **structured execution, repetition, and discipline**, rather than explanation or visualization.

---

## 📂 Folder Structure

```text
scripts/
├── exercises/   # Practice scripts (learning-focused)
├── homework/    # Assignment-specific scripts
└── README.md    # This file
```

---

## 🧪 scripts/exercises/

Contains small `.py` files used for **practice and experimentation**.

Typical use cases:

* reading CSV and TXT files
* testing delimiters and decimal formats
* handling missing values
* saving processed datasets

These files are:

* learning-oriented
* allowed to contain print statements
* not intended for reuse

---

## 📝 scripts/homework/

Contains `.py` files created for **specific assignments**.

Homework scripts should be:

* self-contained
* easy to run and review
* aligned with a given task description

---

## About Utilities (utils)

At the current stage of the course, **no reusable utility modules are used**.

This is intentional:

* the focus is on learning core concepts
* logic is kept explicit inside scripts

If repeated patterns emerge later (e.g. repeated loading or cleaning logic), a `utils/` subfolder may be introduced.

---

## 🧠 Relation to Notebooks

* **notebooks/** → explanation, intuition, experimentation
* **scripts/** → execution, repetition, discipline

Both formats are used intentionally and serve different purposes.

---

> *Notebooks explain. Scripts execute.*
