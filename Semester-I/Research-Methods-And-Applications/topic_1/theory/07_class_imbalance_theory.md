# 🧠 07 Class Imbalance

# ⚖️ Class Imbalance – Theory & Techniques

**Class imbalance** occurs when one class has far more samples than the other(s).
Example: 95% healthy (class 0) vs 5% disease (class 1). A naïve model could predict everyone healthy and score 95% *
*accuracy** while missing almost all positives.

---

## 🚨 Why It's a Problem

- **Accuracy is misleading** when classes are skewed.
- Models tend to **ignore the minority class**.
- Prefer **Precision, Recall, F1**, and **AUC-PR** over plain accuracy.

---

## 🛠️ Techniques to Handle It

| Strategy                    | What it Does                                               | When to Use                         |
|-----------------------------|------------------------------------------------------------|-------------------------------------|
| **Downsampling**            | Randomly drop majority-class samples                       | Fast baseline; large majority class |
| **Oversampling**            | Duplicate minority samples                                 | Small datasets; risk of overfitting |
| **SMOTE / ADASYN**          | Create synthetic minority samples                          | Tabular, continuous features        |
| **class_weight='balanced'** | Reweights the loss to care more about minority errors      | Many sklearn classifiers            |
| **Equal Sample Training**   | Build a balanced **training** set; keep test set untouched | Quick, clear evaluation             |
| **Threshold tuning**        | Adjust decision threshold to favor recall/precision        | After model calibration             |
| **Ensembles**               | RF/GBMs often handle imbalance better                      | Strong baselines                    |

> ⚠️ Balance **training only**; keep the **test set original** to get a realistic evaluation.

---

## 📊 Evaluation with Imbalance

- Report **per-class** Precision/Recall/F1 and **macro-averaged F1**.
- Use **AUC-PR** (precision–recall) in addition to ROC-AUC.
- Apply **StratifiedKFold** to preserve label ratios during CV.

```python
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 🧩 Practical Tips (sklearn)

- Logistic/Tree/RF/SVM: `class_weight='balanced'`
- Calibrate probabilities if you’ll **tune thresholds**: `CalibratedClassifierCV`
- For SMOTE: `imblearn.over_sampling.SMOTE` (from imbalanced-learn)

---

## 🔗 Further Reading

- 📘 [GFG: Handling Imbalanced Classes](https://www.geeksforgeeks.org/how-to-handle-imbalanced-classes-in-machine-learning/)

