# 🧭 ML Classification Practical Playbook (Lecture 1)

*A compact, hands-on guide you can open next to your code. Focused on tabular binary classification (your Framingham & Water datasets), but broadly reusable.*

---

## 0) Fast Workflow Checklist

1. **Load & inspect**
   - `df.info()`, `df.describe()`, `y.value_counts(normalize=True)`
2. **Split (before any transforms)**
   - `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`
3. **Preprocess (fit on train only)**
   - Numeric: impute missing (median/mean), **scale** when needed (LR, kNN, SVM)
   - Categorical: one‑hot encode
4. **Pick baseline(s)**
   - DummyClassifier (stratified), Logistic Regression, Decision Tree/Random Forest, Naive Bayes, k‑NN
5. **Imbalance strategy**
   - Start with `class_weight="balanced"` or balanced train sample
6. **Cross‑validate**
   - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   - Score with **`f1_macro`** (or class‑specific recall if that’s your KPI)
7. **Tune key hyperparams**
   - Small grid with `GridSearchCV` / `RandomizedSearchCV`
8. **Evaluate on test**
   - `classification_report`, confusion matrix, PR‑AUC if class imbalance
9. **(Optional) Threshold tuning & calibration**
   - `CalibratedClassifierCV`, then choose threshold for desired precision/recall
10. **Save artifacts & notes**
    - Model params, metrics, random_state, data version, code commit

---

## 1) Metrics You’ll Actually Use

- **Accuracy**: useful only when classes are balanced.
- **Precision (class 1)**: how clean are positives? (avoid false alarms)
- **Recall (class 1)**: how many real positives caught? (avoid misses)
- **F1 (class 1)** / **F1-macro**: balance precision & recall; macro = treats classes equally.
- **AUC‑PR**: preferred when positive class is rare.

**Quick snippet**
```python
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
print(classification_report(y_true, y_pred, digits=3))
print(confusion_matrix(y_true, y_pred))
# If you have probabilities for class 1:
print("PR AUC:", average_precision_score(y_true, y_proba))
```

**Threshold tuning**
```python
import numpy as np
th = 0.30  # example threshold for class 1
y_pred = (y_proba >= th).astype(int)
```

---

## 2) Preprocessing Patterns

- **Numeric**: impute -> (optionally) scale
- **Categorical**: `OneHotEncoder(handle_unknown="ignore")`
- **Scale** for: **Logistic Regression, k‑NN, SVM**. Not required for trees/forests/NB (GaussianNB may benefit).

**ColumnTransformer + Pipeline**
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])

pre = ColumnTransformer([("num", num_pipe, num_cols),
                         ("cat", cat_pipe, cat_cols)])
```

---

## 3) Algorithm Cheat Sheets (When / What to tune)

### Logistic Regression (baseline for linear signal)
- **When**: baseline; interpretable; good with many features after scaling.
- **Tune**: `C` (1.0 default; lower = stronger regularization), `penalty`, `solver`, `class_weight`.
```python
from sklearn.linear_model import LogisticRegression
clf = Pipeline([("pre", pre),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
```

### Decision Tree (fast rules, can overfit)
- **When**: quick rules; non‑linear splits.
- **Tune**: `max_depth`, `min_samples_leaf`, `min_samples_split`, `class_weight`. Use small depth to avoid overfitting.
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
```

### Random Forest (strong tabular baseline)
- **When**: robust default for tabular; handles interactions.
- **Tune**: `n_estimators` (200–500), `max_depth`, `max_features` (e.g. "sqrt"), `min_samples_leaf`, `class_weight`.
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=400, max_depth=None,
                               class_weight="balanced", n_jobs=-1, random_state=42)
```

### Naive Bayes (fast, high‑dimensional text; simple tabular)
- **When**: text (Multinomial/Bernoulli); simple continuous (GaussianNB).
- **Tune**: `alpha` (smoothing) for Multinomial/Bernoulli.
```python
from sklearn.naive_bayes import GaussianNB
# Works directly on numeric; consider scaling softly or leave as is
```

### k‑NN (non‑parametric, local)
- **When**: small datasets; well‑separated clusters; low dimension; scaled numeric.
- **Tune**: `n_neighbors` (odd integers, 3–31), `weights` ("uniform" vs "distance"), `metric`.
```python
from sklearn.neighbors import KNeighborsClassifier
model = Pipeline([("pre", pre), ("knn", KNeighborsClassifier(n_neighbors=11, weights="distance"))])
```

---

## 4) Class Imbalance Kit

- **Start**: `class_weight="balanced"` for LR/Tree/RF/SVM
- **Alternative**: downsample majority / oversample minority (SMOTE for continuous features)
- **Never** balance the **test** set
- **Report** per‑class metrics; prefer **F1‑macro** or **Recall (minority)**

**SMOTE example (imbalanced‑learn)**
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

model = ImbPipeline([("pre", pre),
                     ("smote", SMOTE(random_state=42)),
                     ("rf", RandomForestClassifier(n_estimators=400, random_state=42))])
```

---

## 5) Cross‑Validation & Tuning Templates

**Stratified CV + Grid Search (F1‑macro)**
```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([("pre", pre),
                 ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

param_grid = {
  "clf__C": [0.1, 1.0, 3.0, 10.0],
  "clf__penalty": ["l2"],
  "clf__solver": ["liblinear", "lbfgs"]
}

gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_score_, gs.best_params_)
```

**Random Forest quick search**
```python
pipe = Pipeline([("pre", pre),
                 ("rf", RandomForestClassifier(class_weight="balanced", random_state=42))])

param_grid = {
  "rf__n_estimators": [200, 400, 600],
  "rf__max_depth": [None, 6, 10, 16],
  "rf__min_samples_leaf": [1, 2, 4],
  "rf__max_features": ["sqrt", 0.5, None]
}
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
```

---

## 6) Reporting & Reproducibility

**Reporting block**
```python
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay

y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
print(confusion_matrix(y_test, y_pred))

if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, 1]
    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    RocCurveDisplay.from_predictions(y_test, y_proba)
```

**Repro tips**
- Set `random_state=42` (or your favorite) everywhere.
- Record: dataset version, preprocessing, CV, params, scores.
- Keep raw **and** processed data snapshots when possible.

---

## 7) Applying to Your Datasets

### Framingham (heart disease)
- **Target**: `TenYearCHD` (imbalanced)
- **Start**: LR (with scaling) + RF baseline; `class_weight="balanced"`
- **Watch**: proper train/test split before any balancing; feature scaling leakage

### Water Potability
- **Target**: `Potability` (imbalanced)
- **Start**: RF (strong tabular baseline), LR (scaled), k‑NN (scaled)
- **Watch**: missing values in `ph`, `Sulfate`, `Trihalomethanes`; impute first

---

## 8) Minimal Reusable Skeleton

```python
# 1) split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

# 2) preprocessing
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# (define 'pre' as shown above)

# 3) model
pipe = Pipeline([("pre", pre),
                 ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))])

# 4) CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro:", scores.mean(), "±", scores.std())

# 5) fit & evaluate
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
```

---

## 9) Common Gotchas

- **Data leakage**: scaling/imputation fit on full data before split → always fit on **train only**
- **Imbalance**: reporting only accuracy → add per‑class metrics or F1‑macro
- **Randomness**: missing `random_state` makes results non‑reproducible
- **k‑NN without scaling**: distances become meaningless
- **Over‑tuned models**: validate with CV; keep a final untouched test set

---

## 10) What to Prioritize in Exercises

- Implement **Pipeline + ColumnTransformer** early (clean experiments)
- Compare **LR vs RF** as baselines; add k‑NN and NB for contrast
- Use **StratifiedKFold + f1_macro** to compare models fairly
- For imbalanced targets, try both **class_weight** and **SMOTE** (train only)
- Finish with **threshold tuning** if the business goal favors precision or recall

---

**Keep this open** while coding. Commit it under `/theory/00_ml_practical_playbook.md` so you (and recruiters) can see your process and standards.
