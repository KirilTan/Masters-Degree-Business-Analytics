# ==============================================================================
# Задача 1 — Класификация на качество (червено вино) със SVM (линейно ядро) + PCA
#
# Цел:
# Да се изгради модел, който класифицира възможно най-добре качеството (quality)
# на проби от winequalityred (червено вино), използвайки:
# - SVM с линейно ядро
# - PCA върху входните данни
# - различни стойности за C и различен брой PCA компоненти
#
# Практически избори за задачата:
# - class_weight='balanced' (силен дисбаланс между класовете quality)
# - GridSearchCV с refit='f1_macro' (търсим модел, който се представя по-добре
#   средно за всички класове, а не само по обща точност)
# - умерено голям грид (показва търсене на параметри, но остава изпълним бързо)
# ==============================================================================

# %%
# ==============================================================================
# 0) Импорти
# ------------------------------------------------------------------------------
# - NumPy/Pandas: обработка на данни
# - Matplotlib: графики
# - scikit-learn: split, CV, pipeline, PCA, SVM и метрики
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# %%
# ==============================================================================
# 1) Зареждане на данните
# ------------------------------------------------------------------------------
# Четем winequalityred.csv (разделител ',').
# Махаме първата колона
# Отпечатваме:
# - размер на набора (редове/колони)
# - общ брой липсващи стойности
# ==============================================================================
df = pd.read_csv("winequalityred.csv", sep=",")
df = df.drop(columns=[df.columns[0]])
print("Размер (shape):", df.shape)
print("Липсващи стойности общо:", int(df.isna().sum().sum()))

# %%
# ==============================================================================
# 2) Разпределение на целевата променлива (quality)
# ------------------------------------------------------------------------------
# Показваме броя проби във всеки клас quality и правим бар-чарт.
# Това е ключово, защото класовете са небалансирани (доминират 5 и 6),
# което влияе на избора на метрики и обучение на модела.
# ==============================================================================
counts = df["quality"].value_counts().sort_index()
print("\nРазпределение на quality (бройки):")
print(counts.to_string())

plt.figure(figsize=(8, 4))
ax = counts.plot(kind="bar")
plt.xlabel("quality")
plt.ylabel("count")
plt.title("Разпределение на quality (червено вино)")
plt.grid(True, alpha=0.3)
for i, v in enumerate(counts.values):
    ax.text(i, v + counts.max() * 0.01, str(v), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()

# %%
# ==============================================================================
# 3) Подготовка на X и y + stratified train/test split
# ------------------------------------------------------------------------------
# X: всички признаци (11 физико-химични характеристики)
# y: quality (многокласова цел)
#
# Използваме stratify=y, за да запазим приблизително същите пропорции на класовете
# в train и test.
# ==============================================================================
X = df.drop(columns=["quality"]).values
y = df["quality"].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print("\nTrain class counts:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test  class counts:", dict(zip(*np.unique(y_test,  return_counts=True))))

# %%
# ==============================================================================
# 4) Cross-validation настройки + метрики
# ------------------------------------------------------------------------------
# Използваме StratifiedKFold (5-fold), за да:
# - запазим разпределението на класовете във всеки fold
# - получим стабилна оценка при разумно време за изпълнение
#
# Метрики:
# - accuracy: обща точност
# - balanced_accuracy: средна точност по класове (важна при дисбаланс)
# - f1_macro: средно F1 по класове (подходяща при редки класове)
# ==============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
}

# %%
# ==============================================================================
# 5) Бейслайн модел (без PCA)
# ------------------------------------------------------------------------------
# Pipeline: StandardScaler → SVC(kernel='linear')
# class_weight='balanced' дава по-голяма тежест на редките класове.
# Това често води до по-ниска accuracy, но по-добро поведение по класове.
# ==============================================================================
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", C=1.0, class_weight="balanced")),
])

baseline_cv = cross_validate(baseline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
print("\n=== Бейзлайн (без PCA, class_weight=balanced) — CV (5-fold) ===")
for k in scoring:
    mean = baseline_cv[f"test_{k}"].mean()
    std = baseline_cv[f"test_{k}"].std()
    print(f"{k:>18}: {mean:.4f} ± {std:.4f}")

# %%
# ==============================================================================
# 6) PCA ориентир: обяснена вариация (train, след скалиране)
# ------------------------------------------------------------------------------
# PCA трябва да се прилага върху стандартизирани данни.
# Изчисляваме колко компоненти са нужни за >=90% и >=95% кумулативна вариация,
# за да имаме ориентир при избора на n_components в GridSearch.
# ==============================================================================
scaler_tmp = StandardScaler()
X_train_scaled = scaler_tmp.fit_transform(X_train)

pca_full = PCA().fit(X_train_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

k90 = int(np.argmax(cum_var >= 0.90) + 1)
k95 = int(np.argmax(cum_var >= 0.95) + 1)

print("\n=== PCA ориентир (train, скалиран) ===")
print("Компоненти за >=90% вариация:", k90)
print("Компоненти за >=95% вариация:", k95)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
plt.axhline(0.90, linestyle="--")
plt.axhline(0.95, linestyle="--")
plt.axvline(k90, linestyle="--")
plt.axvline(k95, linestyle="--")
plt.xlabel("брой PCA компоненти")
plt.ylabel("кумулативна обяснена вариация")
plt.title("PCA: кумулативна обяснена вариация (train)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ==============================================================================
# 7) GridSearchCV: StandardScaler → PCA → Linear SVM (balanced)
# ------------------------------------------------------------------------------
# Търсим най-добра комбинация от:
# - C: контролира regularization при SVM (малко C → по-строг margin; голямо C → по-гъвкав модел)
# - n_components: брой PCA компоненти (компресия/денойзинг vs загуба на информация)
#
# refit='f1_macro': избираме модела, който е най-добър средно за всички класове.
# ==============================================================================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("svm", SVC(kernel="linear", class_weight="balanced")),
])

param_grid = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "pca__n_components": [3, 4, 5, 6, 7, 8, 9, 10, 11],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scoring,
    refit="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=0,
    return_train_score=False,
)

grid.fit(X, y)

print("\n=== Най-добри параметри (refit=f1_macro) ===")
print("Best params:", grid.best_params_)
print("Best mean CV f1_macro:", round(grid.best_score_, 4))

# %%
# ==============================================================================
# 8) Обобщение на резултатите от GridSearch
# ------------------------------------------------------------------------------
# Извеждаме Top 10 конфигурации според mean CV f1_macro.
# Показваме и mean accuracy / mean balanced accuracy като допълнителен контекст.
# ==============================================================================
results = pd.DataFrame(grid.cv_results_)

view_cols = [
    "param_svm__C",
    "param_pca__n_components",
    "mean_test_accuracy",
    "mean_test_balanced_accuracy",
    "mean_test_f1_macro",
    "std_test_f1_macro",
    "rank_test_f1_macro",
]

results_view = results[view_cols].copy().sort_values(
    ["rank_test_f1_macro", "mean_test_accuracy"], ascending=[True, False]
)

print("\n=== Top 10 конфигурации по mean CV f1_macro ===")
print(results_view.head(10).to_string(index=False))

# %%
# ==============================================================================
# 9) Финална оценка върху тестовия дял
# ------------------------------------------------------------------------------
# Обучаваме най-добрия модел (по CV macro F1) върху train и оценяваме върху test.
# Показваме:
# - accuracy, balanced accuracy, macro F1
# - classification report и confusion matrix за анализ по класове
# ==============================================================================
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

print("\n=== Финален модел (best by CV f1_macro) — TEST SPLIT ===")
print("Params:", grid.best_params_)
print(f"Test accuracy         : {acc:.4f}")
print(f"Test balanced accuracy: {bal_acc:.4f}")
print(f"Test macro F1         : {f1m:.4f}")

print("\nClassification report (test):\n")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

labels_sorted = np.sort(np.unique(y))
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
print("Confusion matrix (редове=истински, колони=предсказани):")
print("Labels:", labels_sorted)
print(cm)

# %%
# ==============================================================================
# 10) Графика: mean CV macro F1 спрямо n_components за различни C
# ------------------------------------------------------------------------------
# Визуализира влиянието на:
# - броя PCA компоненти (сложност/информация)
# - параметъра C (regularization)
# върху macro F1 (основната метрика за избор на модел).
# ==============================================================================
plt.figure(figsize=(9, 5))

for C in param_grid["svm__C"]:
    sub = results_view[results_view["param_svm__C"] == C].copy()
    sub["param_pca__n_components"] = sub["param_pca__n_components"].astype(int)
    sub = sub.sort_values("param_pca__n_components")

    plt.plot(
        sub["param_pca__n_components"],
        sub["mean_test_f1_macro"],
        marker="o",
        label=f"C={C}",
    )

plt.xlabel("брой PCA компоненти (n_components)")
plt.ylabel("mean CV f1_macro")
plt.title("SVM(linear)+PCA: macro F1 спрямо компоненти (GridSearch, CV)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
