# ================================================================================
# Задача 2 - LDA модел за качество на вино (бяло вино)
#
# Цел: Съставете модел, който възможно най-правилно да класифицира качеството на винените проби.
#
# В Лекция 2 основният метод е Линеен дискриминантен анализ (LDA), затова тук:
# - създаваме 3-класова целева променлива (ниско / средно / високо) по същия начин както в лекцията
# - обучаваме класификатор, базиран на LDA
# - оценяваме представянето с отделен тестов набор и кръстосана валидация
# - визуализираме разделимостта на класовете в LDA пространство (LD1 срещу LD2)
#
# Забележка: за визуализация ни трябва LDA solver, който поддържа transform() (например svd или eigen).
# За финалния класификатор използваме регуляризиран LDA (lsqr + shrinkage='auto'), който често дава по-стабилни резултати.
# ================================================================================

# %%%
# ================================================================================
# 0) Импорт на нужните библиотеки
# ================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# %%
# ================================================================================
# 1) Зареждане на данните
#
# Набор от данни: Wine Quality (White Wine) (winequality-white.csv).
# Файлът е със разделител ';'.
# ================================================================================

# Зареждане на набора от данни
df = pd.read_csv('winequality-white.csv', sep=";")
print("Размер (shape):", df.shape)
print(df.head())

# %%
# ================================================================================
# 2) Бързи проверки и разпределение на зависима променлива
#
# - Липсващи стойности: очакваме да са 0
# - Разпределение на quality: бар-чарт и бройките върху колоните (за да видим дисбаланса)
#
# Това ни дава контекст: ако класовете са дисбалансирани, метрики като balanced accuracy и macro F1 са особено важни.
# ================================================================================

print("Общо липсващи стойности:", int(df.isna().sum().sum()))

counts = df["quality"].value_counts().sort_index()

ax = counts.plot(kind="bar")
plt.xlabel("quality")
plt.ylabel("count")
plt.title("Разпределение на качеството (бяло вино)")
plt.grid(True, alpha=0.3)

# add numbers on bars
for i, v in enumerate(counts.values):
    ax.text(i, v + counts.max()*0.01, str(v), ha="center", va="bottom", fontsize=9)

plt.show()

# %%
# ================================================================================
# 3) Създаване на етикети (по модела на Лекция 2)
#
# В Лекция 2 се използва 3-класова целева променлива:
# - Клас 1 (ниско качество): quality ≤ 4
# - Клас 2 (средно качество): quality 5–6
# - Клас 3 (високо качество): quality ≥ 7
#
# Така имаме C = 3 класа, следователно LDA може да създаде най-много:
# n_components ≤ C - 1 = 2
#
# Това означава, че можем да визуализираме данните в 2D като LD1 срещу LD2.
# ================================================================================

# 3-класово биниране (по Лекция 2)
df["quality3"] = np.where(
    df["quality"] <= 4, 1,
    np.where(df["quality"] <= 6, 2, 3)
)

counts_q3 = df["quality3"].value_counts().sort_index()
print("Бройки по класове (quality3):", counts_q3.to_dict())

ax = counts_q3.plot(kind="bar")
plt.xlabel("quality3 (1=low, 2=medium, 3=high)")
plt.ylabel("count")
plt.title("Разпределение на целта след биниране (quality3)")
plt.grid(True, alpha=0.3)

for i, v in enumerate(counts_q3.values):
    ax.text(i, v + counts_q3.max()*0.01, str(v), ha="center", va="bottom", fontsize=9)

plt.show()

# %%
# ================================================================================
# 4) Train/test split и матрица на признаците
#
# Предсказваме quality3 от 11-те физико-химични характеристики.
# Използваме стратифицирано разделяне (stratified split), за да запазим приблизително същите пропорции на класовете в train и test.
# ================================================================================

X = df.drop(columns=["quality", "quality3"]).values
y = df["quality3"].values

print("X размер (shape):", X.shape)
print("Класове в y:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print("Бройки по класове (train):", np.unique(y_train, return_counts=True))
print("Бройки по класове (test):", np.unique(y_test, return_counts=True))

# %%
# ================================================================================
# 5) Финален модел: регуляризиран LDA класификатор
#
# Изграждаме pipeline (конвейер):
# 1) StandardScaler - важно, защото признаците са в различни мащаби
# 2) LDA класификатор
#
# Използваме регуляризиран LDA:
# - solver="lsqr"
# - shrinkage="auto"
#
# Регуляризацията помага, когато признаците са силно корелирани и/или шумни — типично за реални таблични данни.
# ================================================================================

# Финален pipeline (конвейер) на модела
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
])

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

print("=== Финален модел (регуляризиран LDA) — тестов дял ===")
print(f"Accuracy (обща точност)       : {acc:.4f}")
print(f"Balanced accuracy             : {bal_acc:.4f}")
print(f"F1 (macro)                   : {f1m:.4f}")

print("\nОтчет за класификация (test):\n")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
print("Confusion matrix (редове=истински, колони=предсказани):\n", cm)

# %%
# ================================================================================
# 6) Кръстосана валидация (по-стабилна оценка)
#
# Един train/test split може да е „късметлийски“ или „некъсметлийски“.
# За по-надеждна оценка използваме:
# - Repeated Stratified K-Fold (запазва пропорциите на класовете във всеки fold)
# - отчитаме средна стойност ± стандартно отклонение за:
#   - Accuracy (обща коректност)
#   - Balanced accuracy (третира всички класове поравно)
#   - Macro F1 (средно F1 по класове)
#
# Това е стилът на оценяване, използван и в Лекция 2.
# ================================================================================

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro"
}

cv_res = cross_validate(final_model, X, y, cv=cv, scoring=scoring, n_jobs=1)

print("=== Кръстосана валидация (10-fold × 3 повторения) ===")
for k in scoring:
    mean = cv_res[f"test_{k}"].mean()
    std = cv_res[f"test_{k}"].std()
    print(f"{k:>18}: {mean:.4f} ± {std:.4f}")

# %%
# ================================================================================
# 7) Визуализация на разделимостта в LDA пространство (LD1 срещу LD2)
#
# За да начертаем LD1/LD2, ни трябва LDA solver, който реализира transform().
# lsqr не реализира transform() (това е причината за типичната грешка при fit_transform с lsqr).
#
# Затова тук обучаваме отделен модел само за визуализация:
# - solver="svd" (поддържа transform)
# - n_components=2
#
# Това е за интуиция и визуален анализ, не за финалната класификация.
# ================================================================================

vis_lda = Pipeline([
    ("scaler", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis(solver="svd", n_components=2))
])

X_ld = vis_lda.fit_transform(X_train, y_train)  # LD scores for training points

plt.figure(figsize=(8, 6))
for cls in [1, 2, 3]:
    subset = X_ld[y_train == cls]
    plt.scatter(subset[:, 0], subset[:, 1], alpha=0.6, label=f"Class {cls}")

plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDA проекция (LD1 срещу LD2) — обучаващ дял")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
