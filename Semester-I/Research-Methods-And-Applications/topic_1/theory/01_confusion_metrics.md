# 📊 Confusion Matrix & Classification Metrics

A **confusion matrix** is a 2x2 table used to evaluate the performance of a binary classification model.
It helps us understand the types of correct and incorrect predictions made by the model.

## 🔢 Confusion Matrix Table

|                     | Predicted Positive  | Predicted Negative  |
|---------------------|---------------------|---------------------|
| **Actual Positive** | True Positive (TP)  | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN)  |

## 📏 Key Evaluation Metrics

| Metric        | Formula                                           | Interpretation                                                                                     |
|---------------|---------------------------------------------------|----------------------------------------------------------------------------------------------------|
| **Precision** | `TP / (TP + FP)`                                  | Of all predicted positives, how many were actually correct?<br>→ Low precision = many false alarms |
| **Recall**    | `TP / (TP + FN)`                                  | Of all actual positives, how many did we catch?<br>→ Low recall = many missed cases                |
| **Accuracy**  | `(TP + TN) / (TP + TN + FP + FN)`                 | Overall correctness of the model                                                                   |
| **F1 Score**  | `2 * (Precision * Recall) / (Precision + Recall)` | Combines Precision and Recall using harmonic mean. Ideal for imbalanced data.                      |

## 🧠 Metric Selection Guidelines

| Use Case Example                    | Preferred Metric |
|-------------------------------------|------------------|
| Detecting spam emails               | Precision        |
| Medical diagnosis (e.g. cancer)     | Recall           |
| Balanced concern between both       | F1 Score         |
| Classes are balanced in the dataset | Accuracy         |

## 📌 Real-World Analogy

### **Spam Email Detection**

- **Precision**: Of emails marked as spam, how many were truly spam?
- **Recall**: Of all spam emails, how many were correctly caught by the filter?

> 🧠 Use **F1 Score** when you care about both precision and recall - especially if the dataset is imbalanced.

---

## 🔗 External Resources for Deepening Understanding

These are excellent and visual references you can return to:

- 📘 [GFG: Confusion Matrix in Machine Learning](https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/) <br>
- 📘 [GFG: Accuracy Evaluation Techniques in Data Mining](https://www.geeksforgeeks.org/data-analysis/techniques-to-evaluate-accuracy-of-classifier-in-data-mining/) <br>
- 📘 [GFG: Precision and Recall (with visuals)](https://www.geeksforgeeks.org/machine-learning/precision-and-recall-in-machine-learning/) <br>
