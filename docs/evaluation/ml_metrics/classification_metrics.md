# Classification Metrics - Complete Guide

A comprehensive guide to understanding and using classification metrics in machine learning.

---

## Why Accuracy Can Be Misleading

**Accuracy** = (Correct Predictions) / (Total Predictions)

Sounds simple, right? But accuracy fails in two critical scenarios:

### 1. Class Imbalance

Imagine a fraud detection dataset with 99% legitimate transactions and only 1% fraud.

A "dumb" model that predicts **everything as legitimate** achieves 99% accuracy! But it catches zero fraud - completely useless.

```
Dataset: 10,000 transactions
- Legitimate: 9,900 (99%)
- Fraud: 100 (1%)

Model predicting ALL as legitimate:
- Accuracy: 9,900 / 10,000 = 99%
- Fraud caught: 0 / 100 = 0%
```

### 2. Unequal Error Costs

Not all mistakes are equal:
- **Medical diagnosis**: Missing cancer (False Negative) is far worse than a false alarm (False Positive)
- **Spam filter**: Blocking important email (False Positive) may be worse than letting spam through (False Negative)

Accuracy treats all errors equally, which is often wrong.

---

## The Confusion Matrix

The confusion matrix is the foundation of classification metrics. For binary classification:

```
                    Predicted
                 Negative | Positive
              +-----------+-----------+
Actual   Neg  |    TN     |    FP     |
              +-----------+-----------+
Actual   Pos  |    FN     |    TP     |
              +-----------+-----------+
```

### The Four Outcomes

| Term | Full Name | Meaning | Example (Disease Detection) |
|------|-----------|---------|----------------------------|
| **TP** | True Positive | Correctly predicted positive | Sick patient correctly diagnosed as sick |
| **TN** | True Negative | Correctly predicted negative | Healthy patient correctly diagnosed as healthy |
| **FP** | False Positive | Incorrectly predicted positive (Type I Error) | Healthy patient wrongly diagnosed as sick |
| **FN** | False Negative | Incorrectly predicted negative (Type II Error) | Sick patient wrongly diagnosed as healthy |

**Memory trick**: 
- First word (True/False) = Was the prediction correct?
- Second word (Positive/Negative) = What did the model predict?

---

## Precision, Recall, and F1

### Precision

**"Of all the things I predicted as positive, how many were actually positive?"**

```
Precision = TP / (TP + FP)
```

- High precision = Few false alarms
- Use when: **False positives are costly** (spam filter - don't want to block real emails)

### Recall (Sensitivity, True Positive Rate)

**"Of all the actual positives, how many did I catch?"**

```
Recall = TP / (TP + FN)
```

- High recall = Catch most positives
- Use when: **False negatives are costly** (cancer detection - don't want to miss any cases)

### F1 Score

**"The harmonic mean of precision and recall"**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- Balances precision and recall
- Use when: You need a single number and both FP and FN matter
- Range: 0 to 1 (1 is perfect)

### The Precision-Recall Tradeoff

You usually can't maximize both:
- Want to catch ALL fraud? Lower your threshold → More FP → Lower precision
- Want ZERO false alarms? Raise your threshold → More FN → Lower recall

---

## ROC-AUC vs PR-AUC

These metrics evaluate models across ALL possible thresholds, not just the default 0.5.

### ROC Curve and ROC-AUC

**ROC** = Receiver Operating Characteristic

The ROC curve plots:
- **X-axis**: False Positive Rate (FPR) = FP / (FP + TN) — "What fraction of negatives did I wrongly flag?"
- **Y-axis**: True Positive Rate (TPR) = Recall = TP / (TP + FN) — "What fraction of positives did I catch?"

**ROC-AUC** = Area Under the ROC Curve
- Range: 0.5 (random guessing) to 1.0 (perfect)
- **Interpretation**: Probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example

```
ROC-AUC Values:
0.5       = Random chance (useless model)
0.6 - 0.7 = Poor
0.7 - 0.8 = Fair
0.8 - 0.9 = Good
0.9 - 1.0 = Excellent
```

### PR Curve and PR-AUC

**PR** = Precision-Recall

The PR curve plots:
- **X-axis**: Recall
- **Y-axis**: Precision

**PR-AUC** = Area Under the PR Curve (also called Average Precision)

### Which One To Use?

| Situation | Use | Why |
|-----------|-----|-----|
| **Balanced classes** | ROC-AUC | Both work well, ROC is more common |
| **Imbalanced classes** | PR-AUC | ROC-AUC can be misleadingly high even with poor positive-class performance |
| **Positive class matters most** | PR-AUC | Focuses on positive class performance |
| **Overall ranking ability** | ROC-AUC | Better for comparing model's general ranking ability |

**Key insight**: In highly imbalanced data (99% negative, 1% positive), even many false positives barely affect FPR because the denominator (total negatives) is huge. This makes ROC-AUC look deceptively good. PR-AUC exposes this problem because precision directly penalizes false positives.

---

## Threshold Tuning

Most classifiers output probabilities, not hard 0/1 predictions. You choose a **threshold** to convert probabilities to predictions:

```
if probability >= threshold:
    predict POSITIVE
else:
    predict NEGATIVE
```

### Default Threshold (0.5)

The default 0.5 threshold isn't always optimal:
- For imbalanced data, you might want a lower threshold to catch more positives
- For high-stakes decisions, you might want a higher threshold for more confidence

### How to Tune the Threshold

1. **Use a validation set** (NOT the test set!)
2. Try many thresholds (e.g., 0.1, 0.2, ..., 0.9)
3. Pick the threshold that optimizes your chosen metric:
   - Maximize F1 for balance
   - Maximize recall for catching all positives
   - Maximize precision for minimizing false alarms

```python
# Example: Find threshold that maximizes F1
best_threshold = 0.5
best_f1 = 0
for threshold in np.arange(0.1, 0.9, 0.05):
    preds = (probabilities >= threshold).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
```

---

## Averaging Strategies (Multiclass)

When you have more than 2 classes, you need to aggregate per-class metrics:

### Macro Average

**Calculate metric for each class, then average them**

```
Macro F1 = (F1_class1 + F1_class2 + F1_class3) / 3
```

- Treats all classes equally
- Use when: All classes are equally important, regardless of size

### Micro Average

**Aggregate all TP, FP, FN across classes, then calculate**

```
Micro Precision = Total_TP / (Total_TP + Total_FP)
```

- Gives more weight to frequent classes
- For balanced classes: micro = macro

### Weighted Average

**Average metrics weighted by class frequency (support)**

```
Weighted F1 = (n1×F1_1 + n2×F1_2 + n3×F1_3) / (n1 + n2 + n3)
```

- Accounts for class imbalance
- Use when: You want performance relative to the actual class distribution

---

## Log Loss (Cross-Entropy)

**Measures how well predicted probabilities match actual outcomes**

```
Log Loss = -1/N × Σ [y×log(p) + (1-y)×log(1-p)]
```

Where:
- y = actual label (0 or 1)
- p = predicted probability

- **Range**: 0 to infinity (lower is better)
- **Key property**: Heavily penalizes confident wrong predictions
- **Use case**: When probability calibration matters (e.g., risk scoring)

---

## Metric Choice by Scenario

| Scenario | Primary Metric | Secondary | Why |
|----------|---------------|-----------|-----|
| **Spam Detection** | Precision | F1 | Don't block real emails (minimize FP) |
| **Fraud Detection** | Recall, PR-AUC | F1 | Catch all fraud (minimize FN), imbalanced data |
| **Medical Diagnosis** | Recall | Specificity | Don't miss sick patients (minimize FN) |
| **Recommender (Click)** | PR-AUC | Precision@K | Imbalanced (few clicks), rank quality matters |
| **Balanced Classification** | Accuracy, F1 | ROC-AUC | All metrics work well here |
| **Credit Scoring** | ROC-AUC, Log Loss | Precision/Recall | Need good probability ranking |

---

## Quick Reference

| Metric | Formula | Best For |
|--------|---------|----------|
| Accuracy | (TP+TN) / Total | Balanced classes only |
| Precision | TP / (TP+FP) | Minimizing false alarms |
| Recall | TP / (TP+FN) | Catching all positives |
| F1 | 2×(P×R)/(P+R) | Balance of P and R |
| ROC-AUC | Area under ROC | Overall ranking (balanced) |
| PR-AUC | Area under PR | Imbalanced datasets |
| Log Loss | Cross-entropy | Probability calibration |
