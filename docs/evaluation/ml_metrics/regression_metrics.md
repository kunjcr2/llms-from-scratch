# Regression Metrics - Complete Guide

A comprehensive guide to understanding and using regression metrics in machine learning.

---

## Core Regression Metrics

### MAE - Mean Absolute Error

**"On average, how far off are my predictions?"**

```
MAE = (1/n) × Σ |y_true - y_pred|
```

**Properties:**
- Same units as target variable (e.g., dollars, meters)
- Easy to interpret: "On average, we're off by $X"
- **Robust to outliers** - doesn't square errors
- Treats all errors equally (linear penalty)

**Example:**
```
Actual:    [100, 200, 300]
Predicted: [110, 190, 320]
Errors:    [10,  10,  20]
MAE = (10 + 10 + 20) / 3 = 13.33
```

---

### MSE - Mean Squared Error

**"Average of squared prediction errors"**

```
MSE = (1/n) × Σ (y_true - y_pred)²
```

**Properties:**
- Units are squared (e.g., dollars², meters²)
- **Penalizes large errors more** (squared penalty)
- Sensitive to outliers
- Mathematically convenient (differentiable everywhere)

**Example:**
```
Actual:    [100, 200, 300]
Predicted: [110, 190, 320]
Errors²:   [100, 100, 400]
MSE = (100 + 100 + 400) / 3 = 200
```

---

### RMSE - Root Mean Squared Error

**"MSE but back to original units"**

```
RMSE = √MSE = √[(1/n) × Σ (y_true - y_pred)²]
```

**Properties:**
- Same units as target (interpretable like MAE)
- Still penalizes large errors heavily
- Most commonly used regression metric
- Easier to compare with target values

**Example:**
```
RMSE = √200 = 14.14
```

**MAE vs RMSE:**
- If RMSE >> MAE → You have some large errors (outliers)
- If RMSE ≈ MAE → Errors are uniformly distributed

---

### R² - Coefficient of Determination

**"How much variance does my model explain?"**

```
R² = 1 - (SS_res / SS_tot)

Where:
- SS_res = Σ (y_true - y_pred)²     (residual sum of squares)
- SS_tot = Σ (y_true - mean(y))²    (total sum of squares)
```

**Properties:**
- **Range**: Usually 0 to 1, but can be negative!
- **R² = 1**: Perfect predictions
- **R² = 0**: Model is as good as predicting the mean
- **R² < 0**: Model is WORSE than predicting the mean (bad model)
- **Unitless** - good for comparing across different problems

**Interpretation:**
```
R² = 0.85 → Model explains 85% of the variance in the target
```

**Caution:**
- R² always increases when you add features (even useless ones)
- Use **Adjusted R²** for model selection with different feature counts

---

### MAPE - Mean Absolute Percentage Error

**"On average, what's my percentage error?"**

```
MAPE = (100/n) × Σ |y_true - y_pred| / |y_true|
```

**Properties:**
- Expressed as percentage (scale-independent)
- Intuitive: "We're off by X% on average"
- **Problem**: Undefined when y_true = 0 (division by zero!)
- **Problem**: Asymmetric - penalizes under-predictions more than over-predictions

**Example:**
```
Actual:    [100, 200]
Predicted: [110, 220]
% Errors:  [10%, 10%]
MAPE = 10%
```

**Safe MAPE Formula** (prevents division by zero):
```python
MAPE = 100 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon))
```

---

## When to Use Each Metric

| Metric | Best When | Avoid When |
|--------|-----------|------------|
| **MAE** | Outliers in data, all errors matter equally | You want to penalize large errors |
| **MSE** | Large errors are especially bad, need differentiable loss | Outliers distort the picture |
| **RMSE** | Same as MSE but want interpretable units | Same as MSE |
| **R²** | Comparing models on same problem, explaining variance | Comparing across different datasets |
| **MAPE** | Need percentage interpretation, comparing across scales | Target has zeros or near-zero values |

### Decision Tree for Metric Selection

```
Are there significant outliers in your data?
├── YES → MAE (robust) or remove outliers first
└── NO → Do you need % interpretation?
    ├── YES → MAPE (if no zeros in target)
    └── NO → Do large errors matter more?
        ├── YES → RMSE
        └── NO → MAE
```

Always report R² alongside your primary metric for context.

---

## Common Pitfalls

### 1. Data Leakage

**Problem**: Information from the test set "leaks" into training.

**Examples:**
- Scaling/normalizing using statistics from the ENTIRE dataset (including test)
- Using future information to predict past events
- Target leakage: features that contain information about the target

**Solution**: Always fit preprocessing on training data only, then transform test data:

```python
# WRONG - leakage!
scaler.fit(X_all)  # Uses test data statistics
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# RIGHT - no leakage
scaler.fit(X_train)  # Only training statistics
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

### 2. Skewed Targets

**Problem**: Target variable is heavily skewed (e.g., house prices, income).

**Symptoms:**
- Model performs well on average but terribly on extreme values
- High RMSE relative to MAE

**Solutions:**
- Log-transform the target: `y = log(1 + y)`
- Use MAE instead of RMSE
- Train on transformed target, inverse-transform predictions

### 3. Comparing Metrics Across Different Scales

**Problem**: MAE of $100 means nothing without context.

**Wrong**: "Model A has MAE=100, Model B has MAE=0.5, so B is better"
(What if A predicts house prices and B predicts temperature?)

**Solutions:**
- Use R² for cross-problem comparison
- Use MAPE for percentage-based comparison
- Normalize by target range: MAE / (max - min)

### 4. Ignoring Outliers in Interpretation

**Problem**: A few extreme errors dominate MSE/RMSE.

```
Errors: [1, 1, 1, 1, 100]
MAE  = 20.8    (reasonable)
RMSE = 44.7    (looks terrible)
```

**Solution**: Report both MAE and RMSE. Large difference indicates outlier influence.

---

## Residual Analysis

**Residuals** = Actual values - Predicted values = y_true - y_pred

### What Residuals Tell You

| Pattern | Interpretation | Action |
|---------|----------------|--------|
| Random scatter around 0 | Good model fit | None needed |
| Pattern/curve | Model missing non-linear relationship | Add polynomial features or use non-linear model |
| Funnel shape (heteroscedasticity) | Variance changes with predictions | Transform target, use weighted regression |
| Outliers | Extreme prediction errors | Investigate those data points |

### Key Residual Statistics

```
Mean Residual:    Should be ~0 (no systematic bias)
Std Residual:     Lower is better (prediction consistency)
```

- **Positive mean residual** → Model systematically under-predicts
- **Negative mean residual** → Model systematically over-predicts

---

## Metric Choice by Scenario

| Scenario | Primary Metric | Secondary | Why |
|----------|---------------|-----------|-----|
| **House Prices** | RMSE, R² | MAPE | Large errors matter, log-transform helps |
| **Stock Prices** | MAPE | MAE | Percentage matters across different stock prices |
| **Demand Forecasting** | MAE | MAPE | Zeros common (no MAPE), all errors matter |
| **Medical Measurements** | MAE, R² | RMSE | Outliers may be genuine, interpretability |
| **Energy Consumption** | RMSE | MAPE | Large prediction errors are costly |
| **Sales Prediction** | MAPE | MAE | Scale varies across products |

---

## Quick Reference

| Metric | Formula | Units | Outlier Sensitive? | Best For |
|--------|---------|-------|-------------------|----------|
| MAE | mean(\|errors\|) | Same as y | No | General use, outliers present |
| MSE | mean(errors²) | y² | Yes | Mathematical optimization |
| RMSE | √MSE | Same as y | Yes | Most common, interpretable |
| R² | 1 - SS_res/SS_tot | None | Somewhat | Model comparison |
| MAPE | mean(\|errors\|/\|y\|) × 100 | % | Depends | Cross-scale comparison |

---

## Baseline Models

Always compare against a baseline! The simplest regression baseline is predicting the **mean**:

```python
baseline_prediction = mean(y_train)  # For all samples
```

If your model doesn't beat this baseline, something is wrong:
- Features are uninformative
- Bug in your code
- Data leakage made training look good

A model with R² = 0 is exactly as good as predicting the mean.
A model with R² < 0 is WORSE than predicting the mean!
