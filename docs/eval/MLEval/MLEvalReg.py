#!/usr/bin/env python3
"""
Regression Metrics Demo - Before & After Training
Usage: python MLEvalReg.py [--path data.csv] [--target col] [--rf]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')


def print_header(title):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def safe_mape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8))


def print_metrics(name, y_true, y_pred):
    """Print regression metrics."""
    print(f"\n--- {name} ---")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")


def print_residuals(y_true, y_pred, n_top=5):
    """Print residual summary."""
    residuals = y_true - y_pred
    print(f"\nResiduals: mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f}")
    
    abs_err = np.abs(residuals)
    top_idx = np.argsort(abs_err)[-n_top:][::-1]
    print(f"Top {n_top} errors: {[f'{abs_err[i]:.2f}' for i in top_idx]}")


def load_data(path, target_col):
    """Load from CSV or generate synthetic data."""
    if path:
        try:
            df = pd.read_csv(path)
            if target_col not in df.columns:
                print(f"Target '{target_col}' not found, using synthetic data")
                return None, None
            y = df[target_col].values.astype(float)
            X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
            return X, y
        except Exception as e:
            print(f"CSV error: {e}, using synthetic data")
            return None, None
    
    # Synthetic with noise + outliers
    X, y = make_regression(n_samples=1000, n_features=12, n_informative=8, noise=15, random_state=42)
    
    # Inject outliers (2%)
    n_out = int(len(y) * 0.02)
    outlier_idx = np.random.RandomState(42).choice(len(y), n_out, replace=False)
    y[outlier_idx] += np.random.RandomState(42).uniform(-3, 3, n_out) * np.std(y)
    
    print("Using synthetic data (1000 samples, 12 features, 2% outliers)")
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="CSV file path")
    parser.add_argument("--target", type=str, default="target", help="Target column")
    parser.add_argument("--rf", action="store_true", help="Also train RandomForest")
    args = parser.parse_args()
    
    X, y = load_data(args.path, args.target)
    if X is None:
        X, y = load_data(None, None)
    
    # Split: 70/15/15
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
    
    print(f"\nSamples: {len(y)} total | Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Target: mean={np.mean(y):.2f}, std={np.std(y):.2f}, range=[{np.min(y):.2f}, {np.max(y):.2f}]")
    
    # ===== BEFORE TRAINING =====
    print_header("BEFORE TRAINING (Baseline)")
    
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    print(f"Baseline predicts: {float(dummy.constant_[0]):.4f} (train mean)")
    print_metrics("DummyRegressor (mean)", y_test, dummy.predict(X_test))
    
    # ===== AFTER TRAINING =====
    print_header("AFTER TRAINING")
    
    pipe = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print_metrics("LinearRegression", y_test, y_pred)
    print_residuals(y_test, y_pred)
    
    # Optional RF
    if args.rf:
        rf = Pipeline([('scaler', StandardScaler()), ('reg', RandomForestRegressor(n_estimators=100, random_state=42))])
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        print_metrics("RandomForest", y_test, y_pred_rf)
        print_residuals(y_test, y_pred_rf)


if __name__ == "__main__":
    main()
