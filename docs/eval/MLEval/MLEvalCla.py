#!/usr/bin/env python3
"""
Classification Metrics Demo - Before & After Training
Usage: python MLEvalCla.py [--path data.csv] [--target col] [--rf]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, log_loss,
    classification_report
)

warnings.filterwarnings('ignore')


def print_header(title):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def print_metrics(name, y_true, y_pred, y_proba=None):
    """Print classification metrics."""
    print(f"\n--- {name} ---")
    
    labels = np.unique(y_true)
    is_binary = len(labels) == 2
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    if is_binary:
        tn, fp, fn, tp = cm.ravel()
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    # Core metrics
    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
    
    if is_binary:
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    else:
        print(f"F1 (macro):    {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
        print(f"F1 (weighted): {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    
    # Probability metrics
    if y_proba is not None and is_binary:
        proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        print(f"Log Loss:  {log_loss(y_true, y_proba):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_true, proba_pos):.4f}")
        print(f"PR-AUC:    {average_precision_score(y_true, proba_pos):.4f}")


def tune_threshold(y_val, proba_val):
    """Find threshold that maximizes F1 on validation set."""
    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.02):
        f1 = f1_score(y_val, (proba_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh


def load_data(path, target_col):
    """Load from CSV or generate synthetic data."""
    if path:
        try:
            df = pd.read_csv(path)
            if target_col not in df.columns:
                print(f"Target '{target_col}' not found, using synthetic data")
                return None, None
            y = df[target_col].values
            X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
            return X, y
        except Exception as e:
            print(f"CSV error: {e}, using synthetic data")
            return None, None
    
    # Synthetic: 80/20 imbalance (not too extreme)
    X, y = make_classification(
        n_samples=1500, n_features=15, n_informative=10, n_redundant=3,
        weights=[0.80, 0.20], flip_y=0.02, class_sep=1.2, random_state=42
    )
    print("Using synthetic data (80/20 class split, 15 features)")
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
    
    labels = np.unique(y)
    is_binary = len(labels) == 2
    
    # Split: 70/15/15
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)
    
    print(f"\nSamples: {len(y)} total | Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # ===== BEFORE TRAINING =====
    print_header("BEFORE TRAINING (Baselines)")
    
    dummy_mf = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    print_metrics("DummyClassifier (most_frequent)", y_test, dummy_mf.predict(X_test))
    
    dummy_st = DummyClassifier(strategy="stratified", random_state=42).fit(X_train, y_train)
    y_pred_st = dummy_st.predict(X_test)
    y_proba_st = dummy_st.predict_proba(X_test) if hasattr(dummy_st, 'predict_proba') else None
    print_metrics("DummyClassifier (stratified)", y_test, y_pred_st, y_proba_st)
    
    # ===== AFTER TRAINING =====
    print_header("AFTER TRAINING")
    
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    print_metrics("LogisticRegression", y_test, y_pred, y_proba)
    
    # Threshold tuning (binary only)
    if is_binary:
        proba_val = pipe.predict_proba(X_val)[:, 1]
        best_t = tune_threshold(y_val, proba_val)
        y_pred_tuned = (y_proba[:, 1] >= best_t).astype(int)
        print(f"\n--- LogisticRegression (threshold={best_t:.2f}) ---")
        print(f"F1: {f1_score(y_test, y_pred_tuned, zero_division=0):.4f}")
    
    # Optional RF
    if args.rf:
        rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
        rf.fit(X_train, y_train)
        print_metrics("RandomForest", y_test, rf.predict(X_test), rf.predict_proba(X_test))


if __name__ == "__main__":
    main()
