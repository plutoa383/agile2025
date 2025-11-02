#!/usr/bin/env python3
"""
Train HistGradientBoostingClassifier with no leakage + upsampling + probability calibration.
Saves calibrated model so predict_proba is not saturated at ~1.0.
"""

import argparse, json, os
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42

def eval_all(model, X, y, pos=1):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)[:,1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label=pos, zero_division=0),
        "recall": recall_score(y, y_pred, pos_label=pos, zero_division=0),
        "f1": f1_score(y, y_pred, pos_label=pos, zero_division=0),
        "brier": brier_score_loss(y, proba),
        "report": classification_report(y, y_pred, zero_division=0, digits=4),
    }

def main():
    ap = argparse.ArgumentParser(description="Train calibrated HGB (numeric-only, no leakage).")
    ap.add_argument("--input", default="uber_data_cleaned.csv")
    ap.add_argument("--artifacts_dir", default="artifacts_calibrated")
    ap.add_argument("--target", default=None, help="If omitted, derive 'success' from Booking Status_Completed.")
    ap.add_argument("--balance", choices=["none","upsample"], default="upsample")
    ap.add_argument("--calibration", choices=["sigmoid","isotonic"], default="sigmoid")
    args = ap.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    df = pd.read_csv(args.input)

    # Target (no leakage)
    target = args.target
    if target is None:
        if "Booking Status_Completed" in df.columns:
            df["success"] = (df["Booking Status_Completed"] == 1).astype(int)
            target = "success"
        else:
            raise ValueError("No --target and 'Booking Status_Completed' not found.")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in input.")

    # Drop all Booking Status_* from features
    status_cols = [c for c in df.columns if c.startswith("Booking Status_")]
    df = df.drop(columns=status_cols, errors="ignore")

    # Numeric-only features
    feature_cols = [c for c in df.columns if c != target]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric features found.")
    df_model = df[numeric_cols + [target]].copy()

    # Optional: balance before splitting
    if args.balance == "upsample":
        maj = df_model[df_model[target] == 1]
        mino = df_model[df_model[target] == 0]
        if len(maj) and len(mino):
            if len(mino) < len(maj):
                min_up = resample(mino, replace=True, n_samples=len(maj), random_state=RANDOM_STATE)
                df_model = pd.concat([maj, min_up]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
            elif len(maj) < len(mino):
                maj_up = resample(maj, replace=True, n_samples=len(mino), random_state=RANDOM_STATE)
                df_model = pd.concat([mino, maj_up]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X_all = df_model[numeric_cols].astype(np.float32)
    y_all = df_model[target].astype(int)

    # Split: train / val / test
    X_tmp, X_test, y_tmp, y_test = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=RANDOM_STATE)

    # Base pipeline
    base = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.08,       # slightly gentler
            max_leaf_nodes=25,        # slightly simpler trees
            min_samples_leaf=30,      # smoother leaves
            random_state=RANDOM_STATE
        ))
    ])

    # Fit base on TRAIN only
    base.fit(X_train, y_train)

    # Calibrate on VAL only (no leakage), keep base fixed
    calib = CalibratedClassifierCV(base, method=args.calibration, cv="prefit")
    calib.fit(X_val, y_val)

    # Evaluate calibrated model
    val_metrics  = eval_all(calib, X_val, y_val)
    test_metrics = eval_all(calib, X_test, y_test)

    # Save
    model_path   = os.path.join(args.artifacts_dir, "model.pkl")
    metrics_path = os.path.join(args.artifacts_dir, "metrics.json")
    meta_path    = os.path.join(args.artifacts_dir, "metadata.json")

    joblib.dump(calib, model_path)
    with open(metrics_path, "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)

    metadata = {
        "model_name": "HistGradientBoostingClassifier+Calibrated",
        "library": "scikit-learn",
        "training_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "random_state": RANDOM_STATE,
        "target": target,
        "feature_columns": numeric_cols,
        "numeric_only": True,
        "dropped_status_columns": status_cols,
        "input_path": args.input,
        "artifacts_dir": args.artifacts_dir,
        "balance": args.balance,
        "calibration": args.calibration,
        "estimator_params": base.named_steps["clf"].get_params(),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    def pretty(d):
        return {k: (round(v,4) if isinstance(v,(float,np.floating)) else v) for k,v in d.items() if k!="report"}

    print("\nValidation:", pretty(val_metrics))
    print("\nTest:", pretty(test_metrics))
    print("\nSaved:", model_path, metrics_path, meta_path)
    print("\nFull test report (val):\n", val_metrics["report"])
    print("\nFull test report (test):\n", test_metrics["report"])

if __name__ == "__main__":
    main()
