# rf_uber_all_columns_patched.py
# Usage examples:
#   py rf_uber_all_columns_patched.py --csv "uber_data_cleaned.csv" --target "Booking Status_Completed"
#   py rf_uber_all_columns_patched.py --csv "uber_data_cleaned.csv" --target "Booking Status_Completed" --n_estimators 1000 --jobs -1 --group_by_customer
#   py rf_uber_all_columns_patched.py --csv uber_data_cleaned.csv --target "Booking Status_Completed" --time_column Date

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

# Headless plotting to avoid Tk errors on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pip install category-encoders
from category_encoders import TargetEncoder

import joblib


def drop_leakage(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Drop columns that cause leakage:
    - Any 'Booking Status_*' columns except the chosen target
    - Obvious unique/transaction IDs (keep Customer ID ONLY for grouping; removed from features later)
    """
    LEAK_PREFIXES = ["Booking Status_"]  # keep the target separately
    LEAK_EXACT = [
        "Booking ID", "Driver ID", "Confirmation Code", "Trip ID", "Ride_ID",
        "Booking_ID", "Transaction ID"
    ]

    cols_to_drop = set()

    # Drop all status-derived columns except the target
    for c in df.columns:
        if any(c.startswith(p) for p in LEAK_PREFIXES) and c != target:
            cols_to_drop.add(c)

    # Drop explicit ID columns if present
    for c in LEAK_EXACT:
        if c in df.columns:
            cols_to_drop.add(c)

    if cols_to_drop:
        print("Dropping leakage columns:", sorted(cols_to_drop))
        df = df.drop(columns=list(cols_to_drop))

    return df


def build_preprocessor(X: pd.DataFrame):
    """Create a ColumnTransformer with numeric imputation and target encoding for categoricals."""
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # TargetEncoder outputs ONE numeric column per categorical feature
        ("target_enc", TargetEncoder(
            min_samples_leaf=20,
            smoothing=10.0,
            handle_unknown="value",
            handle_missing="value"
        ))
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preproc, num_cols, cat_cols


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    time_column: str | None,
    group_by_customer: bool,
    groups: pd.Series | None
):
    """
    Return X_train, X_test, y_train, y_test using:
    - time-based split (if time_column provided)
    - grouped split by customer (if requested and groups provided)
    - else stratified split
    """
    if time_column and time_column in X.columns:
        ts = pd.to_datetime(X[time_column], errors="coerce")
        order = np.argsort(ts.values)  # NaT sorts last; acceptable for split
        X_sorted = X.iloc[order].reset_index(drop=True)
        y_sorted = y.iloc[order].reset_index(drop=True)

        cutoff = int(len(X_sorted) * (1 - test_size))
        X_train, X_test = X_sorted.iloc[:cutoff], X_sorted.iloc[cutoff:]
        y_train, y_test = y_sorted.iloc[:cutoff], y_sorted.iloc[cutoff:]
        print(f"Time-based split on '{time_column}': train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    if group_by_customer and (groups is not None):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print(f"Grouped split: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Stratified split: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--target", required=True, help="Binary target column (0/1)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--jobs", type=int, default=-1, help="n_jobs for RF (-1 uses all cores)")
    ap.add_argument("--class_weight", default="balanced_subsample",
                    help="RF class_weight (None, 'balanced', 'balanced_subsample')")
    ap.add_argument("--time_column", default=None, help="Optional time column for time-based split")
    ap.add_argument("--group_by_customer", action="store_true",
                    help="Use grouped split so the same customer doesn't appear in train & test")
    ap.add_argument("--customer_col", default="Customer ID", help="Customer ID column name (if present)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Available columns: {list(df.columns)}")

    # Drop leakage columns
    df = drop_leakage(df, target=args.target)

    # Target and grouping series (keep groups from original df)
    y = df[args.target].astype(int)
    groups = df[args.customer_col] if (args.customer_col in df.columns) else None

    # Remove target and (later) Customer ID from features
    features = df.drop(columns=[args.target])

    # Exclude Customer ID from features (but keep 'groups' for split if present)
    if args.customer_col in features.columns:
        print(f"Excluding '{args.customer_col}' from features to avoid leakage.")
        features_no_cust = features.drop(columns=[args.customer_col])
    else:
        features_no_cust = features

    # Build preprocessing
    preproc, num_cols, cat_cols = build_preprocessor(features_no_cust)

    # RF model (accuracy-focused defaults)
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight=(None if args.class_weight == "None" else args.class_weight),
        n_jobs=args.jobs,
        random_state=args.random_state,
    )

    pipe = Pipeline(steps=[
        ("preproc", preproc),
        ("rf", rf)
    ])

    X = features_no_cust.copy()
    X_train, X_test, y_train, y_test = split_data(
        X=X, y=y,
        test_size=args.test_size,
        random_state=args.random_state,
        time_column=args.time_column,
        group_by_customer=args.group_by_customer,
        groups=groups
    )

    # Fit model
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            pass

    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Persist artifacts
    with open("rf_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print("Saved metrics to rf_metrics.json")

    if y_proba is not None:
        try:
            RocCurveDisplay.from_predictions(y_test, y_proba)
            plt.title("ROC Curve — Random Forest (Target-encoded)")
            plt.tight_layout()
            plt.savefig("rf_roc_curve.png", dpi=140)
            plt.close()
            print("Saved ROC curve to rf_roc_curve.png")
        except Exception:
            pass

    # Feature importances (one per numeric + one per categorical original column)
    rf_model = pipe.named_steps["rf"]
    out_feature_names = list(num_cols) + list(cat_cols)
    importances = rf_model.feature_importances_

    if len(importances) == len(out_feature_names):
        feat_imp = (
            pd.DataFrame({"feature": out_feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
        )
        print("\n=== Top Feature Importances (aggregated) ===")
        print(feat_imp.head(25).to_string(index=False))
        import os
os.makedirs("caoimhe/artifacts", exist_ok=True)
feat_imp.to_csv("caoimhe/artifacts/rf_feature_importances.csv", index=False)
print("Saved feature importances to caoimhe/artifacts/rf_feature_importances.csv")

        print("Saved feature importances to rf_feature_importances.csv")
    else:
        print("Warning: feature importances length mismatch; skipping CSV export.")

    # Save the trained pipeline for scoring new rides
    joblib.dump(pipe, "rf_pipeline.joblib")
    print("Saved model to rf_pipeline.joblib")


if __name__ == "__main__":
    main()
