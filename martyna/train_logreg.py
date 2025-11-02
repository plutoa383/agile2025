#!/usr/bin/env python3
import argparse, json, os
from datetime import datetime, timezone
import joblib, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, brier_score_loss

RANDOM_STATE = 42

def eval_all(model, X, y):
    p = model.predict_proba(X)[:,1]
    yhat = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "brier": brier_score_loss(y, p),
        "report": classification_report(y, yhat, zero_division=0, digits=4),
    }

def main():
    ap = argparse.ArgumentParser(description="Balanced Logistic Regression (numeric-only, no leakage).")
    ap.add_argument("--input", default="uber_data_cleaned.csv")
    ap.add_argument("--artifacts_dir", default="artifacts_logreg")
    ap.add_argument("--target", default=None)
    args = ap.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    df = pd.read_csv(args.input)

    # Target
    target = args.target
    if target is None:
        if "Booking Status_Completed" in df.columns:
            df["success"] = (df["Booking Status_Completed"] == 1).astype(int)
            target = "success"
        else:
            raise ValueError("No --target and 'Booking Status_Completed' not found.")

    # Drop leak columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("Booking Status_")], errors="ignore")

    feats = [c for c in df.columns if c != target]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].astype(np.float32)
    y = df[target].astype(int)

    # Split 70/15/15
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])

    pipe.fit(X_train, y_train)
    val_metrics  = eval_all(pipe, X_val, y_val)
    test_metrics = eval_all(pipe, X_test, y_test)

    joblib.dump(pipe, os.path.join(args.artifacts_dir, "model.pkl"))
    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)
    with open(os.path.join(args.artifacts_dir, "metadata.json"), "w") as f:
        json.dump({
            "model_name": "LogisticRegression",
            "training_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feature_columns": num,
            "numeric_only": True,
            "artifacts_dir": args.artifacts_dir,
        }, f, indent=2)

    def pretty(d): return {k:(round(v,4) if isinstance(v,(float,np.floating)) else v) for k,v in d.items() if k!="report"}
    print("\nValidation:", pretty(val_metrics))
    print("\nTest:", pretty(test_metrics))
    print("\nSaved:", os.path.join(args.artifacts_dir,"model.pkl"))

if __name__ == "__main__":
    main()
