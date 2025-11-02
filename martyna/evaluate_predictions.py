#!/usr/bin/env python3
"""
Evaluate how accurate the predictions were on new data (if you know the true outcomes).
"""

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def main():
    ap = argparse.ArgumentParser(description="Evaluate prediction accuracy using ground truth.")
    ap.add_argument("--input", default="new_predictions.csv", help="CSV containing predictions")
    ap.add_argument("--target", required=True, help="Column with true outcomes (0/1)")
    ap.add_argument("--prediction", default="predicted_success", help="Column with model predictions")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if args.target not in df.columns or args.prediction not in df.columns:
        raise ValueError(f"Missing '{args.target}' or '{args.prediction}' column in {args.input}")

    y_true = df[args.target]
    y_pred = df[args.prediction]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nPrediction Performance on New Data:")
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()
