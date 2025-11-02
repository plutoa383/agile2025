#!/usr/bin/env python3
import argparse, json, os
import joblib
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Generate predictions with a trained model.")
    ap.add_argument("--input", required=True, help="Path to input CSV with new data.")
    ap.add_argument("--output", default="new_predictions.csv", help="Output CSV path.")
    ap.add_argument("--artifacts_dir", default="artifacts_logreg",
                    help="Directory containing model.pkl and metadata.json.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Optional probability threshold (0..1) to add a Predicted_at_{THRESH} column.")
    args = ap.parse_args()

    model_path = os.path.join(args.artifacts_dir, "model.pkl")
    meta_path = os.path.join(args.artifacts_dir, "metadata.json")

    # Load model and metadata
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    model = joblib.load(model_path)
    metadata = json.load(open(meta_path))

    expected_features = metadata["feature_columns"]

    print(f"\nUsing model from: {args.artifacts_dir}")
    print(f"Expected {len(expected_features)} numeric features")

    # Load input
    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]]

    # Align with expected columns
    missing = [c for c in expected_features if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_features]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        print(f"Ignoring {len(extra)} extra columns: {extra}")

    X = df[expected_features].astype(np.float32, errors="ignore")

    # Predict
    preds = model.predict(X).astype(int)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

    # Build output
    out = pd.DataFrame({
        "Predicted_Success": preds,
        "Confidence(%)": (proba * 100).round(2)
    })

    # Optional threshold
    if args.threshold is not None:
        if not (0.0 <= args.threshold <= 1.0):
            raise ValueError("--threshold must be between 0 and 1 (e.g., 0.8)")
        thr_col = f"Predicted_at_{args.threshold:.2f}"
        out[thr_col] = (proba >= args.threshold).astype(int)

    out.to_csv(args.output, index=False)

    # Console summary
    pd.options.display.float_format = "{:.2f}".format
    avg_conf = float(np.mean(proba) * 100.0)
    success_rate = float(np.mean(preds) * 100.0)
    n_rows = len(out)

    print("\nPrediction Summary")
    print(f"Rows predicted: {n_rows}")
    print(f"Average Confidence: {avg_conf:.2f}%")
    print(f"Predicted Success Rate: {success_rate:.1f}% ({int(np.sum(preds))}/{n_rows} rows)")
    if args.threshold is not None:
        thr_col = f"Predicted_at_{args.threshold:.2f}"
        thr_rate = out[thr_col].mean() * 100.0
        print(f"Success Rate @ threshold {args.threshold:.2f}: {thr_rate:.1f}%")

    print("\nPredictions saved to:", args.output)
    print("\nPreview:")
    print(out.head())

if __name__ == "__main__":
    main()
