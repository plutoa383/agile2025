import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

# === CONFIG ===
model_path = "rf_pipeline.joblib"
files_to_test = [
    ("new_data.csv", "scored_new_data.csv"),
    ("new_bad_data.csv", "scored_new_bad_data.csv"),
    ("new_unlikely_data.csv", "scored_new_unlikely_data.csv"),
]

print(f"Loading model from {model_path} ...")
model = joblib.load(model_path)

# Extract expected column names from the pipeline
expected_cols = model.named_steps["preproc"].feature_names_in_
print(f"Model expects {len(expected_cols)} columns.")

for input_file, output_file in files_to_test:
    print(f"\n=== Testing on {input_file} ===")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")

    # Add any missing columns with np.nan (not pd.NA)
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        print(f"⚠️  Missing columns added as empty: {missing_cols}")
        for c in missing_cols:
            df[c] = np.nan  # Use np.nan for sklearn compatibility

    # Reorder columns to match training
    df = df.reindex(columns=expected_cols)

    # Predict probabilities and binary outcomes
    try:
        proba = model.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        continue

    df["success_probability"] = proba
    df["predicted_success"] = pred
    df.to_csv(output_file, index=False)
    print(f"✅ Saved predictions to {output_file}")

    # If ground truth column exists, evaluate
    if "Booking Status_Completed" in df.columns:
        y_true = df["Booking Status_Completed"]
        acc = accuracy_score(y_true, pred)
        print(f"Accuracy vs true labels: {acc:.4f}")
        print(classification_report(y_true, pred, digits=3))
    else:
        avg_prob = proba.mean()
        print(f"Average predicted success probability: {avg_prob:.3f}")
        print(f"Predicted {sum(pred)} / {len(pred)} rides as successful (threshold=0.5)")
