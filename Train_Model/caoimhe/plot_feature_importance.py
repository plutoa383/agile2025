import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot feature importances from a CSV.")
    ap.add_argument("--csv", required=False,
                    help="Path to rf_feature_importances.csv. If omitted, tries local 'rf_feature_importances.csv' then 'caoimhe/artifacts/rf_feature_importances.csv'.")
    ap.add_argument("--top", type=int, default=15, help="How many top features to show (default: 15).")
    ap.add_argument("--out", default=None, help="Path to save PNG. Default: next to CSV, named feature_importance_plot.png")
    ap.add_argument("--title", default=None, help="Optional plot title")
    args = ap.parse_args()

    # Resolve CSV path
    candidates = []
    if args.csv:
        candidates.append(args.csv)
    candidates += ["rf_feature_importances.csv",
                   os.path.join("caoimhe", "artifacts", "rf_feature_importances.csv")]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if not csv_path:
        raise FileNotFoundError(
            "Could not find feature importance CSV. Tried:\n  " + "\n  ".join(candidates)
        )

    df = pd.read_csv(csv_path)

    # Be forgiving about column names (expect 'feature' and 'importance')
    cols_lower = [c.lower() for c in df.columns]
    try:
        feat_col = df.columns[cols_lower.index("feature")]
        imp_col  = df.columns[cols_lower.index("importance")]
    except ValueError:
        # Fallback: assume first two columns are feature, importance
        if df.shape[1] < 2:
            raise ValueError("CSV must have at least two columns: feature and importance.")
        feat_col, imp_col = df.columns[:2]

    # Prepare data
    df = df[[feat_col, imp_col]].copy()
    df.columns = ["feature", "importance"]
    df = df.sort_values("importance", ascending=False)
    df_top = df.head(max(1, args.top))

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_top["feature"], df_top["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(args.title or f"Top {len(df_top)} Feature Importances")

    # Label bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
                 va="center", ha="left")

    # Save
    out_path = args.out or os.path.join(os.path.dirname(csv_path), "feature_importance_plot.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"✅ Loaded: {csv_path}")
    print(f"✅ Saved plot to: {out_path}")

if __name__ == "__main__":
    main()
