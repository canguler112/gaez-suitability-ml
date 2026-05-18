import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_DIR = Path(r"C:\Users\cangu\OneDrive\Desktop\Agriculture")
ANALYSIS_DIR = PROJECT_DIR / r"outputs\analysis"
PRED_CSV = ANALYSIS_DIR / "xgb_rs04_block2p0_cv_predictions.csv"
OUT_CSV = ANALYSIS_DIR / "xgb_rs04_block2p0_crop_subgroups.csv"


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    print("SCRIPT STARTED")
    print("PRED_CSV =", PRED_CSV)
    print("PRED_CSV exists? ->", PRED_CSV.exists())

    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Prediction file not found: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)
    print("Loaded rows =", len(df))
    print("Crop counts:")
    print(df["crop"].value_counts(dropna=False))

    rows = []
    for crop_name, sub in df.groupby("crop"):
        m = compute_metrics(sub["observed"], sub["predicted"])
        rows.append({
            "crop": crop_name,
            "n": int(len(sub)),
            "r2": m["r2"],
            "rmse": m["rmse"],
            "mae": m["mae"],
            "mean_observed": float(sub["observed"].mean()),
            "mean_predicted": float(sub["predicted"].mean()),
            "mean_residual": float(sub["residual"].mean()),
            "mean_abs_error": float(sub["abs_error"].mean()),
        })

    out_df = pd.DataFrame(rows).sort_values("crop")
    out_df.to_csv(OUT_CSV, index=False)

    print("Saved crop subgroup metrics to:", OUT_CSV)
    print(out_df)
    print("DONE")


if __name__ == "__main__":
    main()