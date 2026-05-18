# run_mean_baseline.py
#
# Purpose:
# Mean predictor baseline for the GAEZ suitability thesis.
#
# This script runs a naive mean baseline under the same spatial block
# cross-validation setup used for the final models:
#   - 0.5 degree spatial block CV
#   - 2.0 degree spatial block CV
#   - 5 folds
#
# Baseline definition:
#   For each fold, predict the mean suitability of the training fold
#   for every observation in the test fold.
#
# Outputs:
#   outputs/metrics/mean_baseline_block0p5_metrics.json
#   outputs/metrics/mean_baseline_block2p0_metrics.json
#   outputs/folds/mean_baseline_block0p5_spatialcv_folds.csv
#   outputs/folds/mean_baseline_block2p0_spatialcv_folds.csv
#   outputs/analysis/mean_baseline_block0p5_cv_predictions.csv
#   outputs/analysis/mean_baseline_block2p0_cv_predictions.csv
#   outputs/metrics/mean_baseline_summary.json

import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =========================================================
# PROJECT SETTINGS
# =========================================================

PROJECT_DIR = r"C:\Users\cangu\OneDrive\Desktop\Agriculture"

DATA_DIR = os.path.join(
    PROJECT_DIR,
    r"data\processed\model_ready\wm100k_v1"
)

OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

BLOCK_SIZES = [0.5, 2.0]
N_SPLITS = 5


# =========================================================
# HELPERS
# =========================================================

def ensure_dirs():
    for sub in ["metrics", "folds", "analysis"]:
        os.makedirs(os.path.join(OUTPUTS_DIR, sub), exist_ok=True)


def block_label(block_size: float) -> str:
    """
    Convert block size to filename-safe label.
    0.5 -> 0p5
    2.0 -> 2p0
    """
    return str(block_size).replace(".", "p")


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def make_spatial_groups(meta: pd.DataFrame, block_deg: float) -> np.ndarray:
    """
    Make spatial block IDs from lat/lon using the same simple block logic
    used in the thesis modelling pipeline.
    """
    lat_bin = np.floor((meta["lat"].values + 90.0) / block_deg).astype(int)
    lon_bin = np.floor((meta["lon"].values + 180.0) / block_deg).astype(int)

    groups = (
        pd.Series(lat_bin).astype(str)
        + "_"
        + pd.Series(lon_bin).astype(str)
    )

    return groups.values


def load_data():
    print("Loading data...", flush=True)

    y_path = os.path.join(DATA_DIR, "y.parquet")
    meta_path = os.path.join(DATA_DIR, "meta.parquet")

    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing y.parquet: {y_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.parquet: {meta_path}")

    y_df = pd.read_parquet(y_path)
    meta = pd.read_parquet(meta_path)

    if "suitability" not in y_df.columns:
        raise ValueError(f"Expected column 'suitability' in y.parquet. Found: {list(y_df.columns)}")

    y = y_df["suitability"].astype(float).reset_index(drop=True)
    meta = meta.reset_index(drop=True)

    print("y shape:", y.shape, flush=True)
    print("meta shape:", meta.shape, flush=True)
    print("Target min:", float(y.min()), flush=True)
    print("Target max:", float(y.max()), flush=True)
    print("Target mean:", float(y.mean()), flush=True)

    if len(y) != len(meta):
        raise ValueError(f"Length mismatch: y={len(y)}, meta={len(meta)}")

    return y, meta


def run_mean_baseline_for_block(y: pd.Series, meta: pd.DataFrame, block_size: float):
    label = block_label(block_size)
    run_name = f"mean_baseline_block{label}"

    print("\n" + "=" * 70, flush=True)
    print(f"Running mean predictor baseline: block size = {block_size}°", flush=True)
    print("=" * 70, flush=True)

    groups = make_spatial_groups(meta, block_size)
    n_groups = len(np.unique(groups))

    print("Number of spatial groups:", n_groups, flush=True)

    if n_groups < N_SPLITS:
        raise ValueError(
            f"Not enough spatial groups ({n_groups}) for {N_SPLITS}-fold CV."
        )

    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_rows = []
    prediction_rows = []

    y_array = y.values

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(np.zeros(len(y_array)), y_array, groups=groups),
        start=1
    ):
        print(f"\nFold {fold}/{N_SPLITS}", flush=True)

        y_train = y_array[train_idx]
        y_test = y_array[test_idx]

        train_mean = float(np.mean(y_train))
        y_pred = np.full(shape=len(test_idx), fill_value=train_mean, dtype=float)

        metrics = compute_metrics(y_test, y_pred)

        metrics["fold"] = int(fold)
        metrics["n_train"] = int(len(train_idx))
        metrics["n_test"] = int(len(test_idx))
        metrics["train_mean_prediction"] = train_mean
        metrics["test_mean_observed"] = float(np.mean(y_test))

        fold_rows.append(metrics)

        print(
            f"Fold {fold} done: "
            f"R2={metrics['r2']:.6f}, "
            f"RMSE={metrics['rmse']:.3f}, "
            f"MAE={metrics['mae']:.3f}, "
            f"train_mean={train_mean:.3f}",
            flush=True
        )

        fold_pred_df = pd.DataFrame({
            "fold": fold,
            "row_index": test_idx,
            "observed": y_test,
            "predicted": y_pred,
            "residual": y_test - y_pred,
            "lat": meta.iloc[test_idx]["lat"].values,
            "lon": meta.iloc[test_idx]["lon"].values,
            "crop": meta.iloc[test_idx]["crop"].values,
        })

        prediction_rows.append(fold_pred_df)

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.concat(prediction_rows, axis=0).sort_values("row_index").reset_index(drop=True)

    # Fold-mean summary, matching RF/XGB reporting style
    summary = {
        "r2_mean": float(folds_df["r2"].mean()),
        "r2_std": float(folds_df["r2"].std(ddof=1)),
        "rmse_mean": float(folds_df["rmse"].mean()),
        "rmse_std": float(folds_df["rmse"].std(ddof=1)),
        "mae_mean": float(folds_df["mae"].mean()),
        "mae_std": float(folds_df["mae"].std(ddof=1)),
        "n_splits": int(N_SPLITS),
    }

    # Overall out-of-sample metrics from concatenated CV predictions
    overall_metrics = compute_metrics(preds_df["observed"], preds_df["predicted"])

    summary["overall_cv_r2"] = overall_metrics["r2"]
    summary["overall_cv_rmse"] = overall_metrics["rmse"]
    summary["overall_cv_mae"] = overall_metrics["mae"]

    print("\nSummary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    folds_path = Path(OUTPUTS_DIR) / "folds" / f"{run_name}_spatialcv_folds.csv"
    preds_path = Path(OUTPUTS_DIR) / "analysis" / f"{run_name}_cv_predictions.csv"
    metrics_path = Path(OUTPUTS_DIR) / "metrics" / f"{run_name}_metrics.json"

    folds_df.to_csv(folds_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    metrics_json = {
        "run_name": run_name,
        "model_name": "Mean predictor baseline",
        "baseline_definition": (
            "For each spatial CV fold, all test observations are predicted "
            "as the mean suitability value of the training fold."
        ),
        "data_dir": DATA_DIR,
        "n_rows": int(len(y)),
        "spatial_block_deg": float(block_size),
        "n_splits": int(N_SPLITS),
        "fold_metrics": fold_rows,
        "spatial_cv_summary": summary,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    print("Saved fold metrics:", folds_path, flush=True)
    print("Saved CV predictions:", preds_path, flush=True)
    print("Saved metrics JSON:", metrics_path, flush=True)

    return run_name, summary


def main():
    print("SCRIPT STARTED: Mean predictor baseline", flush=True)
    print("PROJECT_DIR:", PROJECT_DIR, flush=True)
    print("DATA_DIR:", DATA_DIR, flush=True)
    print("OUTPUTS_DIR:", OUTPUTS_DIR, flush=True)
    print("BLOCK_SIZES:", BLOCK_SIZES, flush=True)

    ensure_dirs()

    y, meta = load_data()

    all_summaries = {}

    for block_size in BLOCK_SIZES:
        run_name, summary = run_mean_baseline_for_block(
            y=y,
            meta=meta,
            block_size=block_size,
        )
        all_summaries[run_name] = summary

    combined_path = Path(OUTPUTS_DIR) / "metrics" / "mean_baseline_summary.json"

    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("ALL DONE", flush=True)
    print("=" * 70, flush=True)
    print("Combined summary saved to:", combined_path, flush=True)
    print("\nFinal summary:", flush=True)
    print(json.dumps(all_summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()