import os
import json
import glob
import argparse
from pathlib import Path

import pandas as pd


def load_json(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_metrics(d, fp):
    row = {
        "file": os.path.basename(fp),
        "run_name": d.get("run_name"),
        "model_name": d.get("model_name"),
        "n_rows": d.get("n_rows"),
        "n_features_used": d.get("n_features_used"),
        "drop_index_cols": d.get("drop_index_cols"),
        "spatial_block_deg": d.get("params", {}).get("spatial_block_deg"),
    }

    params = d.get("params", {})
    for k, v in params.items():
        if k not in row:
            row[k] = v

    rs = d.get("random_split", {})
    row.update({
        "random_r2": rs.get("r2"),
        "random_rmse": rs.get("rmse"),
        "random_mae": rs.get("mae"),
    })

    sp = d.get("spatial_cv", {})
    row.update({
        "spatial_r2_mean": sp.get("r2_mean"),
        "spatial_r2_std": sp.get("r2_std"),
        "spatial_rmse_mean": sp.get("rmse_mean"),
        "spatial_rmse_std": sp.get("rmse_std"),
        "spatial_mae_mean": sp.get("mae_mean"),
        "spatial_mae_std": sp.get("mae_std"),
        "spatial_n_splits": sp.get("n_splits"),
    })

    return row


def sort_runs(df):
    sort_cols = [c for c in ["model_name", "spatial_block_deg", "spatial_r2_mean"] if c in df.columns]
    if all(c in df.columns for c in ["model_name", "spatial_block_deg", "spatial_r2_mean"]):
        return df.sort_values(
            ["model_name", "spatial_block_deg", "spatial_r2_mean"],
            ascending=[True, True, False]
        )
    elif all(c in df.columns for c in ["model_name", "run_name"]):
        return df.sort_values(["model_name", "run_name"])
    return df


def print_best_by_model(df):
    if "spatial_r2_mean" not in df.columns:
        return

    spatial_df = df[df["spatial_r2_mean"].notna()].copy()
    if spatial_df.empty:
        return

    print("\n=== Best run by model and block size ===")
    best = (
        spatial_df.sort_values("spatial_r2_mean", ascending=False)
        .groupby(["model_name", "spatial_block_deg"], as_index=False)
        .first()
    )

    cols = [
        "model_name", "spatial_block_deg", "run_name",
        "spatial_r2_mean", "spatial_r2_std",
        "spatial_rmse_mean", "spatial_rmse_std",
        "spatial_mae_mean", "spatial_mae_std"
    ]
    cols = [c for c in cols if c in best.columns]
    print(best[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.metrics_dir, "*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No *_metrics.json files found in: {args.metrics_dir}")

    rows = []
    for fp in files:
        try:
            d = load_json(fp)
            rows.append(flatten_metrics(d, fp))
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    if not rows:
        raise RuntimeError("No metrics files could be parsed.")

    df = pd.DataFrame(rows)
    df = sort_runs(df)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n=== Aggregated metrics table ===")
    print(df.to_string(index=False))

    print_best_by_model(df)

    print(f"\nSaved CSV to: {out_csv}")


if __name__ == "__main__":
    main()