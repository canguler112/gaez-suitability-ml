import csv
import json
import subprocess
import sys
from pathlib import Path

from sklearn.model_selection import ParameterSampler


# =========================================================
# FIXED PROJECT PATHS
# =========================================================
PYTHON_EXE = sys.executable

PROJECT_DIR = Path(r"C:\Users\cangu\OneDrive\Desktop\Agriculture")
TRAIN_SCRIPT = PROJECT_DIR / "train_baseline_rf.py"

DATA_DIR = PROJECT_DIR / r"data\processed\model_ready\wm100k_v1"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

SEARCH_OUT_DIR = OUTPUTS_DIR / "search"
SEARCH_OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FIXED REPRODUCIBLE SEARCH SETTINGS
# Same protocol as XGB random search:
# - random_state = 42
# - n_iter = 6
# - n_splits = 5
# - block sizes = [0.5, 2.0]
# - drop_index_cols = True
# =========================================================
RANDOM_STATE = 42
N_SPLITS = 5
DROP_INDEX_COLS = True
BLOCK_SIZES = [0.5, 2.0]
N_ITER = 6


# =========================================================
# RF-SPECIFIC SEARCH SPACE
# Chosen around the predefined RF configurations already tested.
# Note: max_depth=None is handled by omitting the CLI argument,
# because train_baseline_rf.py expects --max_depth as int if provided.
# =========================================================
PARAM_DISTRIBUTIONS = {
    "n_estimators": [300, 500, 800, 1000],
    "max_depth": [None, 20, 30],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", "log2"],
}


def block_tag(x: float) -> str:
    return str(x).replace(".", "p")


def sample_configs():
    """
    Deterministic random search sample.
    With fixed random_state, the same RF configs are sampled every rerun.
    """
    sampler = ParameterSampler(
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
    )
    return list(sampler)


def save_sampled_configs(configs):
    out_csv = SEARCH_OUT_DIR / "rf_random_search_sampled_configs.csv"
    fieldnames = [
        "config_id",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, cfg in enumerate(configs, start=1):
            row = {"config_id": f"rs{i:02d}"}
            row.update(cfg)
            writer.writerow(row)

    print(f"Saved sampled RF configs to: {out_csv}")
    return out_csv


def save_search_metadata(configs_csv_path):
    out_txt = SEARCH_OUT_DIR / "rf_random_search_metadata.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("RF constrained randomized search metadata\n")
        f.write("========================================\n")
        f.write(f"Train script: {TRAIN_SCRIPT}\n")
        f.write(f"Data dir: {DATA_DIR}\n")
        f.write(f"Outputs dir: {OUTPUTS_DIR}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"N splits: {N_SPLITS}\n")
        f.write(f"Drop index cols: {DROP_INDEX_COLS}\n")
        f.write(f"Block sizes: {BLOCK_SIZES}\n")
        f.write(f"N iter: {N_ITER}\n")
        f.write(f"Sampled configs csv: {configs_csv_path}\n")
        f.write("\nParameter distributions:\n")
        for k, v in PARAM_DISTRIBUTIONS.items():
            f.write(f"- {k}: {v}\n")

    print(f"Saved RF search metadata to: {out_txt}")


def build_command(config_id: str, cfg: dict, block_size: float):
    run_name = f"rf_{config_id}_block{block_tag(block_size)}_rs"

    cmd = [
        PYTHON_EXE,
        str(TRAIN_SCRIPT),
        "--data_dir", str(DATA_DIR),
        "--outputs_dir", str(OUTPUTS_DIR),
        "--run_name", run_name,
        "--random_state", str(RANDOM_STATE),
        "--spatial_block_deg", str(block_size),
        "--n_splits", str(N_SPLITS),
        "--n_estimators", str(cfg["n_estimators"]),
        "--min_samples_leaf", str(cfg["min_samples_leaf"]),
        "--max_features", str(cfg["max_features"]),
    ]

    # argparse in train_baseline_rf.py expects max_depth as int if supplied.
    # Therefore, for max_depth=None we omit the argument and let the script use default None.
    if cfg["max_depth"] is not None:
        cmd.extend(["--max_depth", str(cfg["max_depth"])])

    if DROP_INDEX_COLS:
        cmd.append("--drop_index_cols")

    return cmd


def run_one(config_id: str, cfg: dict, block_size: float):
    cmd = build_command(config_id, cfg, block_size)

    print("\n" + "=" * 110)
    print(f"Running RF random search | config={config_id} | block={block_size}")
    print("Parameters:", cfg)
    print("Command:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("=" * 110)

    subprocess.run(cmd, check=True)


def flatten_metrics(d, fp):
    row = {
        "file": Path(fp).name,
        "run_name": d.get("run_name"),
        "model_name": d.get("model_name"),
        "n_rows": d.get("n_rows"),
        "n_features_used": d.get("n_features_used"),
        "drop_index_cols": d.get("drop_index_cols"),
    }

    params = d.get("params", {})
    for k, v in params.items():
        row[k] = v

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


def aggregate_rf_search_results():
    metrics_dir = OUTPUTS_DIR / "metrics"
    files = sorted(metrics_dir.glob("rf_rs*_block*_rs_metrics.json"))

    if not files:
        print(f"[WARN] No RF random-search metrics files found in {metrics_dir}")
        return None

    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        rows.append(flatten_metrics(d, fp))

    out_csv = SEARCH_OUT_DIR / "rf_random_search_results.csv"

    fieldnames = sorted({k for row in rows for k in row.keys()})
    preferred = [
        "run_name", "model_name", "spatial_block_deg",
        "spatial_r2_mean", "spatial_r2_std",
        "spatial_rmse_mean", "spatial_rmse_std",
        "spatial_mae_mean", "spatial_mae_std",
        "n_estimators", "max_depth", "min_samples_leaf", "max_features",
        "random_state", "n_splits", "drop_index_cols", "n_features_used", "n_rows", "file"
    ]
    fieldnames = [c for c in preferred if c in fieldnames] + [c for c in fieldnames if c not in preferred]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved aggregated RF random-search results to: {out_csv}")

    # Print best model per block size based on spatial R2
    try:
        import pandas as pd
        df = pd.read_csv(out_csv)
        print("\n=== Best RF random-search run by block size ===")
        for block_size in sorted(df["spatial_block_deg"].dropna().unique()):
            sub = df[df["spatial_block_deg"] == block_size].copy()
            best = sub.sort_values("spatial_r2_mean", ascending=False).iloc[0]
            print(
                f"block={block_size}: {best['run_name']} | "
                f"R2={best['spatial_r2_mean']:.6f}, "
                f"RMSE={best['spatial_rmse_mean']:.3f}, "
                f"MAE={best['spatial_mae_mean']:.3f}"
            )
    except Exception as e:
        print("[WARN] Could not print best RF summary:", e)

    return out_csv


def main():
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find training script: {TRAIN_SCRIPT}")

    configs = sample_configs()
    configs_csv_path = save_sampled_configs(configs)
    save_search_metadata(configs_csv_path)

    for i, cfg in enumerate(configs, start=1):
        config_id = f"rs{i:02d}"
        for block_size in BLOCK_SIZES:
            run_one(config_id, cfg, block_size)

    aggregate_rf_search_results()
    print("\nRF constrained randomized search completed successfully.")


if __name__ == "__main__":
    main()
