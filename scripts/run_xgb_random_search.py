import csv
import subprocess
import sys
from pathlib import Path

from sklearn.model_selection import ParameterSampler


# =========================================================
# FIXED PROJECT PATHS
# =========================================================
PYTHON_EXE = sys.executable

PROJECT_DIR = Path(r"C:\Users\cangu\OneDrive\Desktop\Agriculture")
TRAIN_SCRIPT = PROJECT_DIR / "train_xgboost.py"


DATA_DIR = PROJECT_DIR / r"data\processed\model_ready\wm100k_v1"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

SEARCH_OUT_DIR = OUTPUTS_DIR / "search"
SEARCH_OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FIXED REPRODUCIBLE SEARCH SETTINGS
# =========================================================
RANDOM_STATE = 42
N_SPLITS = 5
DROP_INDEX_COLS = True
BLOCK_SIZES = [0.5, 2.0]
N_ITER = 6



# =========================================================
# CONSTRAINED, SCIENTIFICALLY DEFENSIBLE SEARCH SPACE
# Chosen around already reasonable XGB regions from prior runs.
# =========================================================
PARAM_DISTRIBUTIONS = {
    "n_estimators": [400, 500, 600, 800],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.07],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8],
    "reg_lambda": [0.5, 1.0, 2.0],
}


def block_tag(x: float) -> str:
    return str(x).replace(".", "p")


def sample_configs():
    """
    Deterministic random search sample.
    With fixed random_state, the same configs are sampled every rerun.
    """
    sampler = ParameterSampler(
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
    )
    return list(sampler)


def save_sampled_configs(configs):
    out_csv = SEARCH_OUT_DIR / "xgb_random_search_sampled_configs.csv"
    fieldnames = [
        "config_id",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, cfg in enumerate(configs, start=1):
            row = {"config_id": f"rs{i:02d}"}
            row.update(cfg)
            writer.writerow(row)

    print(f"Saved sampled configs to: {out_csv}")
    return out_csv


def save_search_metadata(configs_csv_path):
    out_txt = SEARCH_OUT_DIR / "xgb_random_search_metadata.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("XGB constrained randomized search metadata\n")
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

    print(f"Saved search metadata to: {out_txt}")


def build_command(config_id: str, cfg: dict, block_size: float):
    run_name = f"xgb_{config_id}_block{block_tag(block_size)}_rs"

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
        "--max_depth", str(cfg["max_depth"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--subsample", str(cfg["subsample"]),
        "--colsample_bytree", str(cfg["colsample_bytree"]),
        "--reg_lambda", str(cfg["reg_lambda"]),
    ]

    if DROP_INDEX_COLS:
        cmd.append("--drop_index_cols")

    return cmd


def run_one(config_id: str, cfg: dict, block_size: float):
    cmd = build_command(config_id, cfg, block_size)

    print("\n" + "=" * 110)
    print(f"Running XGB random search | config={config_id} | block={block_size}")
    print("Parameters:", cfg)
    print("Command:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("=" * 110)

    subprocess.run(cmd, check=True)



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

    
    print("\nXGB constrained randomized search completed successfully.")


if __name__ == "__main__":
    main()