import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


# =========================================================
# FIXED PROJECT PATHS
# =========================================================
PROJECT_DIR = Path(r"C:\Users\cangu\OneDrive\Desktop\Agriculture")
DATA_DIR = PROJECT_DIR / r"data\processed\model_ready\wm100k_v1"
OUTPUT_DIR = PROJECT_DIR / r"outputs\analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "xgb_rs04_block2p0_cv_predictions.csv"


# =========================================================
# FIXED FINAL MODEL: XGB-rs04
# =========================================================
RANDOM_STATE = 42
BLOCK_DEG = 2.0
N_SPLITS = 5
DROP_INDEX_COLS = True

N_ESTIMATORS = 800
MAX_DEPTH = 10
LEARNING_RATE = 0.03
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.7
REG_LAMBDA = 2.0


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def make_spatial_groups(meta, block_deg):
    lat_bin = np.floor((meta["lat"].values + 90.0) / block_deg).astype(int)
    lon_bin = np.floor((meta["lon"].values + 180.0) / block_deg).astype(int)
    groups = pd.Series(lat_bin).astype(str) + "_" + pd.Series(lon_bin).astype(str)
    return groups.values


def build_model():
    return XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_lambda=REG_LAMBDA,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main():
    X = pd.read_parquet(DATA_DIR / "X.parquet")
    y = pd.read_parquet(DATA_DIR / "y.parquet")["suitability"].reset_index(drop=True)
    meta = pd.read_parquet(DATA_DIR / "meta.parquet").reset_index(drop=True)

    always_drop_cols = [c for c in ["cell_id"] if c in X.columns]
    optional_index_cols = [c for c in ["row", "col"] if c in X.columns]

    drop_cols = always_drop_cols.copy()
    if DROP_INDEX_COLS:
        drop_cols.extend(optional_index_cols)

    X_model = X.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)

    groups = make_spatial_groups(meta, BLOCK_DEG)
    gkf = GroupKFold(n_splits=N_SPLITS)

    pred_rows = []
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_model, y, groups=groups), start=1):
        X_tr = X_model.iloc[tr_idx]
        X_te = X_model.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        model = build_model()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        fold_m = compute_metrics(y_te, preds)
        fold_m["fold"] = fold
        fold_m["n_train"] = int(len(tr_idx))
        fold_m["n_test"] = int(len(te_idx))
        fold_metrics.append(fold_m)

        meta_te = meta.iloc[te_idx].reset_index(drop=True)

        fold_df = pd.DataFrame({
            "fold": fold,
            "row_index": te_idx,
            "crop": meta_te["crop"].values,
            "lat": meta_te["lat"].values,
            "lon": meta_te["lon"].values,
            "observed": y_te.values,
            "predicted": preds,
        })
        fold_df["residual"] = fold_df["observed"] - fold_df["predicted"]
        fold_df["abs_error"] = (fold_df["observed"] - fold_df["predicted"]).abs()
        fold_df["squared_error"] = (fold_df["observed"] - fold_df["predicted"]) ** 2
        pred_rows.append(fold_df)

    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df.to_csv(OUTPUT_CSV, index=False)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_path = OUTPUT_DIR / "xgb_rs04_block2p0_cv_predictions_fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    overall = compute_metrics(pred_df["observed"], pred_df["predicted"])
    overall_path = OUTPUT_DIR / "xgb_rs04_block2p0_cv_predictions_overall_metrics.txt"
    with open(overall_path, "w", encoding="utf-8") as f:
        f.write("Final model: XGB-rs04\n")
        f.write(f"n_estimators={N_ESTIMATORS}\n")
        f.write(f"max_depth={MAX_DEPTH}\n")
        f.write(f"learning_rate={LEARNING_RATE}\n")
        f.write(f"subsample={SUBSAMPLE}\n")
        f.write(f"colsample_bytree={COLSAMPLE_BYTREE}\n")
        f.write(f"reg_lambda={REG_LAMBDA}\n")
        f.write(f"block_deg={BLOCK_DEG}\n")
        f.write(f"n_splits={N_SPLITS}\n")
        f.write(f"rows={len(pred_df)}\n")
        f.write(f"R2={overall['r2']:.6f}\n")
        f.write(f"RMSE={overall['rmse']:.6f}\n")
        f.write(f"MAE={overall['mae']:.6f}\n")

    print("Saved predictions to:", OUTPUT_CSV)
    print("Saved fold metrics to:", fold_metrics_path)
    print("Saved overall metrics to:", overall_path)
    print("Overall metrics:", overall)


if __name__ == "__main__":
    main()