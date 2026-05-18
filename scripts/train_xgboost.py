import os
import json
import math
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def ensure_dirs(outputs_dir):
    for sub in ["metrics", "folds", "models", "feature_importance"]:
        os.makedirs(os.path.join(outputs_dir, sub), exist_ok=True)


def make_spatial_groups(meta, block_deg):
    lat_bin = np.floor((meta["lat"].values + 90.0) / block_deg).astype(int)
    lon_bin = np.floor((meta["lon"].values + 180.0) / block_deg).astype(int)
    groups = pd.Series(lat_bin).astype(str) + "_" + pd.Series(lon_bin).astype(str)
    return groups.values


def build_model(args):
    return XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )


def run_random_split(model, X, y, random_state=42, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    return model, metrics


def run_spatial_cv(model_builder, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        model = model_builder()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        m = compute_metrics(y_te, preds)
        m["fold"] = fold
        m["n_train"] = int(len(tr_idx))
        m["n_test"] = int(len(te_idx))
        rows.append(m)

    folds_df = pd.DataFrame(rows)
    summary = {
        "r2_mean": float(folds_df["r2"].mean()),
        "r2_std": float(folds_df["r2"].std(ddof=1)),
        "rmse_mean": float(folds_df["rmse"].mean()),
        "rmse_std": float(folds_df["rmse"].std(ddof=1)),
        "mae_mean": float(folds_df["mae"].mean()),
        "mae_std": float(folds_df["mae"].std(ddof=1)),
        "n_splits": int(n_splits),
    }
    return folds_df, summary


def save_feature_importance(model, feature_names, out_csv):
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--outputs_dir", required=True)
    parser.add_argument("--run_name", required=True)

    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--run_random_split", action="store_true")
    parser.add_argument("--spatial_block_deg", type=float, default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--drop_index_cols", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)

    # XGBoost hyperparameters
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_lambda", type=float, default=1.0)

    args = parser.parse_args()

    ensure_dirs(args.outputs_dir)

    data_dir = Path(args.data_dir)
    X = pd.read_parquet(data_dir / "X.parquet")
    y = pd.read_parquet(data_dir / "y.parquet")["suitability"]
    meta = pd.read_parquet(data_dir / "meta.parquet")

    always_drop_cols = [c for c in ["cell_id"] if c in X.columns]
    optional_index_cols = [c for c in ["row", "col"] if c in X.columns]

    drop_cols = always_drop_cols.copy()
    if args.drop_index_cols:
        drop_cols.extend(optional_index_cols)

    X_model = X.drop(columns=drop_cols, errors="ignore").copy()

    print("Dropped columns before modeling:", drop_cols)
    print("X original shape:", X.shape)
    print("X model shape:", X_model.shape)

    metrics_json = {
        "run_name": args.run_name,
        "model_name": "XGBRegressor",
        "data_dir": str(data_dir),
        "n_rows": int(len(X_model)),
        "n_features_used": int(X_model.shape[1]),
        "always_dropped_cols": always_drop_cols,
        "drop_index_cols": bool(args.drop_index_cols),
        "optional_index_cols_dropped": optional_index_cols if args.drop_index_cols else [],
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_lambda": args.reg_lambda,
            "random_state": args.random_state,
            "test_size": args.test_size,
            "n_splits": args.n_splits,
            "spatial_block_deg": args.spatial_block_deg,
        }
    }

    if args.run_random_split:
        model = build_model(args)
        fitted_model, random_metrics = run_random_split(
            model, X_model, y,
            random_state=args.random_state,
            test_size=args.test_size
        )
        metrics_json["random_split"] = random_metrics

        model_path = Path(args.outputs_dir) / "models" / f"{args.run_name}_randomsplit.joblib"
        joblib.dump(fitted_model, model_path)

        fi_path = Path(args.outputs_dir) / "feature_importance" / f"{args.run_name}_feature_importance_randomsplit.csv"
        save_feature_importance(fitted_model, X_model.columns.tolist(), fi_path)

    if args.spatial_block_deg is not None:
        groups = make_spatial_groups(meta, args.spatial_block_deg)

        folds_df, spatial_summary = run_spatial_cv(
            lambda: build_model(args),
            X_model,
            y,
            groups=groups,
            n_splits=args.n_splits
        )
        metrics_json["spatial_cv"] = spatial_summary

        folds_path = Path(args.outputs_dir) / "folds" / f"{args.run_name}_spatialcv_folds.csv"
        folds_df.to_csv(folds_path, index=False)

        full_model = build_model(args)
        full_model.fit(X_model, y)

        full_model_path = Path(args.outputs_dir) / "models" / f"{args.run_name}_fullfit.joblib"
        joblib.dump(full_model, full_model_path)

        fi_full_path = Path(args.outputs_dir) / "feature_importance" / f"{args.run_name}_feature_importance_fullfit.csv"
        save_feature_importance(full_model, X_model.columns.tolist(), fi_full_path)

    metrics_path = Path(args.outputs_dir) / "metrics" / f"{args.run_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()