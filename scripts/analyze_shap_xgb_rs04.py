# analyze_shap_xgb_rs04.py
#
# Purpose:
# Feature contribution analysis for the FINAL XGB-rs04 model.
#
# This script uses native XGBoost feature contributions:
#     booster.predict(DMatrix, pred_contribs=True)
#
# This is used as a practical SHAP-style explanation method.
#
# Outputs:
#   outputs/shap/xgb_rs04_block2p0/
#       - top_features_all.csv
#       - grouped_importance.csv
#       - is_wheat_summary.csv
#       - top_features_wheat.csv
#       - top_features_maize.csv
#       - X_sample_used.csv
#       - meta_sample_used.csv
#       - bias_values.csv
#       - run_metadata.json
#
# Run:
#   conda activate gaez
#   python analyze_shap_xgb_rs04.py

import os
import json
import glob
import warnings
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

print("SCRIPT FILE LOADED")


# =========================================================
# FIXED THESIS SETTINGS
# =========================================================
PROJECT_DIR = r"C:\Users\cangu\OneDrive\Desktop\Agriculture"

DATA_DIR = os.path.join(
    PROJECT_DIR,
    r"data\processed\model_ready\wm100k_v1"
)

# Final XGB-rs04 model.
# If this exact file name does not exist, the script will search for likely rs04 model files.
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    r"outputs\models\xgb_rs04_block2p0_rs_fullfit.joblib"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    r"outputs\shap\xgb_rs04_block2p0"
)

SAMPLE_SIZE = 5000
RANDOM_STATE = 42
DROP_INDEX_COLS = True
MAX_DISPLAY = 20


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_parquets(data_dir: str):
    x_path = os.path.join(data_dir, "X.parquet")
    y_path = os.path.join(data_dir, "y.parquet")
    meta_path = os.path.join(data_dir, "meta.parquet")

    print("Checking parquet paths...")
    print("X path:", x_path, "exists:", os.path.exists(x_path))
    print("y path:", y_path, "exists:", os.path.exists(y_path))
    print("meta path:", meta_path, "exists:", os.path.exists(meta_path))

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"X.parquet not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"y.parquet not found: {y_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.parquet not found: {meta_path}")

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)
    meta = pd.read_parquet(meta_path)

    return X, y, meta


def drop_feature_columns(X: pd.DataFrame, drop_index_cols: bool):
    """
    Match thesis modelling feature set:
    - Always drop cell_id if present.
    - Drop row and col if DROP_INDEX_COLS=True.
    """
    X = X.copy()

    dropped = []

    if "cell_id" in X.columns:
        X = X.drop(columns=["cell_id"])
        dropped.append("cell_id")

    if drop_index_cols:
        cols_to_drop = [c for c in ["row", "col"] if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            dropped.extend(cols_to_drop)

    print("Dropped feature columns:", dropped)
    return X, dropped


def find_candidate_model_files(project_dir: str):
    """
    Search for likely final XGB-rs04 model files if MODEL_PATH is not found.
    """
    models_dir = os.path.join(project_dir, "outputs", "models")

    patterns = [
        "*xgb*rs04*block2p0*fullfit*.joblib",
        "*xgb*rs04*2p0*fullfit*.joblib",
        "*xgb*rs04*.joblib",
        "*xgb*block2p0*rs*fullfit*.joblib",
    ]

    candidates = []
    for pattern in patterns:
        found = glob.glob(os.path.join(models_dir, pattern))
        candidates.extend(found)

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    return unique_candidates


def resolve_model_file(model_path: str):
    """
    Resolve the model path.
    If the expected MODEL_PATH does not exist, search for likely XGB-rs04 model files.
    """
    path = Path(model_path)

    print("Resolving model path:", model_path)
    print("Expected model exists?", path.exists())

    if path.is_file():
        return str(path)

    print("\nExpected model file not found.")
    print("Searching for candidate XGB-rs04 model files...")

    candidates = find_candidate_model_files(PROJECT_DIR)

    if not candidates:
        raise FileNotFoundError(
            "Could not find the final XGB-rs04 model file.\n"
            f"Expected path was:\n{model_path}\n\n"
            "No candidate files were found in outputs/models.\n"
            "Please check the exact model filename in:\n"
            f"{os.path.join(PROJECT_DIR, 'outputs', 'models')}"
        )

    print("Candidate model files found:")
    for i, c in enumerate(candidates, start=1):
        print(f"{i}. {c}")

    chosen = candidates[0]
    print("Using first candidate:", chosen)

    return chosen


def load_model(model_path: str):
    """
    Load model and return native XGBoost Booster.
    Supports joblib/pkl/sklearn XGBRegressor or native XGBoost model files.
    """
    model_file = resolve_model_file(model_path)
    suffix = Path(model_file).suffix.lower()

    print("\nLoading model file:", model_file)
    print("Model suffix:", suffix)

    if suffix in [".joblib", ".pkl", ".pickle"]:
        model = joblib.load(model_file)

        if hasattr(model, "get_booster"):
            print("Loaded sklearn XGBRegressor. Using native booster via get_booster().")
            booster = model.get_booster()
            return booster, model_file

        if isinstance(model, xgb.Booster):
            print("Loaded native XGBoost Booster.")
            return model, model_file

        raise TypeError(
            f"Loaded object is not an XGBRegressor or xgb.Booster. Type: {type(model)}"
        )

    if suffix in [".json", ".ubj", ".bin"]:
        model = XGBRegressor()
        model.load_model(model_file)

        if hasattr(model, "get_booster"):
            print("Loaded native XGBoost model. Using booster via get_booster().")
            booster = model.get_booster()
            return booster, model_file

    raise ValueError(f"Unsupported model file format: {model_file}")


def sample_rows(
    X: pd.DataFrame,
    y: pd.DataFrame,
    meta: pd.DataFrame,
    sample_size: int,
    random_state: int
):
    """
    Take a deterministic sample for contribution analysis.
    """
    n = len(X)
    print("Sampling rows. Full n =", n)

    if sample_size >= n:
        print("Sample size >= full data size. Using all rows.")
        return X.copy(), y.copy(), meta.copy()

    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=sample_size, replace=False)
    idx = np.sort(idx)

    print("Sampled n =", len(idx))

    return (
        X.iloc[idx].reset_index(drop=True),
        y.iloc[idx].reset_index(drop=True),
        meta.iloc[idx].reset_index(drop=True),
    )


def compute_xgb_contributions(booster, X_sample: pd.DataFrame):
    """
    Compute native XGBoost feature contributions.

    Output from pred_contribs=True has shape:
        n_samples x (n_features + 1)

    The last column is the bias/base value.
    """
    print("\nPreparing DMatrix for XGBoost contributions...")

    dmatrix = xgb.DMatrix(
        X_sample,
        feature_names=list(X_sample.columns)
    )

    print("Computing XGBoost pred_contribs=True...")
    contribs = booster.predict(
        dmatrix,
        pred_contribs=True,
        validate_features=True
    )

    contribs = np.asarray(contribs)

    expected_cols = X_sample.shape[1] + 1

    print("Contribution matrix raw shape:", contribs.shape)
    print("Expected columns:", expected_cols)

    if contribs.ndim != 2 or contribs.shape[1] != expected_cols:
        raise ValueError(
            f"Unexpected contribution shape: {contribs.shape}. "
            f"Expected (?, {expected_cols})."
        )

    feature_contribs = contribs[:, :-1]
    bias_values = contribs[:, -1]

    contrib_df = pd.DataFrame(
        feature_contribs,
        columns=X_sample.columns
    )

    print("Feature contribution matrix shape:", contrib_df.shape)
    print("Bias vector shape:", bias_values.shape)

    return contrib_df, bias_values


def feature_importance_table(X_sample: pd.DataFrame, contrib_df: pd.DataFrame):
    """
    Rank features by mean absolute contribution.
    """
    mean_abs_contrib = contrib_df.abs().mean(axis=0).values

    df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_contrib": mean_abs_contrib
    })

    df = df.sort_values("mean_abs_contrib", ascending=False).reset_index(drop=True)

    total = df["mean_abs_contrib"].sum()
    if total > 0:
        df["importance_pct"] = 100 * df["mean_abs_contrib"] / total
    else:
        df["importance_pct"] = 0.0

    return df


def classify_feature_group(feature_name: str):
    """
    Group features into:
    - crop_indicator
    - soil
    - climate
    - other
    """
    f = str(feature_name).lower()

    if f == "is_wheat":
        return "crop_indicator"

    soil_keywords = [
        "soil", "sg_", "clay", "sand", "silt", "soc", "ocd", "oc",
        "bdod", "cfvo", "cec", "nitrogen", "phh2o", "wv", "texture"
    ]

    if any(k in f for k in soil_keywords):
        return "soil"

    climate_keywords = [
        "t2m", "t2m_max", "t2m_min", "t2mdew",
        "rh2m", "prectotcorr", "prectot",
        "ws2m", "allsky", "clrsky",
        "temp", "temperature",
        "prec", "precip", "rain",
        "rad", "radiation", "solar",
        "wind", "humidity",
        "vpd", "evap", "pet", "dew", "frost"
    ]

    if any(k in f for k in climate_keywords):
        return "climate"

    return "other"


def grouped_importance_table(importance_df: pd.DataFrame):
    """
    Sum mean absolute contributions by feature group.
    """
    df = importance_df.copy()
    df["group"] = df["feature"].apply(classify_feature_group)

    grouped = (
        df.groupby("group", as_index=False)["mean_abs_contrib"]
        .sum()
        .sort_values("mean_abs_contrib", ascending=False)
        .reset_index(drop=True)
    )

    total = grouped["mean_abs_contrib"].sum()
    if total > 0:
        grouped["importance_pct"] = 100 * grouped["mean_abs_contrib"] / total
    else:
        grouped["importance_pct"] = 0.0

    return grouped


def is_wheat_summary(X_sample: pd.DataFrame, contrib_df: pd.DataFrame):
    """
    Summarize contribution of the binary crop indicator.
    """
    if "is_wheat" not in X_sample.columns:
        return pd.DataFrame([{
            "feature_present": False,
            "message": "is_wheat column not found in X_sample"
        }])

    vals = contrib_df["is_wheat"].values
    feat = X_sample["is_wheat"].values

    mask_wheat = feat == 1
    mask_maize = feat == 0

    out = {
        "feature_present": True,
        "overall_mean_contrib": float(np.mean(vals)),
        "overall_mean_abs_contrib": float(np.mean(np.abs(vals))),
        "wheat_rows": int(mask_wheat.sum()),
        "maize_rows": int(mask_maize.sum()),
        "mean_contrib_when_is_wheat_1": float(np.mean(vals[mask_wheat])) if mask_wheat.sum() > 0 else np.nan,
        "mean_contrib_when_is_wheat_0": float(np.mean(vals[mask_maize])) if mask_maize.sum() > 0 else np.nan,
        "mean_abs_contrib_when_is_wheat_1": float(np.mean(np.abs(vals[mask_wheat]))) if mask_wheat.sum() > 0 else np.nan,
        "mean_abs_contrib_when_is_wheat_0": float(np.mean(np.abs(vals[mask_maize]))) if mask_maize.sum() > 0 else np.nan,
    }

    return pd.DataFrame([out])


def subset_top_features(
    X_sample: pd.DataFrame,
    contrib_df: pd.DataFrame,
    meta_sample: pd.DataFrame,
    crop_name: str
):
    """
    Compute top feature contribution table for one crop.
    """
    if "crop" not in meta_sample.columns:
        print("meta_sample has no crop column. Skipping crop-specific top features.")
        return pd.DataFrame(columns=["feature", "mean_abs_contrib", "importance_pct"])

    mask = meta_sample["crop"].astype(str).str.lower() == crop_name.lower()

    if mask.sum() == 0:
        print(f"No rows found for crop: {crop_name}")
        return pd.DataFrame(columns=["feature", "mean_abs_contrib", "importance_pct"])

    X_sub = X_sample.loc[mask].reset_index(drop=True)
    contrib_sub = contrib_df.loc[mask].reset_index(drop=True)

    return feature_importance_table(X_sub, contrib_sub)


def save_metadata(
    output_dir: str,
    model_file: str,
    dropped_columns,
    full_x_shape,
    model_x_shape,
    sample_x_shape,
    bias_values
):
    meta = {
        "analysis_name": "xgb_rs04_native_feature_contributions",
        "model": "XGB-rs04",
        "data_dir": DATA_DIR,
        "expected_model_path": MODEL_PATH,
        "resolved_model_file": model_file,
        "output_dir": OUTPUT_DIR,
        "sample_size_requested": SAMPLE_SIZE,
        "random_state": RANDOM_STATE,
        "drop_index_cols": DROP_INDEX_COLS,
        "dropped_columns": dropped_columns,
        "max_display": MAX_DISPLAY,
        "full_X_shape_raw_saved": list(full_x_shape),
        "X_shape_after_column_drops": list(model_x_shape),
        "sample_X_shape": list(sample_x_shape),
        "mean_bias": float(np.mean(bias_values)),
        "std_bias": float(np.std(bias_values)),
        "note": (
            "Native XGBoost feature contributions computed with pred_contribs=True. "
            "The last contribution column is the bias/base value and is stored separately."
        ),
    }

    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    print("Entered main()")
    print("PROJECT_DIR =", PROJECT_DIR)
    print("DATA_DIR =", DATA_DIR)
    print("MODEL_PATH =", MODEL_PATH)
    print("OUTPUT_DIR =", OUTPUT_DIR)

    ensure_dir(OUTPUT_DIR)
    print("OUTPUT_DIR exists now?", os.path.exists(OUTPUT_DIR))

    # -------------------------
    # Load data
    # -------------------------
    print("\nLoading data...")
    X_raw, y, meta = load_parquets(DATA_DIR)

    print("Original X shape:", X_raw.shape)
    print("y shape:", y.shape)
    print("meta shape:", meta.shape)

    X_model, dropped_cols = drop_feature_columns(X_raw, DROP_INDEX_COLS)
    print("X shape after column drops:", X_model.shape)

    if not (len(X_model) == len(y) == len(meta)):
        raise ValueError(
            f"Length mismatch: X={len(X_model)}, y={len(y)}, meta={len(meta)}"
        )

    # -------------------------
    # Load final XGB-rs04 model
    # -------------------------
    print("\nLoading final XGB-rs04 model...")
    booster, model_file = load_model(MODEL_PATH)
    print("Loaded model from:", model_file)

    # -------------------------
    # Sample rows
    # -------------------------
    print("\nSampling rows for contribution analysis...")
    X_sample, y_sample, meta_sample = sample_rows(
        X_model,
        y,
        meta,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    print("Sampled X shape:", X_sample.shape)
    print("Sampled y shape:", y_sample.shape)
    print("Sampled meta shape:", meta_sample.shape)

    # -------------------------
    # Compute contributions
    # -------------------------
    print("\nComputing XGBoost native feature contributions...")
    contrib_df, bias_values = compute_xgb_contributions(booster, X_sample)

    # -------------------------
    # Build tables
    # -------------------------
    print("\nBuilding importance tables...")
    importance_df = feature_importance_table(X_sample, contrib_df)
    grouped_df = grouped_importance_table(importance_df)
    is_wheat_df = is_wheat_summary(X_sample, contrib_df)

    wheat_df = subset_top_features(X_sample, contrib_df, meta_sample, "wheat")
    maize_df = subset_top_features(X_sample, contrib_df, meta_sample, "maize")

    # -------------------------
    # Save outputs
    # -------------------------
    print("\nSaving CSV outputs...")

    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "top_features_all.csv"),
        index=False
    )

    grouped_df.to_csv(
        os.path.join(OUTPUT_DIR, "grouped_importance.csv"),
        index=False
    )

    is_wheat_df.to_csv(
        os.path.join(OUTPUT_DIR, "is_wheat_summary.csv"),
        index=False
    )

    wheat_df.to_csv(
        os.path.join(OUTPUT_DIR, "top_features_wheat.csv"),
        index=False
    )

    maize_df.to_csv(
        os.path.join(OUTPUT_DIR, "top_features_maize.csv"),
        index=False
    )

    X_sample.to_csv(
        os.path.join(OUTPUT_DIR, "X_sample_used.csv"),
        index=False
    )

    meta_sample.to_csv(
        os.path.join(OUTPUT_DIR, "meta_sample_used.csv"),
        index=False
    )

    pd.DataFrame({"bias_value": bias_values}).to_csv(
        os.path.join(OUTPUT_DIR, "bias_values.csv"),
        index=False
    )

    save_metadata(
        output_dir=OUTPUT_DIR,
        model_file=model_file,
        dropped_columns=dropped_cols,
        full_x_shape=X_raw.shape,
        model_x_shape=X_model.shape,
        sample_x_shape=X_sample.shape,
        bias_values=bias_values
    )

    # -------------------------
    # Print summary
    # -------------------------
    print("\nDONE")
    print("Outputs saved to:", OUTPUT_DIR)

    print("\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))

    print("\nGrouped importance:")
    print(grouped_df.to_string(index=False))

    print("\nis_wheat summary:")
    print(is_wheat_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR OCCURRED:")
        print(str(e))
        print("\nFULL TRACEBACK:")
        traceback.print_exc()
        raise