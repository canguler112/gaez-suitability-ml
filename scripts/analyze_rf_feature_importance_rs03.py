# analyze_rf_feature_importance_rs03.py

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


print("SCRIPT STARTED: RF-rs03 feature importance")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(r"C:\Users\cangu\OneDrive\Desktop\Agriculture")

DATA_DIR = BASE_DIR / "data" / "processed" / "model_ready" / "wm100k_v1"
X_PATH = DATA_DIR / "X.parquet"
Y_PATH = DATA_DIR / "y.parquet"
META_PATH = DATA_DIR / "meta.parquet"

MODEL_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "analysis" / "rf_rs03_feature_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT_PATH = MODEL_DIR / "rf_rs03_fullfit.joblib"

print(f"X_PATH = {X_PATH}")
print(f"Y_PATH = {Y_PATH}")
print(f"META_PATH = {META_PATH}")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")

# -----------------------------
# Load data
# -----------------------------
if not X_PATH.exists():
    raise FileNotFoundError(f"X file not found: {X_PATH}")
if not Y_PATH.exists():
    raise FileNotFoundError(f"y file not found: {Y_PATH}")

X = pd.read_parquet(X_PATH)
y_df = pd.read_parquet(Y_PATH)

if isinstance(y_df, pd.DataFrame):
    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    elif "suitability" in y_df.columns:
        y = y_df["suitability"]
    else:
        raise ValueError(f"Could not identify target column in y.parquet: {y_df.columns.tolist()}")
else:
    y = y_df

print(f"Loaded X shape = {X.shape}")
print(f"Loaded y length = {len(y)}")

# -----------------------------
# Match modelling feature set
# -----------------------------
DROP_COLS = [
    "cell_id",
    "row",
    "col",
    "power_error",
    "power_lat",
    "power_lon",
]

existing_drop_cols = [c for c in DROP_COLS if c in X.columns]
if existing_drop_cols:
    print(f"Dropping non-modelling columns: {existing_drop_cols}")
    X_model = X.drop(columns=existing_drop_cols)
else:
    X_model = X.copy()

# Keep only numeric columns
non_numeric_cols = X_model.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"Dropping non-numeric columns: {non_numeric_cols}")
    X_model = X_model.drop(columns=non_numeric_cols)

print(f"Final modelling X shape = {X_model.shape}")

# Safety check
if X_model.isna().sum().sum() > 0:
    na_count = int(X_model.isna().sum().sum())
    raise ValueError(f"X_model still contains {na_count} missing values. Check preprocessing.")

# -----------------------------
# RF-rs03 parameters
# -----------------------------
RF_RS03_PARAMS = {
    "n_estimators": 500,
    "max_depth": 30,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

# -----------------------------
# Load existing model if possible, otherwise train full-fit model
# -----------------------------
def find_existing_rf_model(model_dir: Path):
    if not model_dir.exists():
        return None

    candidates = []
    patterns = [
        "*rf*rs03*.joblib",
        "*rf_rs03*.joblib",
        "*random*forest*rs03*.joblib",
    ]

    for pattern in patterns:
        candidates.extend(model_dir.glob(pattern))

    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


existing_model_path = find_existing_rf_model(MODEL_DIR)

model = None

if existing_model_path is not None:
    print(f"Found existing RF model candidate: {existing_model_path}")
    try:
        loaded = joblib.load(existing_model_path)

        # Handle common saved formats
        if hasattr(loaded, "feature_importances_"):
            model = loaded
        elif isinstance(loaded, dict):
            for key in ["model", "estimator", "rf", "best_model"]:
                if key in loaded and hasattr(loaded[key], "feature_importances_"):
                    model = loaded[key]
                    break

        if model is None:
            print("Existing file loaded, but no feature_importances_ found. Will train full-fit RF-rs03.")
        else:
            print("Loaded existing RF model successfully.")

    except Exception as e:
        print(f"Could not load existing RF model. Error: {e}")
        print("Will train full-fit RF-rs03.")

if model is None:
    print("Training full-fit RF-rs03 model...")
    model = RandomForestRegressor(**RF_RS03_PARAMS)
    model.fit(X_model, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT_PATH)
    print(f"Saved full-fit RF-rs03 model to: {MODEL_OUT_PATH}")

# -----------------------------
# Extract feature importance
# -----------------------------
if not hasattr(model, "feature_importances_"):
    raise AttributeError("The RF model does not expose feature_importances_.")

importances = np.asarray(model.feature_importances_, dtype=float)

if len(importances) != X_model.shape[1]:
    raise ValueError(
        f"Feature importance length ({len(importances)}) does not match "
        f"number of model columns ({X_model.shape[1]}). "
        "This usually means the loaded model was trained with a different feature set."
    )

importance_df = pd.DataFrame({
    "feature": X_model.columns,
    "importance": importances,
})

importance_sum = importance_df["importance"].sum()
if importance_sum > 0:
    importance_df["share"] = importance_df["importance"] / importance_sum
else:
    importance_df["share"] = 0.0

importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

# -----------------------------
# Feature grouping
# -----------------------------
CLIMATE_PREFIXES = (
    "PRECTOTCORR",
    "T2M",
    "T2M_MIN",
    "T2M_MAX",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
    "T2MDEW",
)

SOIL_PREFIXES = (
    "sg_",
    "clay",
    "sand",
    "silt",
    "soc",
    "phh2o",
    "bdod",
    "cfvo",
)

def assign_group(feature: str) -> str:
    if feature == "is_wheat":
        return "Crop indicator"
    if feature.startswith(CLIMATE_PREFIXES):
        return "Climate variables"
    if feature.startswith(SOIL_PREFIXES):
        return "Soil variables"
    return "Other variables"

importance_df["feature_group"] = importance_df["feature"].apply(assign_group)

group_df = (
    importance_df
    .groupby("feature_group", as_index=False)
    .agg(
        total_importance=("importance", "sum"),
        n_features=("feature", "count"),
    )
)

group_sum = group_df["total_importance"].sum()
if group_sum > 0:
    group_df["share"] = group_df["total_importance"] / group_sum
else:
    group_df["share"] = 0.0

group_df = group_df.sort_values("total_importance", ascending=False).reset_index(drop=True)

# -----------------------------
# Save outputs
# -----------------------------
importance_csv = OUTPUT_DIR / "rf_rs03_feature_importance_all_features.csv"
top_csv = OUTPUT_DIR / "rf_rs03_feature_importance_top30.csv"
group_csv = OUTPUT_DIR / "rf_rs03_feature_importance_grouped.csv"
summary_txt = OUTPUT_DIR / "rf_rs03_feature_importance_summary.txt"
summary_json = OUTPUT_DIR / "rf_rs03_feature_importance_summary.json"

importance_df.to_csv(importance_csv, index=False)
importance_df.head(30).to_csv(top_csv, index=False)
group_df.to_csv(group_csv, index=False)

summary = {
    "model": "RF-rs03",
    "importance_type": "Random Forest impurity-based feature_importances_",
    "n_rows": int(X_model.shape[0]),
    "n_features": int(X_model.shape[1]),
    "dropped_columns": existing_drop_cols + non_numeric_cols,
    "rf_params": RF_RS03_PARAMS,
    "loaded_existing_model": str(existing_model_path) if existing_model_path is not None else None,
    "trained_fullfit_model_path": str(MODEL_OUT_PATH) if MODEL_OUT_PATH.exists() else None,
    "grouped_importance": group_df.to_dict(orient="records"),
    "top_15_features": importance_df.head(15).to_dict(orient="records"),
}

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("RF-rs03 Feature Importance Summary\n")
    f.write("=================================\n\n")
    f.write("Importance type: Random Forest impurity-based feature_importances_\n")
    f.write("Note: These values are model-specific RF importances, not SHAP values.\n")
    f.write("They should not be directly compared numerically with XGBoost pred_contribs.\n\n")

    f.write(f"Rows used: {X_model.shape[0]}\n")
    f.write(f"Features used: {X_model.shape[1]}\n")
    f.write(f"Dropped columns: {existing_drop_cols + non_numeric_cols}\n\n")

    f.write("Grouped importance:\n")
    for _, row in group_df.iterrows():
        f.write(
            f"- {row['feature_group']}: "
            f"{row['total_importance']:.6f} "
            f"({row['share'] * 100:.2f}%), "
            f"n_features={int(row['n_features'])}\n"
        )

    f.write("\nTop 15 individual features:\n")
    for _, row in importance_df.head(15).iterrows():
        f.write(
            f"- {row['feature']}: "
            f"{row['importance']:.6f} "
            f"({row['share'] * 100:.2f}%), "
            f"group={row['feature_group']}\n"
        )

print("\nSaved outputs:")
print(f"- {importance_csv}")
print(f"- {top_csv}")
print(f"- {group_csv}")
print(f"- {summary_txt}")
print(f"- {summary_json}")

print("\nGrouped importance:")
print(group_df)

print("\nTop 15 features:")
print(importance_df.head(15)[["feature", "importance", "share", "feature_group"]])

print("\nSCRIPT FINISHED")