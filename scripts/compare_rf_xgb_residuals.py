import math
from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================================================
# FIXED PROJECT PATHS
# =========================================================
PROJECT_DIR = Path(r"C:\Users\cangu\OneDrive\Desktop\Agriculture")
ANALYSIS_DIR = PROJECT_DIR / r"outputs\analysis"

RF_PRED_CSV = ANALYSIS_DIR / "rf_rsbest_block2p0_cv_predictions.csv"
XGB_PRED_CSV = ANALYSIS_DIR / "xgb_rs04_block2p0_cv_predictions.csv"

OUT_DIR = ANALYSIS_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================
N_BINS = 5
PLOT_SAMPLE_N_PER_MODEL = 20000
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(df):
    return {
        "n": int(len(df)),
        "r2": float(r2_score(df["observed"], df["predicted"])),
        "rmse": float(rmse(df["observed"], df["predicted"])),
        "mae": float(mean_absolute_error(df["observed"], df["predicted"])),
        "mean_observed": float(df["observed"].mean()),
        "mean_predicted": float(df["predicted"].mean()),
        "mean_residual": float(df["residual"].mean()),
        "residual_std": float(df["residual"].std()),
        "mean_abs_error": float(df["abs_error"].mean()),
    }


def load_predictions(path, model_name):
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {
        "fold",
        "row_index",
        "crop",
        "lat",
        "lon",
        "observed",
        "predicted",
        "residual",
        "abs_error",
        "squared_error",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    df["model"] = model_name
    return df


def save_overall_metrics(combined):
    rows = []

    for model_name, sub in combined.groupby("model"):
        row = {"model": model_name}
        row.update(compute_metrics(sub))
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    out_path = OUT_DIR / "overall_metrics_comparison.csv"
    out_df.to_csv(out_path, index=False)

    print("\n=== Overall metrics comparison ===")
    print(out_df.to_string(index=False))
    print("Saved:", out_path)


def save_crop_metrics(combined):
    rows = []

    for (model_name, crop_name), sub in combined.groupby(["model", "crop"]):
        row = {
            "model": model_name,
            "crop": crop_name,
        }
        row.update(compute_metrics(sub))
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["crop", "r2"], ascending=[True, False])
    out_path = OUT_DIR / "crop_metrics_comparison.csv"
    out_df.to_csv(out_path, index=False)

    print("\n=== Crop-level metrics comparison ===")
    print(out_df.to_string(index=False))
    print("Saved:", out_path)


def save_error_by_observed_bin(combined):
    """
    Important:
    Bins are created on the combined observed values, so RF and XGB are compared
    on identical suitability ranges.
    """
    combined = combined.copy()
    combined["observed_bin"] = pd.qcut(
        combined["observed"],
        q=N_BINS,
        duplicates="drop"
    )

    out_df = (
        combined
        .groupby(["model", "observed_bin"], observed=False)
        .agg(
            n=("observed", "size"),
            observed_min=("observed", "min"),
            observed_max=("observed", "max"),
            observed_mean=("observed", "mean"),
            predicted_mean=("predicted", "mean"),
            residual_mean=("residual", "mean"),
            residual_std=("residual", "std"),
            abs_error_mean=("abs_error", "mean"),
            squared_error_mean=("squared_error", "mean"),
        )
        .reset_index()
    )

    out_df["observed_bin"] = out_df["observed_bin"].astype(str)

    out_path = OUT_DIR / "error_by_observed_bin_comparison.csv"
    out_df.to_csv(out_path, index=False)

    print("\n=== Error by observed bin comparison ===")
    print(out_df.to_string(index=False))
    print("Saved:", out_path)


def make_plot_sample(combined):
    parts = []

    for model_name, sub in combined.groupby("model"):
        if len(sub) > PLOT_SAMPLE_N_PER_MODEL:
            parts.append(
                sub.sample(
                    n=PLOT_SAMPLE_N_PER_MODEL,
                    random_state=RANDOM_STATE
                )
            )
        else:
            parts.append(sub.copy())

    return pd.concat(parts, ignore_index=True)


def save_observed_vs_predicted_plot(plot_df):
    fig = px.scatter(
        plot_df,
        x="observed",
        y="predicted",
        color="model",
        facet_col="model",
        opacity=0.25,
        title="Observed vs Predicted: RF-rsbest vs XGB-rs04 (2.0° Spatial CV)",
    )

    min_v = min(plot_df["observed"].min(), plot_df["predicted"].min())
    max_v = max(plot_df["observed"].max(), plot_df["predicted"].max())

    fig.add_shape(
        type="line",
        x0=min_v,
        y0=min_v,
        x1=max_v,
        y1=max_v,
        line=dict(dash="dash"),
        row="all",
        col="all",
    )

    out_path = OUT_DIR / "observed_vs_predicted_rf_vs_xgb.html"
    fig.write_html(str(out_path))
    print("Saved:", out_path)


def save_residual_vs_observed_plot(plot_df):
    fig = px.scatter(
        plot_df,
        x="observed",
        y="residual",
        color="model",
        facet_col="model",
        opacity=0.25,
        title="Residuals vs Observed: RF-rsbest vs XGB-rs04 (2.0° Spatial CV)",
    )

    fig.add_hline(y=0, line_dash="dash")

    out_path = OUT_DIR / "residual_vs_observed_rf_vs_xgb.html"
    fig.write_html(str(out_path))
    print("Saved:", out_path)


def save_residual_histogram(combined):
    fig = px.histogram(
        combined,
        x="residual",
        color="model",
        nbins=70,
        barmode="overlay",
        opacity=0.55,
        title="Residual Distribution: RF-rsbest vs XGB-rs04 (2.0° Spatial CV)",
    )

    out_path = OUT_DIR / "residual_histogram_rf_vs_xgb.html"
    fig.write_html(str(out_path))
    print("Saved:", out_path)


def save_abs_error_by_bin_plot(combined):
    temp = combined.copy()
    temp["observed_bin"] = pd.qcut(
        temp["observed"],
        q=N_BINS,
        duplicates="drop"
    )

    bin_df = (
        temp
        .groupby(["model", "observed_bin"], observed=False)
        .agg(
            observed_mean=("observed", "mean"),
            abs_error_mean=("abs_error", "mean"),
            residual_mean=("residual", "mean"),
        )
        .reset_index()
    )

    bin_df["observed_bin"] = bin_df["observed_bin"].astype(str)

    fig = px.line(
        bin_df,
        x="observed_mean",
        y="abs_error_mean",
        color="model",
        markers=True,
        title="Mean Absolute Error by Observed Suitability Bin",
    )

    out_path = OUT_DIR / "abs_error_by_observed_bin_rf_vs_xgb.html"
    fig.write_html(str(out_path))
    print("Saved:", out_path)


def save_residual_mean_by_bin_plot(combined):
    temp = combined.copy()
    temp["observed_bin"] = pd.qcut(
        temp["observed"],
        q=N_BINS,
        duplicates="drop"
    )

    bin_df = (
        temp
        .groupby(["model", "observed_bin"], observed=False)
        .agg(
            observed_mean=("observed", "mean"),
            residual_mean=("residual", "mean"),
        )
        .reset_index()
    )

    bin_df["observed_bin"] = bin_df["observed_bin"].astype(str)

    fig = px.line(
        bin_df,
        x="observed_mean",
        y="residual_mean",
        color="model",
        markers=True,
        title="Mean Residual by Observed Suitability Bin",
    )

    fig.add_hline(y=0, line_dash="dash")

    out_path = OUT_DIR / "mean_residual_by_observed_bin_rf_vs_xgb.html"
    fig.write_html(str(out_path))
    print("Saved:", out_path)


def save_interpretation_notes():
    out_path = OUT_DIR / "interpretation_notes.txt"

    text = """RF-rsbest vs XGB-rs04 Residual Comparison Notes
================================================

Residual definition:
observed suitability - predicted suitability

Interpretation:
- Negative residual = overprediction
- Positive residual = underprediction

What to check:
1. Overall metrics:
   - Compare R², RMSE, and MAE.
   - XGB-rs04 is expected to outperform RF-rsbest.

2. Residual distribution:
   - Check whether XGB has a narrower residual distribution than RF.
   - A narrower distribution suggests less residual dispersion.

3. Residuals by observed suitability:
   - Check whether both models overpredict low suitability values.
   - Check whether both models underpredict high suitability values.
   - This indicates regression-to-the-mean.

4. Error by observed bin:
   - Compare mean absolute error across suitability ranges.
   - If errors are largest in the lowest and highest bins, this supports the interpretation that extreme suitability values are harder to approximate.

5. Crop-level comparison:
   - Check whether XGB improves performance for both wheat and maize.
   - If maize remains harder for both models, discuss crop-level heterogeneity.

Thesis-ready interpretation template:
Both models showed a regression-to-the-mean error pattern under 2.0° spatial block cross-validation. Low observed suitability values tended to be overpredicted, whereas high observed suitability values tended to be underpredicted. This suggests that extreme suitability conditions were harder to approximate than mid-range suitability values. XGB-rs04 should be interpreted as the stronger model if it reduces RMSE, MAE, residual spread, and bin-level errors relative to RF-rsbest.
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("Saved:", out_path)


def main():
    print("SCRIPT STARTED")
    print("RF_PRED_CSV =", RF_PRED_CSV)
    print("XGB_PRED_CSV =", XGB_PRED_CSV)
    print("OUT_DIR =", OUT_DIR)

    rf = load_predictions(RF_PRED_CSV, "RF-rsbest")
    xgb = load_predictions(XGB_PRED_CSV, "XGB-rs04")

    print("Loaded RF rows:", len(rf))
    print("Loaded XGB rows:", len(xgb))

    combined = pd.concat([rf, xgb], ignore_index=True)

    save_overall_metrics(combined)
    save_crop_metrics(combined)
    save_error_by_observed_bin(combined)

    plot_df = make_plot_sample(combined)
    print("Plot sample rows:", len(plot_df))

    save_observed_vs_predicted_plot(plot_df)
    save_residual_vs_observed_plot(plot_df)
    save_residual_histogram(combined)
    save_abs_error_by_bin_plot(combined)
    save_residual_mean_by_bin_plot(combined)
    save_interpretation_notes()

    print("\nDONE")
    print("All comparison outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()