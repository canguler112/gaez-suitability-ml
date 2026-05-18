from pathlib import Path
import pandas as pd
import plotly.express as px

PROJECT_DIR = Path(r"C:\Users\cangu\OneDrive\Desktop\Agriculture")
ANALYSIS_DIR = PROJECT_DIR / r"outputs\analysis"
PRED_CSV = ANALYSIS_DIR / "xgb_rs04_block2p0_cv_predictions.csv"
OUT_DIR = ANALYSIS_DIR / "xgb_rs04_block2p0_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 5
PLOT_SAMPLE_N = 20000
RANDOM_STATE = 42


def main():
    print("SCRIPT STARTED")
    print("PRED_CSV =", PRED_CSV)
    print("PRED_CSV exists? ->", PRED_CSV.exists())

    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Prediction file not found: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)
    print("Loaded rows =", len(df))

    if len(df) > PLOT_SAMPLE_N:
        plot_df = df.sample(n=PLOT_SAMPLE_N, random_state=RANDOM_STATE).copy()
        print(f"Using sample for interactive plots: {len(plot_df)} rows")
    else:
        plot_df = df.copy()
        print("Using full data for interactive plots")

    # 1) Observed vs Predicted
    fig1 = px.scatter(
        plot_df,
        x="observed",
        y="predicted",
        opacity=0.25,
        title="Observed vs Predicted (XGB-rs04, 2.0 spatial CV)"
    )
    min_v = min(plot_df["observed"].min(), plot_df["predicted"].min())
    max_v = max(plot_df["observed"].max(), plot_df["predicted"].max())
    fig1.add_shape(
        type="line",
        x0=min_v, y0=min_v,
        x1=max_v, y1=max_v,
        line=dict(dash="dash")
    )
    fig1.write_html(str(OUT_DIR / "observed_vs_predicted.html"))
    print("Saved observed_vs_predicted.html")

    # 2) Residual vs Observed
    fig2 = px.scatter(
        plot_df,
        x="observed",
        y="residual",
        opacity=0.25,
        title="Residuals vs Observed (XGB-rs04, 2.0 spatial CV)"
    )
    fig2.add_hline(y=0, line_dash="dash")
    fig2.write_html(str(OUT_DIR / "residual_vs_observed.html"))
    print("Saved residual_vs_observed.html")

    # 3) Residual histogram
    fig3 = px.histogram(
        df,
        x="residual",
        nbins=50,
        title="Residual Distribution (XGB-rs04, 2.0 spatial CV)"
    )
    fig3.write_html(str(OUT_DIR / "residual_histogram.html"))
    print("Saved residual_histogram.html")

    # 4) Error by observed bins
    df["observed_bin"] = pd.qcut(df["observed"], q=N_BINS, duplicates="drop")
    bin_summary = (
        df.groupby("observed_bin", observed=False)
        .agg(
            n=("observed", "size"),
            observed_mean=("observed", "mean"),
            predicted_mean=("predicted", "mean"),
            residual_mean=("residual", "mean"),
            abs_error_mean=("abs_error", "mean"),
            residual_std=("residual", "std"),
        )
        .reset_index()
    )
    bin_summary.to_csv(OUT_DIR / "error_by_observed_bin.csv", index=False)
    print("Saved error_by_observed_bin.csv")

    print("DONE")


if __name__ == "__main__":
    main()