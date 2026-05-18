"""
generate_clean_appendix_b_figures.py
------------------------------------
Creates clean versions of Appendix B Figure 5 and Figure 6.

Figure 5:
    Residual distribution for RF-rs03 and XGB-rs04.

Figure 6:
    Mean absolute error by observed suitability bin for RF-rs03 and XGB-rs04.

Inputs:
    outputs/analysis/xgb_rs04_block2p0_cv_predictions.csv
    outputs/analysis/rf_rsbest_block2p0_cv_predictions.csv

Outputs:
    outputs/analysis/appendix_b_figures_clean/Figure5_residual_distribution_rf_rs03_xgb_rs04.pdf
    outputs/analysis/appendix_b_figures_clean/Figure5_residual_distribution_rf_rs03_xgb_rs04.png
    outputs/analysis/appendix_b_figures_clean/Figure6_mae_by_observed_bin_rf_rs03_xgb_rs04.pdf
    outputs/analysis/appendix_b_figures_clean/Figure6_mae_by_observed_bin_rf_rs03_xgb_rs04.png
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # save-only backend

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\Users\cangu\OneDrive\Desktop\Agriculture")

XGB_CSV = PROJECT_DIR / "outputs" / "analysis" / "xgb_rs04_block2p0_cv_predictions.csv"
RF_CSV = PROJECT_DIR / "outputs" / "analysis" / "rf_rsbest_block2p0_cv_predictions.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "analysis" / "appendix_b_figures_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STYLE
# ============================================================

RF_COLOR = "#4C72B0"
XGB_COLOR = "#C44E52"

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def clean_axes(ax):
    """Apply thesis-style axis formatting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)


def validate_input(df, name):
    required = ["observed", "predicted", "residual", "abs_error"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=required).copy()
    after = len(df)

    if before != after:
        print(f"{name}: dropped {before - after:,} rows with missing numeric values.")

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("Loading prediction files")
print("=" * 80)

if not XGB_CSV.exists():
    raise FileNotFoundError(f"XGB prediction file not found: {XGB_CSV}")

if not RF_CSV.exists():
    raise FileNotFoundError(f"RF prediction file not found: {RF_CSV}")

xgb = pd.read_csv(XGB_CSV)
rf = pd.read_csv(RF_CSV)

xgb = validate_input(xgb, "XGB-rs04")
rf = validate_input(rf, "RF-rs03")

print(f"XGB-rs04 rows: {len(xgb):,}")
print(f"RF-rs03 rows:  {len(rf):,}")


# ============================================================
# FIGURE 5 — RESIDUAL DISTRIBUTION
# ============================================================

print("\nCreating Figure 5: residual distribution...")

fig, ax = plt.subplots(figsize=(6, 4))

# Fixed range to reduce influence of extreme tails in display
resid_min = -6000
resid_max = 6000
bins = np.linspace(resid_min, resid_max, 41)

ax.hist(
    rf["residual"],
    bins=bins,
    color=RF_COLOR,
    alpha=0.55,
    label="RF-rs03",
    edgecolor="none"
)

ax.hist(
    xgb["residual"],
    bins=bins,
    color=XGB_COLOR,
    alpha=0.55,
    label="XGB-rs04",
    edgecolor="none"
)

ax.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=0.8,
    alpha=0.7
)

ax.set_xlabel(r"Residual (observed $-$ predicted suitability)")
ax.set_ylabel("Count")
ax.set_xlim(resid_min, resid_max)

# Force ordinary numeric y-axis labels rather than 1e4 offset notation
formatter = ScalarFormatter(useMathText=False)
formatter.set_scientific(False)
formatter.set_useOffset(False)
ax.yaxis.set_major_formatter(formatter)

ax.legend(loc="upper right", frameon=False)
clean_axes(ax)

fig5_pdf = OUTPUT_DIR / "Figure5_residual_distribution_rf_rs03_xgb_rs04.pdf"
fig5_png = OUTPUT_DIR / "Figure5_residual_distribution_rf_rs03_xgb_rs04.png"

fig.savefig(fig5_pdf)
fig.savefig(fig5_png)
plt.close(fig)

print(f"Saved: {fig5_pdf}")
print(f"Saved: {fig5_png}")


# ============================================================
# FIGURE 6 — MAE BY OBSERVED SUITABILITY BIN
# ============================================================

print("\nCreating Figure 6: mean absolute error by observed suitability bin...")

labels = ["Lowest", "Low-mid", "Middle", "High-mid", "Highest"]

# Use RF observed quintile edges, matching the thesis Table 7 logic
_, bin_edges = pd.qcut(
    rf["observed"],
    q=5,
    labels=labels,
    duplicates="drop",
    retbins=True
)

rf_plot = rf.copy()
xgb_plot = xgb.copy()

rf_plot["bin"] = pd.cut(
    rf_plot["observed"],
    bins=bin_edges,
    labels=labels,
    include_lowest=True
)

xgb_plot["bin"] = pd.cut(
    xgb_plot["observed"],
    bins=bin_edges,
    labels=labels,
    include_lowest=True
)

rf_bin = (
    rf_plot
    .groupby("bin", observed=True)
    .agg(mae=("abs_error", "mean"))
    .reindex(labels)
)

xgb_bin = (
    xgb_plot
    .groupby("bin", observed=True)
    .agg(mae=("abs_error", "mean"))
    .reindex(labels)
)

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(
    x,
    rf_bin["mae"],
    color=RF_COLOR,
    marker="o",
    linewidth=1.5,
    markersize=5,
    label="RF-rs03"
)

ax.plot(
    x,
    xgb_bin["mae"],
    color=XGB_COLOR,
    marker="s",
    linewidth=1.5,
    markersize=5,
    label="XGB-rs04"
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_xlabel("Observed suitability bin")
ax.set_ylabel("Mean absolute error")

# Gives a little breathing room without exaggerating
y_min = max(0, min(rf_bin["mae"].min(), xgb_bin["mae"].min()) - 150)
y_max = max(rf_bin["mae"].max(), xgb_bin["mae"].max()) + 150
ax.set_ylim(y_min, y_max)

ax.legend(loc="upper left", frameon=False)
clean_axes(ax)

fig6_pdf = OUTPUT_DIR / "Figure6_mae_by_observed_bin_rf_rs03_xgb_rs04.pdf"
fig6_png = OUTPUT_DIR / "Figure6_mae_by_observed_bin_rf_rs03_xgb_rs04.png"

fig.savefig(fig6_pdf)
fig.savefig(fig6_png)
plt.close(fig)

print(f"Saved: {fig6_pdf}")
print(f"Saved: {fig6_png}")


# ============================================================
# PRINT TABLE FOR CHECKING
# ============================================================

print("\nFigure 6 data check:")
check_table = pd.DataFrame({
    "bin": labels,
    "RF-rs03_MAE": rf_bin["mae"].values,
    "XGB-rs04_MAE": xgb_bin["mae"].values,
}).round(2)

print(check_table.to_string(index=False))

print("\nDONE")
print(f"All clean Appendix B figures written to:\n{OUTPUT_DIR}")