# EDA_final_debugsave.py
# Purpose: Save EDA figures and diagnostics for thesis.
# Run:
#   conda activate gaez


import os
import traceback
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # save-only backend
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# SETTINGS
# =========================

PATH = r"C:\Users\cangu\OneDrive\Desktop\Agriculture\data\processed\wheat_maize_100k_power_soilgrids_0_30_1000m.parquet"

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "eda_figs")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# STYLE
# =========================

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


def log(msg: str):
    print(msg, flush=True)


def write_error_file(e: Exception):
    err_path = os.path.join(OUT_DIR, "eda_error.txt")
    with open(err_path, "w", encoding="utf-8") as f:
        f.write("EDA crashed with exception:\n\n")
        f.write("".join(traceback.format_exc()))
    log(f"[saved] {err_path}")


def save_fig(fname: str):
    """
    Save figure as PNG.
    If fname ends with .png, also save a matching PDF version.
    """
    out_path = os.path.join(OUT_DIR, fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    if fname.lower().endswith(".png"):
        pdf_path = out_path[:-4] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        log(f"[saved] {pdf_path}")

    plt.close()
    log(f"[saved] {out_path}")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =========================
# MAIN
# =========================

try:
    log(">>> RUNNING EDA_FINAL_DEBUGSAVE <<<")
    log(f"CWD    = {os.getcwd()}")
    log(f"OUTDIR = {OUT_DIR}")

    if not os.path.exists(PATH):
        raise FileNotFoundError(f"Dataset not found: {PATH}")

    df = pd.read_parquet(PATH, engine="pyarrow")
    log(f"Loaded df: {df.shape}")

    crop_col, target_col, lat_col, lon_col = "crop", "suitability", "lat", "lon"

    for c in [crop_col, target_col, lat_col, lon_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df[crop_col] = df[crop_col].astype(str).str.lower()
    df[target_col] = safe_numeric(df[target_col])
    df[lat_col] = safe_numeric(df[lat_col])
    df[lon_col] = safe_numeric(df[lon_col])

    # -------- BASIC CHECKS
    log("\n=== BASIC CHECKS ===")
    log(f"Rows, Cols: {df.shape}")

    log("\nCrop counts:")
    log(df[crop_col].value_counts(dropna=False).to_string())

    log("\nTarget summary:")
    log(df[target_col].describe().to_string())

    missing_pct = (df.isna().mean().sort_values(ascending=False) * 100).round(2)

    log("\nTop-10 missingness columns (%):")
    log(missing_pct.head(10).to_string())

    summary_path = os.path.join(OUT_DIR, "eda_summary_basic.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Crop counts:\n")
        f.write(df[crop_col].value_counts(dropna=False).to_string())
        f.write("\n\nTarget summary:\n")
        f.write(df[target_col].describe().to_string())
        f.write("\n\nTop-10 missingness (%):\n")
        f.write(missing_pct.head(10).to_string())

    log(f"[saved] {summary_path}")

    # -------- PLOTS
    log("\n=== PLOTS: TARGET DISTRIBUTIONS ===")
    plt.close("all")

    # 1) All-crops histogram
    log("Checkpoint: creating histogram figure (all crops)...")
    fig, ax = plt.subplots(figsize=(6, 4))
    series_all = df[target_col].dropna()
    log(f"Checkpoint: series length = {len(series_all)}")

    ax.hist(series_all, bins=60)
    ax.set_xlabel("GAEZ suitability score")
    ax.set_ylabel("Count")
    clean_axes(ax)

    log("Checkpoint: saving 01_target_all.png ...")
    save_fig("01_target_all.png")

    # 2) Per-crop histograms
    log("Checkpoint: per-crop histograms...")

    for c in sorted(df[crop_col].dropna().unique()):
        sub = df.loc[df[crop_col] == c, target_col].dropna()
        log(f"  crop={c}, n={len(sub)}")

        if len(sub) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sub, bins=60)
        ax.set_xlabel("GAEZ suitability score")
        ax.set_ylabel("Count")
        clean_axes(ax)

        save_fig(f"02_target_{c}.png")

    # 3) Boxplot by crop 
    crops_present = sorted(df[crop_col].dropna().unique())

    if len(crops_present) >= 2:
        log("Checkpoint: boxplot by crop...")

        data = [
            df.loc[df[crop_col] == c, target_col].dropna().values
            for c in crops_present
        ]

        labels = [
            f"{c.capitalize()} (n = {len(df.loc[df[crop_col] == c, target_col].dropna()):,})"
            for c in crops_present
        ]

        fig, ax = plt.subplots(figsize=(5, 4))

        box = ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            widths=0.5,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )

        colors = ["#4C72B0", "#C44E52"]

        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        ax.set_ylabel("GAEZ suitability score")
        ax.set_ylim(0, 10500)
        clean_axes(ax)

        save_fig("03_boxplot_by_crop.png")

    # 4) Correlations
    log("\n=== CORRELATIONS ===")

    ann_cols = [c for c in df.columns if c.upper().endswith("_ANN")]
    log(f"ANN cols: {len(ann_cols)}")

    if ann_cols:
        ann_corr = (
            df[ann_cols + [target_col]]
            .corr(numeric_only=True)[target_col]
            .sort_values(ascending=False)
        )

        ann_path = os.path.join(OUT_DIR, "corr_ann_climate.csv")
        ann_corr.to_csv(ann_path, header=["corr_with_target"])
        log(f"[saved] {ann_path}")

    soil_cols = [
        c for c in df.columns
        if c.startswith("sg_") and c.endswith("_0_30")
    ]

    log(f"Soil cols: {len(soil_cols)}")

    if soil_cols:
        soil_corr = (
            df[soil_cols + [target_col]]
            .corr(numeric_only=True)[target_col]
            .sort_values(ascending=False)
        )

        soil_path = os.path.join(OUT_DIR, "corr_soilgrids.csv")
        soil_corr.to_csv(soil_path, header=["corr_with_target"])
        log(f"[saved] {soil_path}")

    # 5) Spatial target distribution
    log("\n=== SPATIAL TARGET DISTRIBUTION ===")

    x = df[lon_col].to_numpy()
    y = df[lat_col].to_numpy()
    z = df[target_col].to_numpy()

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    log(f"Spatial finite count: {m.sum()}")

    if m.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))

        sc = ax.scatter(
            x[m],
            y[m],
            c=z[m],
            s=1,
            alpha=0.65,
            cmap="viridis",
            vmin=0,
            vmax=10000,
            edgecolors="none",
        )

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 75)

        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax.set_yticks([-60, -30, 0, 30, 60])

        clean_axes(ax)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("GAEZ suitability score (0–10,000)")

        save_fig("99_spatial_target.png")

    log("\n✅ EDA finished successfully.")
    log(f"✅ Open folder: {OUT_DIR}")

except Exception as e:
    write_error_file(e)
    raise