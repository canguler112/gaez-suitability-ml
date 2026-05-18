# ============================================================
# Build 50k stratified point samples from GAEZ suitability GeoTIFFs
# - Wheat: suLr_whe.tif
# - Maize: suLr_mze.tif
#
# Outputs:
#   data/interim/points/wheat_points_50k.parquet (+ .csv)
#   data/interim/points/maize_points_50k.parquet (+ .csv)
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ----------------------------
# User settings
# ----------------------------
INPUT_DIR = "data/raw/gaez"
OUT_DIR = "data/interim/points"

FILES = {
    "wheat": "suLr_whe.tif",
    "maize": "suLr_mze.tif",
}

N_TOTAL = 50_000
N_ZEROS = 10_000          # 0 suitability samples
N_NONZERO = N_TOTAL - N_ZEROS
Q_BINS = 8                # number of quantile bins for >0 samples
SEED = 42

# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)

def raster_to_df(tif_path: str) -> pd.DataFrame:
    """
    Read a single-band raster and return a DataFrame with row/col, lat/lon, suitability.
    Filters out NoData.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform

    # Handle nodata (sometimes nodata is None)
    if nodata is None:
        # common for some rasters: treat NaN as nodata if present
        mask_valid = ~np.isnan(arr)
    else:
        mask_valid = arr != nodata

    rows, cols = np.where(mask_valid)
    vals = arr[rows, cols].astype(np.int32, copy=False)

    # Convert row/col to lon/lat (cell center)
    lons, lats = xy(transform, rows, cols, offset="center")

    df = pd.DataFrame({
        "row": rows.astype(np.int32),
        "col": cols.astype(np.int32),
        "lat": np.array(lats, dtype=np.float64),
        "lon": np.array(lons, dtype=np.float64),
        "suitability": vals
    })
    return df

def stratified_sample(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """
    Build a 50k stratified sample:
      - N_ZEROS from suitability == 0
      - N_NONZERO from suitability > 0, equally across quantile bins
    """
    # Basic cleaning
    df = df.dropna(subset=["suitability", "lat", "lon"]).copy()

    zeros = df[df["suitability"] == 0]
    nonzero = df[df["suitability"] > 0]

    if len(zeros) < N_ZEROS:
        raise ValueError(
            f"[{crop}] Not enough zero cells to sample {N_ZEROS}. Found {len(zeros)}."
        )
    if len(nonzero) < N_NONZERO:
        raise ValueError(
            f"[{crop}] Not enough nonzero cells to sample {N_NONZERO}. Found {len(nonzero)}."
        )

    zeros_sample = zeros.sample(n=N_ZEROS, random_state=SEED)

    # Quantile bins on nonzero suitability
    # duplicates="drop" handles cases where many values repeat
    nonzero = nonzero.copy()
    nonzero["bin"] = pd.qcut(nonzero["suitability"], q=Q_BINS, duplicates="drop")

    n_bins = nonzero["bin"].nunique()
    if n_bins < 2:
        raise ValueError(f"[{crop}] Too few unique bins after qcut. Try reducing Q_BINS.")

    per_bin = N_NONZERO // n_bins
    remainder = N_NONZERO - (per_bin * n_bins)

    # Sample per bin; distribute remainder by giving +1 to first bins
    sampled_parts = []
    for i, (b, g) in enumerate(nonzero.groupby("bin")):
        take = per_bin + (1 if i < remainder else 0)
        take = min(take, len(g))  # safety
        sampled_parts.append(g.sample(n=take, random_state=SEED))

    nonzero_sample = pd.concat(sampled_parts, ignore_index=True)

    out = pd.concat([zeros_sample, nonzero_sample], ignore_index=True)
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle

    # Create stable cell_id (row/col based)
    out["cell_id"] = crop + "_" + out["row"].astype(str) + "_" + out["col"].astype(str)
    out["crop"] = crop

    # Keep clean columns
    out = out[["cell_id", "crop", "lat", "lon", "row", "col", "suitability"]]

    if len(out) != N_TOTAL:
        raise RuntimeError(f"[{crop}] Sample size mismatch: {len(out)} != {N_TOTAL}")

    return out

def save_outputs(df_sample: pd.DataFrame, crop: str):
    parquet_path = os.path.join(OUT_DIR, f"{crop}_points_50k.parquet")
    csv_path = os.path.join(OUT_DIR, f"{crop}_points_50k.csv")

    df_sample.to_parquet(parquet_path, index=False)
    df_sample.to_csv(csv_path, index=False)

    print(f"[{crop}] Saved:")
    print(f"  - {parquet_path}")
    print(f"  - {csv_path}")

def quick_checks(df_sample: pd.DataFrame, crop: str):
    print("\n" + "=" * 60)
    print(f"[{crop}] QUICK CHECKS")
    print("=" * 60)
    print("Rows:", len(df_sample))
    zero_rate = (df_sample["suitability"] == 0).mean()
    print(f"Zero rate: {zero_rate:.3f} (expected ~{N_ZEROS/N_TOTAL:.3f})")
    print("Suitability min/max:", df_sample["suitability"].min(), df_sample["suitability"].max())
    print(df_sample["suitability"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dirs()

    for crop, fname in FILES.items():
        tif_path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"Missing file: {tif_path}")

        print(f"\nProcessing {crop}: {tif_path}")
        df_full = raster_to_df(tif_path)
        print(f"[{crop}] Valid cells in raster: {len(df_full):,}")

        df_sample = stratified_sample(df_full, crop=crop)
        save_outputs(df_sample, crop=crop)
        quick_checks(df_sample, crop=crop)

    print("\nDONE ✅ Next step: use these point files to pull NASA POWER + SoilGrids features.")

if __name__ == "__main__":
    main()
