import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# =========================
# 0) LOAD
# =========================
PATH = r"C:\Users\cangu\OneDrive\Desktop\Agriculture\data\processed\wheat_maize_100k_power_soilgrids_0_30_1000m.parquet"
assert os.path.exists(PATH), f"Dosya bulunamadı: {PATH}"

try:
    df = pd.read_parquet(PATH, engine="pyarrow")
except ImportError as e:
    raise ImportError(
        "Parquet okumak için 'pyarrow' gerekli.\n"
        "Conda: conda install -c conda-forge pyarrow\n"
        "Pip:   pip install pyarrow"
    ) from e

print("Loaded df:", df.shape)

# =========================
# 1) AUTO-DETECT COLUMNS
# =========================
def find_col(candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

crop_col = find_col(["crop", "Crop", "CROP"])
target_col = find_col(["suitability", "suit_index", "gaez_suitability", "suit", "target"])

# If not found, try heuristics
if crop_col is None:
    for c in df.columns:
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
            uniq = set(map(str.lower, df[c].dropna().unique()[:10]))
            if "wheat" in uniq or "maize" in uniq:
                crop_col = c
                break

if target_col is None:
    # choose numeric col that looks like 0-10000-ish
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    best = None
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        q01, q99 = np.quantile(s, [0.01, 0.99])
        if 0 <= q01 and q99 <= 10000:
            # prefer columns with 'suit' in name
            score = (("suit" in c.lower()) * 10) + (("gaez" in c.lower()) * 5)
            score += 1
            if best is None or score > best[0]:
                best = (score, c)
    target_col = best[1] if best else None

lat_col = find_col(["lat", "latitude", "y"])
lon_col = find_col(["lon", "longitude", "x"])

if crop_col is None or target_col is None:
    raise ValueError(
        f"Kolon tespiti başarısız. crop_col={crop_col}, target_col={target_col}\n"
        f"Kolonlar: {list(df.columns)[:30]} ... (toplam {len(df.columns)})"
    )

print("Detected columns:")
print("  crop_col  =", crop_col)
print("  target_col=", target_col)
print("  lat_col   =", lat_col)
print("  lon_col   =", lon_col)

# Ensure crop values are lowercase strings for grouping
df[crop_col] = df[crop_col].astype(str).str.lower()

# =========================
# 2) SANITY CHECKS (TABLE OUTPUT)
# =========================
print("\n=== BASIC CHECKS ===")
print("Rows, Cols:", df.shape)
print("\nCrop counts:")
print(df[crop_col].value_counts(dropna=False))

print("\nTarget summary:")
print(df[target_col].describe())

print("\nTop-10 missingness columns (%):")
missing_pct = df.isna().mean().sort_values(ascending=False) * 100
print(missing_pct.head(10).round(2))

# =========================
# 3) TARGET DISTRIBUTION
# =========================
plt.figure()
df[target_col].hist(bins=60)
plt.title("Target distribution (all crops)")
plt.xlabel(target_col)
plt.ylabel("Count")
plt.show()

# By crop: histogram overlay (separate plots for clarity)
for c in sorted(df[crop_col].unique()):
    sub = df.loc[df[crop_col] == c, target_col].dropna()
    if len(sub) == 0:
        continue
    plt.figure()
    sub.hist(bins=60)
    plt.title(f"Target distribution – {c}")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.show()

# Boxplot wheat vs maize (if both exist)
crops_present = sorted(df[crop_col].unique())
if len(crops_present) >= 2:
    plt.figure()
    data = [df.loc[df[crop_col] == c, target_col].dropna().values for c in crops_present]
    plt.boxplot(data, labels=crops_present, showfliers=False)
    plt.title("Target by crop (boxplot)")
    plt.ylabel(target_col)
    plt.show()

# =========================
# 4) CLIMATE FEATURES (ANN quick look)
# =========================
# pick ANN columns if they exist
ann_cols = [c for c in df.columns if c.upper().endswith("_ANN") or c.upper().endswith("ANN")]
key_ann_candidates = [
    "T2M_ANN",
    "PRECTOTCORR_ANN",
    "ALLSKY_SFC_SW_DWN_ANN",
    "RH2M_ANN",
    "WS2M_ANN",
    "T2MDEW_ANN"
]
key_ann = [c for c in key_ann_candidates if c in df.columns]

print("\n=== ANN climate columns found ===")
print(key_ann if key_ann else "(none of the key ANN columns found)")

# Correlations with target (only numeric)
if key_ann:
    corr = df[key_ann + [target_col]].corr(numeric_only=True)[target_col].sort_values(ascending=False)
    print("\nCorr(target) with key ANN vars:")
    print(corr)

# Hexbin plots (nice, clean) for a few key ones
def hexbin_plot(xcol, ycol, title):
    x = df[xcol].values
    y = df[ycol].values
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return
    plt.figure()
    plt.hexbin(x[m], y[m], gridsize=60)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.colorbar(label="count")
    plt.show()

for xcol in key_ann[:3]:
    hexbin_plot(xcol, target_col, f"{target_col} vs {xcol}")

# =========================
# 5) SOIL FEATURES QUICK LOOK
# =========================
soil_candidates = ["clay", "sand", "silt", "soc", "phh2o", "bdod", "cfvo"]
soil_cols = [c for c in soil_candidates if c in df.columns]

print("\n=== Soil columns found ===")
print(soil_cols if soil_cols else "(none of the standard soil columns found)")

if soil_cols:
    soil_corr = df[soil_cols + [target_col]].corr(numeric_only=True)[target_col].sort_values(ascending=False)
    print("\nCorr(target) with soil vars:")
    print(soil_corr)

# Show a couple scatter/hexbin for soil (2-3 only, simple)
for xcol in soil_cols[:3]:
    hexbin_plot(xcol, target_col, f"{target_col} vs {xcol}")

# =========================
# 6) FEATURE REDUNDANCY (example: T2M vs MIN/MAX if exist)
# =========================
redundancy_triplet = [c for c in ["T2M_ANN", "T2M_MIN_ANN", "T2M_MAX_ANN"] if c in df.columns]
if len(redundancy_triplet) >= 2:
    print("\n=== Redundancy check (ANN temperature family) ===")
    print(df[redundancy_triplet].corr(numeric_only=True))

# =========================
# 7) OPTIONAL: SPATIAL QUICK CHECK (if lat/lon exist)
# =========================
if lat_col and lon_col:
    x = df[lon_col].values
    y = df[lat_col].values
    z = df[target_col].values
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    plt.figure()
    sc = plt.scatter(x[m], y[m], c=z[m], s=1)
    plt.xlabel(lon_col)
    plt.ylabel(lat_col)
    plt.title("Spatial distribution of target (quick check)")
    plt.colorbar(sc, label=target_col)
    plt.show()

print("\n✅ EDA done. If you want, I can add: (1) outlier report, (2) per-crop correlation tables, (3) export figures to a folder.")
