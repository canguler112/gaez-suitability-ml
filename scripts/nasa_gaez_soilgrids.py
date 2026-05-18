import os
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import requests
import rasterio
from pyproj import Transformer


# -----------------------------
# INPUT / OUTPUT
# -----------------------------
IN_PARQUET  = "data/processed/wheat_maize_100k_power_1981_2010.parquet"
OUT_PARQUET = "data/processed/wheat_maize_100k_power_soilgrids_0_30_1000m.parquet"

SOIL_DIR = Path("data/raw/soilgrids_1000m")  # where GeoTIFFs will live
SOIL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# SoilGrids selection
# -----------------------------
PROPS  = ["clay", "sand", "silt", "soc", "phh2o", "bdod", "cfvo"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm"]  # needed for 0–30 cm
WEIGHTS = {"0-5cm": 5/30, "5-15cm": 10/30, "15-30cm": 15/30}

BASE_URL = "https://files.isric.org/soilgrids/latest/data_aggregated/1000m"

# SoilGrids integer scaling → conventional units (divide by factor)
# From SoilGrids docs table.  :contentReference[oaicite:7]{index=7}
DIV_FACTOR = {
    "clay": 10,    # g/kg -> % (g/100g)
    "sand": 10,
    "silt": 10,
    "soc":  10,    # dg/kg -> g/kg
    "phh2o": 10,   # pH*10 -> pH
    "bdod": 100,   # cg/cm3 -> kg/dm3
    "cfvo": 100,   # cm3/dm3 -> vol%
}

# -----------------------------
# Download helper (optional)
# -----------------------------
def url_for(prop: str, depth: str) -> str:
    # Pattern verified on ISRIC WebDAV index pages. :contentReference[oaicite:8]{index=8}
    return f"{BASE_URL}/{prop}/{prop}_{depth}_mean_1000.tif"

def local_path(prop: str, depth: str) -> Path:
    return SOIL_DIR / f"{prop}_{depth}_mean_1000.tif"

def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return  # already downloaded

    print(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        t0 = time.time()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total and done % (50 * chunk_size) == 0:
                    mb = done / (1024**2)
                    mbt = total / (1024**2)
                    rate = mb / max(time.time() - t0, 1e-6)
                    print(f"  {out_path.name}: {mb:.0f}/{mbt:.0f} MB ({rate:.1f} MB/s)")
    print(f"Saved: {out_path}")

# -----------------------------
# Raster sampling
# -----------------------------
def sample_raster_at_points(raster_path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Samples raster at given WGS84 lat/lon points.
    Auto-transforms points into raster CRS if needed.
    Returns float array with nodata -> np.nan
    """
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        crs = src.crs

        # prepare coordinate transformer if raster isn't EPSG:4326
        if crs is not None and crs.to_string() not in ("EPSG:4326", "WGS84"):
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            xs, ys = transformer.transform(lons, lats)
        else:
            xs, ys = lons, lats

        coords = list(zip(xs, ys))

        # rasterio.sample returns an iterator of arrays (bands)
        out = np.empty(len(coords), dtype="float64")
        out.fill(np.nan)

        # progress every 10k points
        for i, val in enumerate(src.sample(coords)):
            v = val[0]
            if nodata is not None and v == nodata:
                out[i] = np.nan
            else:
                out[i] = float(v)

            if (i + 1) % 10000 == 0:
                print(f"    sampled {i+1}/{len(coords)}")

        return out

# -----------------------------
# Main pipeline
# -----------------------------
def main(download_first: bool = True) -> None:
    df = pd.read_parquet(IN_PARQUET)

    # use cell centers
    lats = df["lat"].to_numpy(dtype="float64")
    lons = df["lon"].to_numpy(dtype="float64")

    # (Optional) download rasters
    if download_first:
        for prop in PROPS:
            for depth in DEPTHS:
                u = url_for(prop, depth)
                p = local_path(prop, depth)
                download_file(u, p)

    # Collect per-property depth arrays
    # store in dict: (prop, depth) -> np.array
    values: Dict[Tuple[str, str], np.ndarray] = {}

    total_rasters = len(PROPS) * len(DEPTHS)
    raster_idx = 0

    for prop in PROPS:
        for depth in DEPTHS:
            raster_idx += 1
            rp = local_path(prop, depth)
            if not rp.exists():
                raise FileNotFoundError(f"Missing raster: {rp} (set download_first=True or download manually)")

            print(f"\n[{raster_idx}/{total_rasters}] Sampling {rp.name}")
            arr = sample_raster_at_points(rp, lats, lons)

            # convert integer-scaled values to conventional units (divide by factor)
            arr = arr / DIV_FACTOR[prop]  # :contentReference[oaicite:9]{index=9}

            values[(prop, depth)] = arr

    # Weighted average to 0–30 cm for each property
    for prop in PROPS:
        v0 = values[(prop, "0-5cm")]
        v1 = values[(prop, "5-15cm")]
        v2 = values[(prop, "15-30cm")]

        # weighted sum; if any depth is nan -> nan
        stacked = np.vstack([v0, v1, v2])
        w = np.array([WEIGHTS["0-5cm"], WEIGHTS["5-15cm"], WEIGHTS["15-30cm"]], dtype="float64")

        # compute weighted avg with nan handling: require all three depths present
        ok = np.all(~np.isnan(stacked), axis=0)
        out = np.full(stacked.shape[1], np.nan, dtype="float64")
        out[ok] = (stacked[:, ok].T @ w)

        df[f"sg_{prop}_0_30"] = out

    # Save
    Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n✅ Wrote final file: {OUT_PARQUET}")
    print("Added columns:", [f"sg_{p}_0_30" for p in PROPS])


if __name__ == "__main__":
    # set False if you already downloaded the GeoTIFFs manually
    main(download_first=True)
