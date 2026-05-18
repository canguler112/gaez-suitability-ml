#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_model_data.py

Thesis-grade data preparation for GAEZ v4 surrogate modeling.

Key decisions (project plan):
- Drop rows where NASA POWER retrieval failed (identified via non-null `power_error`)
- Drop `power_error` column afterwards
- Drop auxiliary POWER matching coordinates (`power_lat`, `power_lon`) from predictors
- Identify SoilGrids columns and:
    * Drop rows where ALL SoilGrids columns are missing (likely non-land / no-coverage)
    * Median-impute remaining missing values in SoilGrids columns
- Unified model: crop encoded as binary feature `is_wheat`
- Keep `lat`, `lon` in meta for spatial CV, but exclude from X by default.

Outputs (output_dir/tag/):
- X.parquet, y.parquet, meta.parquet
- feature_list.json
- data_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


SOIL_KEYWORDS = ["clay", "sand", "silt", "soc", "phh2o", "bdod", "cfvo"]
POWER_COORD_COLS = ["power_lat", "power_lon"]


# -----------------------------
# Helpers: column inference
# -----------------------------
def _best_col_match(cols: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first case-insensitive exact match, else None."""
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def infer_lat_lon_cols(cols: List[str]) -> Tuple[str, str]:
    lat = _best_col_match(cols, ["lat", "latitude", "y"])
    lon = _best_col_match(cols, ["lon", "longitude", "lng", "x"])
    if lat and lon:
        return lat, lon
    raise ValueError(
        f"Could not infer lat/lon columns. Found lat={lat}, lon={lon}. "
        f"Columns include: {list(cols)[:60]} ..."
    )


def infer_target_col(cols: List[str]) -> str:
    target = _best_col_match(
        cols,
        ["suitability", "gaez_suitability", "suitability_index", "target", "y"]
    )
    if target:
        return target
    raise ValueError(
        "Could not infer target column. Expected something like "
        "'suitability' / 'gaez_suitability' / 'suitability_index'. "
        f"Columns include: {list(cols)[:60]} ..."
    )


def infer_crop_col(cols: List[str]) -> str:
    crop = _best_col_match(cols, ["crop", "crop_name", "crop_type"])
    if crop:
        return crop
    for c in cols:
        if "crop" in c.lower():
            return c
    raise ValueError(
        "Could not infer crop column (e.g., 'crop'). "
        f"Columns include: {list(cols)[:60]} ..."
    )


def infer_soil_columns(cols: List[str]) -> List[str]:
    soil_cols = []
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in SOIL_KEYWORDS):
            soil_cols.append(c)
    return soil_cols


# -----------------------------
# Report
# -----------------------------
@dataclass
class PrepReport:
    input_path: str
    output_dir: str

    n_rows_in: int
    n_cols_in: int
    n_rows_out: int

    target_col: str
    crop_col: str
    lat_col: str
    lon_col: str

    dropped_cols: List[str]

    dropped_power_error_rows: int
    dropped_power_error_rate: float

    soil_cols: List[str]
    dropped_all_soil_missing_rows: int
    dropped_all_soil_missing_rate: float

    soil_missing_before_pct: dict
    soil_missing_after_pct: dict

    target_min: float
    target_max: float

    x_num_features: int


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to merged parquet/csv dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory for model-ready artifacts")
    parser.add_argument("--tag", default="v1", help="Tag name for this processed dataset version")
    parser.add_argument("--drop_latlon_from_X", action="store_true", help="Exclude lat/lon from X (recommended)")
    parser.add_argument("--target_clip_0_10000", action="store_true", help="Clip target to [0, 10000] (optional safeguard)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    elif in_path.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(in_path)
    else:
        raise ValueError(f"Unsupported input format: {in_path.suffix}")

    n_rows_in, n_cols_in = df.shape

    # Infer columns
    cols = list(df.columns)
    lat_col, lon_col = infer_lat_lon_cols(cols)
    target_col = infer_target_col(cols)
    crop_col = infer_crop_col(cols)

    dropped_cols: List[str] = []

    # 1) Handle NASA POWER retrieval failures
    dropped_power_rows = 0
    dropped_power_rate = 0.0
    if "power_error" in df.columns:
        fail_mask = df["power_error"].notna()
        dropped_power_rows = int(fail_mask.sum())
        dropped_power_rate = float(fail_mask.mean())
        if dropped_power_rows > 0:
            df = df.loc[~fail_mask].copy()
            print(f"Dropped {dropped_power_rows} rows due to power_error (NASA POWER retrieval failures).")
        df = df.drop(columns=["power_error"])
        dropped_cols.append("power_error")

    # Drop auxiliary POWER matching coords if present (not real predictors)
    for c in POWER_COORD_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
            dropped_cols.append(c)

    # 2) Soil columns
    soil_cols = infer_soil_columns(list(df.columns))
    if len(soil_cols) == 0:
        raise ValueError(
            "No SoilGrids columns inferred. Expected keywords: "
            f"{SOIL_KEYWORDS}. Check your column names."
        )

    # Missingness before (soil only)
    soil_missing_before = (df[soil_cols].isna().mean() * 100).round(6).to_dict()

    # Drop rows where ALL soil cols are missing (likely non-land / no-coverage)
    all_soil_missing = df[soil_cols].isna().all(axis=1)
    dropped_all_soil = int(all_soil_missing.sum())
    dropped_all_soil_rate = float(all_soil_missing.mean())
    if dropped_all_soil > 0:
        df = df.loc[~all_soil_missing].copy()
        print(
            f"Dropped {dropped_all_soil} rows where all SoilGrids vars were missing "
            f"(likely non-land/no-coverage)."
        )

    # Median impute remaining soil missing
    medians = df[soil_cols].median(numeric_only=True)
    df[soil_cols] = df[soil_cols].fillna(medians)

    soil_missing_after = (df[soil_cols].isna().mean() * 100).round(6).to_dict()

    # 3) Crop binary encoding
    crop_series = df[crop_col].astype(str).str.lower()
    df["is_wheat"] = crop_series.str.contains("wheat").astype(int)

    # 4) Optional target clip
    if args.target_clip_0_10000:
        df[target_col] = df[target_col].clip(0, 10000)

    # Build outputs
    meta = df[[lat_col, lon_col, crop_col]].copy()
    y = df[[target_col]].copy()

    exclude = {target_col, crop_col}
    if args.drop_latlon_from_X:
        exclude.update({lat_col, lon_col})

    X = df.drop(columns=[c for c in df.columns if c in exclude])

    # Final NA sanity
    if X.isna().any().any():
        na_cols = X.columns[X.isna().any()].tolist()
        miss_rates = (X[na_cols].isna().mean() * 100).sort_values(ascending=False).head(15)
        raise ValueError(
            f"X still contains missing values in columns: {na_cols[:25]} (showing up to 25).\n"
            f"Top missing-rate columns (%):\n{miss_rates.to_string()}\n"
            "You may need to handle these explicitly."
        )

    # Save artifacts
    X_path = out_dir / "X.parquet"
    y_path = out_dir / "y.parquet"
    meta_path = out_dir / "meta.parquet"

    X.to_parquet(X_path, index=False)
    y.to_parquet(y_path, index=False)
    meta.to_parquet(meta_path, index=False)

    feature_list = {
        "target": target_col,
        "crop_col": crop_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "soil_cols": soil_cols,
        "X_cols": list(X.columns),
    }
    with open(out_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(feature_list, f, indent=2)

    report = PrepReport(
        input_path=str(in_path),
        output_dir=str(out_dir),
        n_rows_in=n_rows_in,
        n_cols_in=n_cols_in,
        n_rows_out=int(df.shape[0]),
        target_col=target_col,
        crop_col=crop_col,
        lat_col=lat_col,
        lon_col=lon_col,
        dropped_cols=dropped_cols,
        dropped_power_error_rows=dropped_power_rows,
        dropped_power_error_rate=dropped_power_rate,
        soil_cols=soil_cols,
        dropped_all_soil_missing_rows=dropped_all_soil,
        dropped_all_soil_missing_rate=dropped_all_soil_rate,
        soil_missing_before_pct=soil_missing_before,
        soil_missing_after_pct=soil_missing_after,
        target_min=float(df[target_col].min()),
        target_max=float(df[target_col].max()),
        x_num_features=int(X.shape[1]),
    )
    with open(out_dir / "data_report.json", "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print("✅ Data preparation complete.")
    print(f"Saved: {out_dir}")
    print(f"Rows in:  {n_rows_in:,}")
    print(f"Rows out: {df.shape[0]:,}")
    if dropped_power_rows:
        print(f"Dropped (power_error): {dropped_power_rows:,} = {dropped_power_rate:.3%}")
    if dropped_all_soil:
        print(f"Dropped (all-soil-missing): {dropped_all_soil:,} = {dropped_all_soil_rate:.3%}")
    print(f"X features: {X.shape[1]:,}")
    print(f"Target: {target_col} | Added crop feature: is_wheat")


if __name__ == "__main__":
    main()