import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Config
# -----------------------------
POWER_BASE = "https://power.larc.nasa.gov/api/temporal/climatology/point"
COMMUNITY = "AG"
START_YEAR = 1981
END_YEAR = 2010
FORMAT = "JSON"

# Final feature list (pilot + 2 extra)
POWER_PARAMS = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MIN",
    "T2M_MAX",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
    "T2MDEW",
]

# Concurrency (start conservative; reduce if you hit 429)
MAX_WORKERS = 12

# Retry/backoff
MAX_RETRIES = 6
BASE_SLEEP = 1.0  # seconds


# -----------------------------
# I/O helpers
# -----------------------------
def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file type: {p.suffix} (use .csv or .parquet)")


def write_any(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
        return
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
        return
    raise ValueError(f"Unsupported output type: {p.suffix} (use .csv or .parquet)")


# -----------------------------
# POWER helpers
# -----------------------------
def build_power_url(lat: float, lon: float) -> str:
    params_str = ",".join(POWER_PARAMS)
    return (
        f"{POWER_BASE}"
        f"?parameters={params_str}"
        f"&community={COMMUNITY}"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&format={FORMAT}"
        f"&start={START_YEAR}"
        f"&end={END_YEAR}"
    )


def flatten_power_parameters(power_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens:
      json["properties"]["parameter"][PARAM][PERIOD]  -> columns PARAM_PERIOD
    PERIOD typically includes: JAN..DEC and ANN (annual).
    Also captures grid-used coordinates if present.
    """
    out: Dict[str, Any] = {}

    props = power_json.get("properties", {})
    param_block = props.get("parameter", {})

    # Nearest grid point used by POWER (often returned as geometry.coordinates: [lon, lat])
    geom = power_json.get("geometry", {})
    coords = geom.get("coordinates", None)
    if isinstance(coords, list) and len(coords) >= 2:
        out["power_lon"] = coords[0]
        out["power_lat"] = coords[1]

    if "elevation" in props:
        out["power_elevation_m"] = props.get("elevation")

    for p_name, p_vals in param_block.items():
        if isinstance(p_vals, dict):
            for period, value in p_vals.items():
                out[f"{p_name}_{period}"] = value
        else:
            out[p_name] = p_vals

    return out


def fetch_power_point(session: requests.Session, cell_id: Any, lat: float, lon: float) -> Dict[str, Any]:
    url = build_power_url(lat, lon)

    last_err: Optional[str] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=60)

            # rate limit / transient errors
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}"
                sleep_s = BASE_SLEEP * (2 ** (attempt - 1)) + 0.1 * attempt
                time.sleep(min(sleep_s, 30))
                continue

            resp.raise_for_status()
            data = resp.json()

            flat = flatten_power_parameters(data)
            flat["cell_id"] = cell_id
            return flat

        except Exception as e:
            last_err = str(e)
            sleep_s = BASE_SLEEP * (2 ** (attempt - 1)) + 0.1 * attempt
            time.sleep(min(sleep_s, 30))

    return {"cell_id": cell_id, "power_error": last_err}


def enrich_with_power(df: pd.DataFrame) -> pd.DataFrame:
    required = {"cell_id", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    rows = df[["cell_id", "lat", "lon"]].to_dict("records")

    results: List[Dict[str, Any]] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "gaez-power-pipeline/1.0"})
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = []
            for r in rows:
                # skip totally invalid coords but keep row
                if pd.isna(r["lat"]) or pd.isna(r["lon"]):
                    results.append({"cell_id": r["cell_id"], "power_error": "NaN lat/lon"})
                    continue
                futs.append(ex.submit(fetch_power_point, session, r["cell_id"], float(r["lat"]), float(r["lon"])))

            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 1000 == 0:
                    print(f"Fetched {done}/{len(futs)} points")

    power_df = pd.DataFrame(results)

    # Merge by stable key (not exact lat/lon)
    out = df.merge(power_df, on="cell_id", how="left")
    return out


def main(
    wheat_path: str,
    maize_path: str,
    out_combined_path: str,
    out_wheat_path: Optional[str] = None,
    out_maize_path: Optional[str] = None,
) -> None:
    wheat = read_any(wheat_path)
    maize = read_any(maize_path)

    combined = pd.concat([wheat, maize], ignore_index=True)

    enriched = enrich_with_power(combined)

    write_any(enriched, out_combined_path)
    print(f"✅ wrote combined: {out_combined_path}")

    if out_wheat_path and "crop" in enriched.columns:
        write_any(enriched[enriched["crop"].str.lower() == "wheat"], out_wheat_path)
        print(f"✅ wrote wheat: {out_wheat_path}")

    if out_maize_path and "crop" in enriched.columns:
        write_any(enriched[enriched["crop"].str.lower() == "maize"], out_maize_path)
        print(f"✅ wrote maize: {out_maize_path}")


if __name__ == "__main__":
    main(
    wheat_path="data/interim/points/wheat_points_50k.parquet",
    maize_path="data/interim/points/maize_points_50k.parquet",
    out_combined_path="data/processed/wheat_maize_100k_power_1981_2010.parquet",
    )

