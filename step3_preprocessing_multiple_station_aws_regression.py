#!/usr/bin/env python3

import json
import gzip
from io import BytesIO, StringIO

import boto3
import numpy as np
import pandas as pd

# =========================
# S3 / IO CONFIG
# =========================
REGION_NAME = "eu-north-1"

# Input bucket: 1-minute CSVs from Step 2
BUCKET_IN = "tica93-dmi-all-stations"
IN_ROOT = "1min"                                # 1min/{station_id}/...
IN_SUFFIX = "_1min_DMI_2020_2025.csv"           # NEW: 2020–2025

# Output bucket: regression preprocessed parquet
BUCKET_OUT = "tica93-dmi-all-stations-regresion"

s3 = boto3.client("s3", region_name=REGION_NAME)

# =========================
# Feature knobs
# =========================
LAG_MINUTES = [1, 2, 3, 5, 10]
ROLL_SUM_WINDOWS = [10, 30, 60]
ROLL_MEAN_WINDOWS = [5, 10, 30, 60]
HORIZON = 10   # predict up to y_mm_t+10 and y_flag_t+10

BASE_COLS = [
    "time", "precip_past1min", "precip_flag_1min",
    "temp_dry", "humidity", "cloud_cover",
    "wind_speed", "wind_dir_deg", "wind_dir_sin", "wind_dir_cos",
]

# Minimal set used later for LightGBM binary model (Step 4); we keep for compatibility
LGBM_BASE_COLS = [
    "time", "station_id",
    "humidity", "temp", "cloud_cover", "wind_speed",
    "wind_dir_sin", "wind_dir_cos",
    "precip_past1min", "precip_flag_1min",
    "gap_after_prev",
    "y_flag_t+1",
]


# =========================
# Helpers
# =========================
def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast floats/ints to save space."""
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64", "int32"]).columns:
        mn, mx = df[c].min(), df[c].max()
        if 0 <= mn <= mx <= 255:
            df[c] = df[c].astype("uint8")
        elif -128 <= mn <= mx <= 127:
            df[c] = df[c].astype("int8")
        else:
            df[c] = df[c].astype("int32")
    return df


def _read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    print(f"Downloading s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    try:
        # try gzip
        df = pd.read_csv(StringIO(gzip.decompress(body).decode("utf-8")))
    except Exception:
        df = pd.read_csv(BytesIO(body))
    return df


def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        raise ValueError("Input must contain a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def _clip_physicals(df: pd.DataFrame) -> pd.DataFrame:
    """Clip to physically realistic ranges."""
    if "precip_past1min" in df.columns:
        df["precip_past1min"] = df["precip_past1min"].clip(lower=0.0)
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(lower=0.0, upper=100.0)
    if "cloud_cover" in df.columns:
        df["cloud_cover"] = df["cloud_cover"].clip(lower=0.0, upper=1.0)
    if "wind_speed" in df.columns:
        df["wind_speed"] = df["wind_speed"].clip(lower=0.0)
    if "wind_dir_deg" in df.columns:
        df["wind_dir_deg"] = (df["wind_dir_deg"] % 360.0)
    return df


def _add_targets(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Add future rainfall (mm) and rain-flag targets up to given horizon."""
    if "precip_past1min" not in df.columns:
        df["precip_past1min"] = 0.0
    for h in range(1, horizon + 1):
        df[f"y_mm_t+{h}"] = df["precip_past1min"].shift(-h)
        df[f"y_flag_t+{h}"] = (df[f"y_mm_t+{h}"] > 0.0).astype("float32")
    return df


def _add_lags_and_rolls(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged rainfall / flags, rolling sums/means, and derivatives."""
    # Lags on rainfall
    if "precip_past1min" in df.columns:
        for L in LAG_MINUTES:
            df[f"lag_precip_{L}m"] = df["precip_past1min"].shift(L)
    if "precip_flag_1min" in df.columns:
        for L in LAG_MINUTES:
            df[f"lag_pf_{L}m"] = df["precip_flag_1min"].shift(L)

    # Rolling sums of rainfall
    if "precip_past1min" in df.columns:
        for W in ROLL_SUM_WINDOWS:
            df[f"roll_precip_sum_{W}m"] = df["precip_past1min"].rolling(W, min_periods=1).sum()

    # Rolling means for key meteo variables
    mean_cols = [c for c in ["temp_dry", "humidity", "cloud_cover", "wind_speed"]
                 if c in df.columns]
    for col in mean_cols:
        for W in ROLL_MEAN_WINDOWS:
            df[f"{col}_mean_{W}m"] = df[col].rolling(W, min_periods=1).mean()

    # Simple finite differences (short-term trends)
    for col in ["temp_dry", "humidity", "wind_speed"]:
        if col in df.columns:
            df[f"d_{col}_1m"] = df[col].diff(1)
            df[f"d_{col}_5m"] = df[col].diff(5)

    # Wind direction smoothing in sin/cos space (if present)
    if "wind_dir_sin" in df.columns and "wind_dir_cos" in df.columns:
        for W in [5, 10, 30]:
            df[f"wind_dir_sin_mean_{W}m"] = df["wind_dir_sin"].rolling(W, min_periods=1).mean()
            df[f"wind_dir_cos_mean_{W}m"] = df["wind_dir_cos"].rolling(W, min_periods=1).mean()

    # Calendar/time features
    t = pd.to_datetime(df["time"], utc=True)
    df["minute"] = t.dt.minute.astype("int16")
    df["hour"]   = t.dt.hour.astype("int8")
    df["dow"]    = t.dt.weekday.astype("int8")
    df["month"]  = t.dt.month.astype("int8")
    df["year"]   = t.dt.year.astype("int16")

    # Cyclic encodings
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24.0)

    return df


def _final_clean(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Drop rows with missing future targets; mild gap filling."""
    mask = pd.Series(True, index=df.index)
    for h in range(1, horizon + 1):
        mask &= df[f"y_mm_t+{h}"].notna()
    df = df.loc[mask].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    # short-range forward/backward filling to remove tiny holes
    df = df.fillna(method="ffill", limit=10).fillna(method="bfill", limit=10)
    return df


def _save_parquet_month(
    df_month: pd.DataFrame,
    year: int,
    month: int,
    station_id: str,
    all_cols: list,
):
    """Save full feature+target parquet for one station, one month."""
    # align schema
    missing = [c for c in all_cols if c not in df_month.columns]
    for c in missing:
        df_month[c] = np.nan
    extra = [c for c in df_month.columns if c not in all_cols]
    if extra:
        df_month = df_month.drop(columns=extra)
    df_month = df_month.reindex(columns=all_cols)

    df_month = _downcast_numeric(df_month)

    buf = BytesIO()
    df_month.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)

    out_prefix = f"{station_id}_preproc_1min"
    key = f"{out_prefix}/year={year}/month={month:02d}/{station_id}_preproc_{year}-{month:02d}.parquet"
    s3.upload_fileobj(buf, BUCKET_OUT, key)
    print(f"  wrote → s3://{BUCKET_OUT}/{key}   (rows={len(df_month):,})")


def _save_parquet_month_lgbm_base(
    df_month: pd.DataFrame,
    year: int,
    month: int,
    station_id: str,
):
    """Save slim 'base' parquet with a fixed schema for the binary model."""
    df_out = df_month.copy()
    for c in LGBM_BASE_COLS:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out = df_out[LGBM_BASE_COLS].copy()

    float_cols = df_out.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df_out[float_cols] = df_out[float_cols].astype("float32")

    buf = BytesIO()
    df_out.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)

    out_prefix = f"{station_id}_preproc_1min"
    key = (
        f"{out_prefix}/year={year}/month={month:02d}/"
        f"{station_id}_preproc_{year}-{month:02d}__lightgbm_base.parquet"
    )
    s3.upload_fileobj(buf, BUCKET_OUT, key)
    print(f"  wrote (LGBM base) → s3://{BUCKET_OUT}/{key}   (rows={len(df_out):,})")


# =========================
# Station discovery
# =========================
def discover_1min_files() -> dict:
    """
    Returns mapping: {station_id: key_to_1min_csv}
    where key looks like '1min/06138/06138_1min_DMI_2020_2025.csv'
    in the *input* bucket BUCKET_IN.
    """
    print(f"Scanning input bucket {BUCKET_IN} for 1-min files...")
    mapping = {}
    continuation_token = None

    prefix = IN_ROOT + "/" if IN_ROOT else ""

    while True:
        kwargs = {"Bucket": BUCKET_IN, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(IN_SUFFIX):
                continue
            # key = "1min/06138/06138_1min_DMI_2020_2025.csv"
            fname = key.split("/")[-1]
            station_id = fname.split("_")[0]
            mapping.setdefault(station_id, key)

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    print(f"Found {len(mapping)} station(s): {sorted(mapping.keys())}")
    return mapping


# =========================
# Per-station processing
# =========================
def process_station(station_id: str, key_1min: str):
    """
    Process a single station:
      - read 1-min CSV (2020–2025) from BUCKET_IN
      - physical clipping
      - add multi-horizon targets
      - add lags/rolling features
      - partition by year/month into BUCKET_OUT
    NOTE: We do NOT skip stations that already exist in BUCKET_OUT;
          this allows updating with new years (e.g. adding 2025).
    """
    print("\n==============================")
    print(f" PREPROCESSING STATION {station_id}")
    print(f"    from s3://{BUCKET_IN}/{key_1min}")
    print("==============================")

    df = _read_csv_from_s3(BUCKET_IN, key_1min)

    # Hard requirement: have minute precipitation
    if "precip_past1min" not in df.columns:
        print(f"{station_id} does NOT have precip_past1min → EXCLUDED")
        return

    # Make sure precip is numeric and create 1-min flag if missing
    df["precip_past1min"] = pd.to_numeric(df["precip_past1min"], errors="coerce")
    if "precip_flag_1min" not in df.columns:
        df["precip_flag_1min"] = (df["precip_past1min"] > 0.0).astype("float32")

    # Base subset of columns we care about (plus time)
    keep = [c for c in BASE_COLS if c in df.columns]
    if "time" not in keep:
        keep = ["time"] + keep
    df = df[keep].copy()

    df = _ensure_time(df)
    df = _clip_physicals(df)

    # Targets, lags, rolls, time encodings
    df = _add_targets(df, horizon=HORIZON)
    df = _add_lags_and_rolls(df)
    df = _final_clean(df, horizon=HORIZON)

    df["station_id"] = str(station_id)
    df = _downcast_numeric(df)

    # Standardize names for later steps (binary model etc.)
    if "temp" not in df.columns:
        df["temp"] = df.get("temp_dry", np.nan)
    for c in ["wind_dir_sin", "wind_dir_cos"]:
        if c not in df.columns:
            df[c] = np.nan

    # y_flag_t+1 for compatibility with classification step
    if "y_flag_t+1" not in df.columns:
        if "y_mm_t+1" in df.columns:
            df["y_flag_t+1"] = (df["y_mm_t+1"] > 0.0).astype("int8")
        else:
            base_flag = df.get("precip_flag_1min", 0.0).astype("float32")
            df["y_flag_t+1"] = base_flag.shift(-1).fillna(0).astype("int8")
    else:
        df["y_flag_t+1"] = df["y_flag_t+1"].astype("int8")

    dt = pd.to_datetime(df["time"], utc=True)
    df["gap_after_prev"] = (dt.diff().dt.total_seconds().fillna(60) > 60).astype("int8")

    # Freeze schema: features first, then y_*
    feature_first = [c for c in df.columns if not c.startswith("y_")]
    target_last   = [c for c in df.columns if c.startswith("y_")]
    all_cols = feature_first + target_last

    # Persist schema.json per station in BUCKET_OUT
    out_prefix = f"{station_id}_preproc_1min"
    try:
        s3.put_object(
            Bucket=BUCKET_OUT,
            Key=f"{out_prefix}/schema.json",
            Body=json.dumps({"columns": all_cols}, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(
            f"Wrote schema → s3://{BUCKET_OUT}/{out_prefix}/schema.json "
            f"(cols={len(all_cols)})"
        )
    except Exception as e:
        print(f"Could not write schema.json for station {station_id}: {e}")

    # Partition by year-month (based on time)
    df["year"] = pd.to_datetime(df["time"], utc=True).dt.year.astype("int16")
    df["month"] = pd.to_datetime(df["time"], utc=True).dt.month.astype("int8")

    print(f"Writing monthly parquet for station {station_id}…")
    for (yr, mo), g in df.groupby(["year", "month"], sort=True):
        part = g.drop(columns=["year", "month"])
        _save_parquet_month(part.copy(), int(yr), int(mo), station_id, all_cols)
        _save_parquet_month_lgbm_base(part.copy(), int(yr), int(mo), station_id)

    print(
        f"Done station {station_id}. Preprocessed under:\n"
        f"    s3://{BUCKET_OUT}/{station_id}_preproc_1min/year=*/month=*/{station_id}_preproc_YYYY-MM*.parquet"
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    stations = discover_1min_files()
    if not stations:
        print("No 1-min files found. Check IN_ROOT / IN_SUFFIX / BUCKET_IN.")
        raise SystemExit(1)

    for station_id, key in sorted(stations.items()):
        process_station(station_id, key)

    print("\n All stations preprocessed for 2020–2025 (regression).")
