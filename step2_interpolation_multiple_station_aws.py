#!/usr/bin/env python3
import boto3
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import gzip
import botocore
import re

# ============================================================
# S3 SETTINGS
# ============================================================
BUCKET_NAME = "tica93-dmi-all-stations"
REGION_NAME = "eu-north-1"

# If your raw files are under a folder, e.g. "raw_10min/", set this:
RAW_PREFIX = ""   # e.g. "raw_10min/"

# Where to write 1-min outputs
OUT_ROOT = "1min"   # => 1min/{station_id}/{station_id}_1min_DMI_2020_2025.csv

s3 = boto3.client("s3", region_name=REGION_NAME)

# ---------- interpolation knobs ----------
SHORT_GAP_LIMIT_MIN = 60       # only interpolate across gaps <= 60 min for met vars
PRECIP_BLOCK_FILL   = True     # use 10-min block expansion for precip_past10min
# ----------------------------------------


# ============================================================
# Small helpers
# ============================================================
def ensure_bucket_exists(bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"☑️ Bucket '{bucket}' exists.")
    except Exception:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": REGION_NAME},
        )
        print(f"Created bucket '{bucket}'.")


def _bound_clip(series, lower=None, upper=None):
    if series is None:
        return series
    if lower is not None:
        series = series.clip(lower=lower)
    if upper is not None:
        series = series.clip(upper=upper)
    return series


def _normalize_cloud_cover(s):
    """Map cloud cover to [0,1] if it looks like 0-100 input."""
    if s.dropna().empty:
        return s
    mx = s.max(skipna=True)
    if mx is not None and mx > 1.5:
        return s / 100.0
    return s


def _cap_interpolation(series, limit_minutes):
    """
    Interpolate linearly in 'time' for gaps of at most limit_minutes.
    Larger gaps left for optional ffill/bfill later.
    """
    if series.empty:
        return series
    return series.interpolate(method="time", limit=limit_minutes, limit_direction="both")


def _resample_to_minute(series_like, full_idx, unit_bounds=None,
                        limit_minutes=SHORT_GAP_LIMIT_MIN):
    """
    Generic helper: align to minute index and interpolate linearly in time with bounds.
    """
    s = pd.to_numeric(series_like, errors="coerce")
    s = s.reindex(full_idx)
    s = _cap_interpolation(s, limit_minutes)
    if unit_bounds:
        s = _bound_clip(s, unit_bounds[0], unit_bounds[1])
    return s


def _handle_wind_direction_deg(deg_series_10min, full_idx):
    """
    Interpolate wind direction circularly by projecting to sin/cos, interpolating,
    then projecting back to degrees. Also returns sin/cos columns (for ML).
    """
    s = pd.to_numeric(deg_series_10min, errors="coerce")
    s = s.reindex(full_idx)  # align to minute grid (NaNs inserted)
    rad = np.deg2rad((s % 360).astype(float))
    sin = np.sin(rad)
    cos = np.cos(rad)

    sin_i = _cap_interpolation(pd.Series(sin, index=full_idx), SHORT_GAP_LIMIT_MIN)
    cos_i = _cap_interpolation(pd.Series(cos, index=full_idx), SHORT_GAP_LIMIT_MIN)

    length = np.sqrt(sin_i**2 + cos_i**2)
    length = length.replace(0, np.nan)
    sin_n = sin_i / length
    cos_n = cos_i / length

    deg = (np.rad2deg(np.arctan2(sin_n, cos_n)) + 360.0) % 360.0

    out = pd.DataFrame(index=full_idx)
    out["wind_dir_deg"] = deg
    out["wind_dir_sin"] = sin_n
    out["wind_dir_cos"] = cos_n
    return out


# ============================================================
# Interpolation core
# ============================================================
def interpolate_to_1min(df: pd.DataFrame) -> pd.DataFrame | None:
    if "time" not in df.columns:
        return None

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")
    if df.empty:
        return None

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
    out = pd.DataFrame(index=full_idx)

    # ---------------------
    # Precipitation logic
    # ---------------------
    has_1min = "precip_past1min" in df.columns and df["precip_past1min"].notna().any()
    has_10min = "precip_past10min" in df.columns and df["precip_past10min"].notna().any()

    if has_1min:
        # Minute precip: keep on minute grid; assume missing = 0.0 (no rain)
        s = pd.to_numeric(df["precip_past1min"], errors="coerce").reindex(full_idx)
        s = s.fillna(0.0)
        out["precip_past1min"] = _bound_clip(s, 0.0, None)
        out["precip_flag_1min"] = (out["precip_past1min"] > 0).astype(np.int8)

    elif has_10min and PRECIP_BLOCK_FILL:
        # Expand each 10-min sum into a 10-min block with equal per-minute mm.
        s10 = pd.to_numeric(df["precip_past10min"], errors="coerce").dropna()
        out["precip_past1min"] = 0.0
        # Assumption: each 10-min value corresponds to the past 10 minutes ending at t
        for t, val in s10.items():
            t_aligned = pd.Timestamp(t).tz_convert("UTC").floor("min")
            block = pd.date_range(
                t_aligned - pd.Timedelta(minutes=9),
                t_aligned,
                freq="1min",
                tz="UTC",
            )
            block = block.intersection(full_idx)
            if len(block) > 0:
                out.loc[block, "precip_past1min"] = float(val) / max(len(block), 1)
        out["precip_past1min"] = _bound_clip(out["precip_past1min"].fillna(0.0), 0.0, None)
        out["precip_flag_1min"] = (out["precip_past1min"] > 0).astype(np.int8)

    # ---------------------
    # Temperature (°C): linear interp; modest gaps only
    # ---------------------
    if "temp_dry" in df.columns:
        out["temp_dry"] = _resample_to_minute(df["temp_dry"], full_idx, unit_bounds=None)

    # ---------------------
    # Humidity: scale to [0,100] if fractional; clip; linear interp
    # ---------------------
    if "humidity" in df.columns:
        h = pd.to_numeric(df["humidity"], errors="coerce")
        if h.max(skipna=True) is not None and h.max(skipna=True) <= 1.5:
            h = h * 100.0
        h.index = df.index
        out["humidity"] = _resample_to_minute(h, full_idx, unit_bounds=(0.0, 100.0))

    # ---------------------
    # Cloud cover: normalize to [0,1]; linear interp; clip
    # ---------------------
    if "cloud_cover" in df.columns:
        cc = _normalize_cloud_cover(pd.to_numeric(df["cloud_cover"], errors="coerce"))
        out["cloud_cover"] = _resample_to_minute(cc, full_idx, unit_bounds=(0.0, 1.0))

    # ---------------------
    # Wind speed (m/s): non-negative; linear interp
    # ---------------------
    if "wind_speed" in df.columns:
        out["wind_speed"] = _resample_to_minute(df["wind_speed"], full_idx, unit_bounds=(0.0, None))

    # ---------------------
    # Wind direction (deg): circular interpolation via sin/cos
    # ---------------------
    if "wind_dir" in df.columns:
        w = _handle_wind_direction_deg(df["wind_dir"], full_idx)
        out = out.join(w, how="outer")

    # Cleanup: small forward/backfill pass for remaining tiny holes
    out = out.ffill(limit=SHORT_GAP_LIMIT_MIN).bfill(limit=SHORT_GAP_LIMIT_MIN)

    # Drop columns that stayed empty
    out = out.dropna(axis=1, how="all")

    # Finalize
    out.insert(0, "time", out.index)
    return out


# ============================================================
# Station discovery + processing (NEW, more robust)
# ============================================================
def discover_stations() -> dict:
    """
    Scan the bucket for 10-min files and return:
      {station_id: best_key}

    Any file whose basename matches:
      ^(\d+)_10min_DMI_(\d{4})_(\d{4})\.csv$

    For each station_id, we pick the file with the **largest end_year**,
    so if both 2020_2024 and 2020_2025 exist, 2020_2025 is used.
    """
    print("Scanning bucket for 10-min files (all years)...")
    station_to_best = {}  # station_id -> (end_year, key)

    continuation_token = None
    pattern = re.compile(r"^(\d+)_10min_DMI_(\d{4})_(\d{4})\.csv$")

    while True:
        list_kwargs = {"Bucket": BUCKET_NAME}
        if RAW_PREFIX:
            list_kwargs["Prefix"] = RAW_PREFIX
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**list_kwargs)
        contents = resp.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            fname = key.split("/")[-1]

            m = pattern.match(fname)
            if not m:
                continue

            station_id = m.group(1)
            end_year = int(m.group(3))

            # Update if this station is new or this file has a later end_year
            if station_id not in station_to_best or end_year > station_to_best[station_id][0]:
                station_to_best[station_id] = (end_year, key)

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    # Flatten to {station_id: key}
    station_to_key = {sid: info[1] for sid, info in station_to_best.items()}
    print(f"Found {len(station_to_key)} station(s) with 10-min files:")
    print("   ", ", ".join(sorted(station_to_key.keys())))
    return station_to_key


def read_10min_csv_from_s3(key: str) -> pd.DataFrame:
    print(f"  Reading {key}")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    body = obj["Body"].read()

    # Auto-handle gzip vs plain
    try:
        df = pd.read_csv(StringIO(gzip.decompress(body).decode("utf-8")))
    except Exception:
        df = pd.read_csv(BytesIO(body))

    return df


def write_1min_csv_to_s3(df_1min: pd.DataFrame, station_id: str):
    # For naming, we now use a generic "2020_2025" in output,
    # since most stations now cover this range. If some stations are 2020–2024,
    # they will still be stored under this name, just with an earlier end time.
    out_key = f"{OUT_ROOT}/{station_id}/{station_id}_1min_DMI_2020_2025.csv"
    print(f"  Uploading → s3://{BUCKET_NAME}/{out_key}")
    buf = BytesIO()
    df_1min.to_csv(buf, index=False)
    buf.seek(0)
    s3.upload_fileobj(buf, BUCKET_NAME, out_key)


def process_station(station_id: str, key: str):
    """
    Process single station:
      - load raw 10-min CSV
      - optional filter by station_id column
      - interpolate to 1-min
      - upload result
    """
    print("\n==============================")
    print(f" STATION {station_id}")
    print("==============================")

    try:
        df = read_10min_csv_from_s3(key)
    except botocore.exceptions.ClientError as e:
        print(f"  Error reading {key}: {e}")
        return

    # Safety filter if multiple stations are in one file
    if "station_id" in df.columns:
        df = df[df["station_id"].astype(str) == str(station_id)]
        if df.empty:
            print(f"  No rows for station_id={station_id} in {key}, skipping.")
            return

    df_1min = interpolate_to_1min(df)
    if df_1min is None or df_1min.empty:
        print("  Nothing to save (empty 1-min table). Skipping.")
        return

    write_1min_csv_to_s3(df_1min, station_id)
    print(" Finished.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    ensure_bucket_exists(BUCKET_NAME)
    station_to_key = discover_stations()

    if not station_to_key:
        print("No 10-min files found. Check RAW_PREFIX / naming pattern.")
        raise SystemExit(1)

    for stid, key in sorted(station_to_key.items()):
        process_station(stid, key)

    print("\n All stations processed for 1-min 2020–2025 (where available)!")

