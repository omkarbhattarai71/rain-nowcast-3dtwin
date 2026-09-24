#!/usr/bin/env python3

import boto3
import pandas as pd
import numpy as np
from io import BytesIO
import pyarrow.parquet as pq
import pyarrow as pa

# ================================
# S3 CONFIGURATION
# ================================
REGION = "eu-north-1"

s3 = boto3.client("s3", region_name=REGION)

RADAR_BUCKET = "tica93-dmi-radar-cache"
RADAR_PREFIX = "derived/station_timeseries"

WEATHER_BUCKET = "tica93-dmi-all-stations-regresion"
OUTPUT_SUFFIX = "_preproc_1min_radar"   # NEW output folder suffix


# =====================================================
# Helpers — read/write Parquet from S3
# =====================================================
def read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return pq.read_table(BytesIO(data)).to_pandas()


def write_parquet_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = BytesIO()
    table = pa.Table.from_pandas(df)
    pq.write_table(table, buf)
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, key)


# =====================================================
# Discover radar months from station_timeseries paths
# =====================================================
def discover_radar_months():
    """
    Return list of (year, month) that have any radar station-timeseries.
    Looks under:
      s3://RADAR_BUCKET/derived/station_timeseries/YYYY/MM/DD/*.parquet
    """
    months = set()
    token = None

    while True:
        kwargs = {"Bucket": RADAR_BUCKET, "Prefix": RADAR_PREFIX}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            # derived/station_timeseries/YYYY/MM/DD/file.parquet
            parts = key.split("/")
            if len(parts) < 5:
                continue
            try:
                year = int(parts[2])
                month = int(parts[3])
                months.add((year, month))
            except Exception:
                continue

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    return sorted(months)


# =====================================================
# Month processing
# =====================================================
def process_month(year: int, month: int) -> None:
    print("\n==============================")
    print(f" Processing month {year}-{month:02d}")
    print("==============================")

    # ---- Load ALL radar daily files for this month ----
    radar_prefix_month = f"{RADAR_PREFIX}/{year}/{month:02d}/"
    radar_keys = []
    token = None

    while True:
        kwargs = {"Bucket": RADAR_BUCKET, "Prefix": radar_prefix_month}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                radar_keys.append(obj["Key"])

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    if not radar_keys:
        print("  No radar station TS files for this month, skipping.")
        return

    radar_frames = []
    for key in sorted(radar_keys):
        print(f"   Loading radar TS: s3://{RADAR_BUCKET}/{key}")
        df_day = read_parquet_s3(RADAR_BUCKET, key)
        radar_frames.append(df_day)

    radar_df = pd.concat(radar_frames, ignore_index=True)
    print(f"📡 Loaded radar rows: {len(radar_df):,}")
    print(f"   Radar columns: {list(radar_df.columns)}")

    # ---- Detect time column ----
    time_candidates = [
        "time",
        "radar_time",
        "datetime",
        "timestamp",
        "validtime",
        "valid_time",
    ]
    time_col = None
    for cand in time_candidates:
        if cand in radar_df.columns:
            time_col = cand
            break

    if time_col is None:
        print(" No suitable time-like column found in radar_df.")
        print("   Columns:", list(radar_df.columns))
        return

    # ---- Detect station column ----
    station_candidates = ["station_id", "station", "stid"]
    station_col = None
    for cand in station_candidates:
        if cand in radar_df.columns:
            station_col = cand
            break

    if station_col is None:
        print(" No suitable station column found in radar_df.")
        return

    # ---- Detect radar reflectivity column ----
    # Prefer calibrated dbzh_dbz if available
    value_candidates = ["dbzh_dbz", "radar_dbzh", "DBZH", "dbzh", "dbzh_raw"]
    radar_value_col = None
    for cand in value_candidates:
        if cand in radar_df.columns:
            radar_value_col = cand
            break

    if radar_value_col is None:
        print(" No radar DBZH column found.")
        print("   Columns:", list(radar_df.columns))
        return

    # ---- Normalise radar dataframe ----
    radar_df["time"] = pd.to_datetime(radar_df[time_col], utc=True, errors="coerce")

    # normalise station_id to 5-digit strings (leading zeros!)
    s = radar_df[station_col].astype(str).str.strip()
    # if purely numeric and length < 5, zero-pad
    s = s.where(~s.str.match(r"^\d+$"), s.str.zfill(5))
    radar_df["station_id"] = s

    radar_df = radar_df[["station_id", "time", radar_value_col]].rename(
        columns={radar_value_col: "radar_dbzh"}
    )

    radar_df = radar_df.sort_values(["station_id", "time"]).reset_index(drop=True)

    print(
        "   Unique radar stations in "
        f"{year}-{month:02d}: {sorted(radar_df['station_id'].unique())[:20]} ..."
    )

    # ---- Find matching weather preproc monthly files ----
    print(" Searching weather preprocessed parquet files for this month...")
    weather_keys = []
    token = None
    needle = f"year={year}/month={month:02d}/"

    while True:
        kwargs = {"Bucket": WEATHER_BUCKET}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if (
                needle in k
                and "_preproc_1min/" in k
                and k.endswith(".parquet")
                and OUTPUT_SUFFIX not in k  # avoid already radar-merged files
            ):
                weather_keys.append(k)

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    if not weather_keys:
        print("  No preprocessed weather files for this month, skipping.")
        return

    print(f"🌦 Found {len(weather_keys)} weather preproc files for {year}-{month:02d}")

    # ---- Process each station/month weather file ----
    for wkey in sorted(weather_keys):
        print(f"\n Merging radar into {wkey}")

        # Example: "05005_preproc_1min/year=2025/month=09/05005_preproc_2025-09.parquet"
        station_prefix = wkey.split("_preproc_1min")[0]
        station_id = station_prefix.split("/")[-1]

        wdf = read_parquet_s3(WEATHER_BUCKET, wkey)

        # Ensure columns exist
        if "time" not in wdf.columns:
            print(f"   Weather file {wkey} has no 'time' column, skipping.")
            continue
        if "station_id" not in wdf.columns:
            print(f"   Weather file {wkey} has no 'station_id' column, skipping.")
            continue

        wdf["time"] = pd.to_datetime(wdf["time"], utc=True, errors="coerce")
        # normalise weather station_id the same way (5-digit)
        wdf["station_id"] = (
            wdf["station_id"].astype(str).str.strip().str.zfill(5)
        )

        # Select radar rows for this station (IDs now consistently 5-digit)
        rad = radar_df[radar_df["station_id"] == station_id].copy()

        if rad.empty:
            print(
                f"   No radar points for station {station_id} in "
                f"{year}-{month:02d}. Writing weather-only copy."
            )
            merged = wdf.copy()
        else:
            # Merge (left join to keep all weather rows)
            merged = pd.merge(
                wdf,
                rad,
                on=["time", "station_id"],
                how="left",
            )

            # Rolling radar features (time-based windows)
            merged = merged.sort_values("time").set_index("time")
            merged["radar_dbzh_max_30m"] = merged["radar_dbzh"].rolling("30min").max()
            merged["radar_dbzh_mean_30m"] = merged["radar_dbzh"].rolling("30min").mean()
            merged["radar_dbzh_max_60m"] = merged["radar_dbzh"].rolling("60min").max()
            merged = merged.reset_index()

        # ------------------------------
        # Save to new S3 output folder
        # ------------------------------
        out_key = wkey.replace("_preproc_1min", OUTPUT_SUFFIX)
        print(f"  ⬆️ Writing radar-enhanced file → s3://{WEATHER_BUCKET}/{out_key}")
        write_parquet_s3(merged, WEATHER_BUCKET, out_key)

    print(f"\n✅ Finished month: {year}-{month:02d}")


# =====================================================
# MAIN
# =====================================================
def main():
    months = discover_radar_months()
    print(f" Scanning radar station TS under s3://{RADAR_BUCKET}/{RADAR_PREFIX}/ ...")
    print(f" Radar station TS months found: {months}")

    if not months:
        print(" No radar station_timeseries months found. Exiting.")
        return

    for (year, month) in months:
        process_month(year, month)

    print("\n ALL DONE — radar successfully merged into preprocessed regression data!")


if __name__ == "__main__":
    main()



