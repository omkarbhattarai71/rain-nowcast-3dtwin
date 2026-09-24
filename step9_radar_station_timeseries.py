#!/usr/bin/env python3

import re
import sys
import math
import datetime as dt
from io import BytesIO

import boto3
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =========================================================
# CONFIG
# =========================================================
REGION = "eu-north-1"
RADAR_BUCKET = "tica93-dmi-radar-cache"

RAW_PREFIX = "pseudoCappi"
LOOKUP_KEY = "stations_xy_lookup.csv"
OUT_PREFIX = "derived/station_timeseries"

DEFAULT_ZR_A = 200.0
DEFAULT_ZR_B = 1.6
NODATA_DEFAULT = 255.0

FNAME_RE = re.compile(r".*?(\d{8})_(\d{4})\..*?\.h5$")

s3 = boto3.client("s3", region_name=REGION)


# =========================================================
# HELPERS
# =========================================================
def daterange(start_date: dt.date, end_date: dt.date):
    delta = (end_date - start_date).days
    for i in range(delta + 1):
        yield start_date + dt.timedelta(days=i)


def s3_list_keys(bucket: str, prefix: str):
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def s3_read_bytes(bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def s3_write_parquet(df: pd.DataFrame, bucket: str, key: str):
    buf = BytesIO()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, key)


def s3_key_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def normalize_station_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\D+", "", regex=True)
    s = s.str.zfill(5)
    return s


def parse_inside_grid(col: pd.Series) -> pd.Series:
    if col.dtype == object:
        norm = col.astype(str).str.strip().str.lower()
        return norm.isin(["1", "true", "t", "yes", "y"])
    return col.astype(bool)


def load_station_lookup() -> pd.DataFrame:
    print(f" Loading station lookup from s3://{RADAR_BUCKET}/{LOOKUP_KEY}")
    b = s3_read_bytes(RADAR_BUCKET, LOOKUP_KEY)
    df = pd.read_csv(BytesIO(b))

    required = {"station_id", "ix", "iy"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Lookup CSV missing required columns {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["station_id"] = normalize_station_id(df["station_id"])
    df["ix"] = pd.to_numeric(df["ix"], errors="coerce")
    df["iy"] = pd.to_numeric(df["iy"], errors="coerce")
    df = df.dropna(subset=["ix", "iy"]).copy()
    df["ix"] = df["ix"].astype(int)
    df["iy"] = df["iy"].astype(int)

    if "inside_grid" in df.columns:
        df = df[parse_inside_grid(df["inside_grid"])].copy()

    df = df.drop_duplicates(subset=["station_id"], keep="first").reset_index(drop=True)

    print(f" Loaded {len(df)} station(s) inside radar grid.")
    return df


def parse_radar_time_from_filename(fname: str) -> dt.datetime:
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Cannot parse radar time from filename: {fname}")

    ymd = m.group(1)
    hm = m.group(2)

    return dt.datetime(
        int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]),
        int(hm[0:2]), int(hm[2:4]),
        tzinfo=dt.timezone.utc,
    )


def read_odim_dbzh_grid(h5_bytes: bytes):
    """
    Returns:
      dbzh_dbz: float32 2D array
      nodata_mask: bool 2D
      zr_a, zr_b: floats
    """
    with h5py.File(BytesIO(h5_bytes), "r") as f:
        data = f["dataset1/data1/data"][()]  # uint8
        what = f["dataset1/what"].attrs

        gain = float(what.get("gain", 1.0))
        offset = float(what.get("offset", 0.0))
        nodata = float(what.get("nodata", NODATA_DEFAULT))

        zr_a = DEFAULT_ZR_A
        zr_b = DEFAULT_ZR_B
        if "how" in f:
            how = f["how"].attrs
            if "zr-a" in how:
                zr_a = float(how["zr-a"])
            if "zr-b" in how:
                zr_b = float(how["zr-b"])

        data_f = data.astype("float32")
        nodata_mask = data_f == nodata
        dbzh_dbz = gain * data_f + offset
        dbzh_dbz[nodata_mask] = np.nan

    return dbzh_dbz, nodata_mask, zr_a, zr_b


def dbzh_to_rainrate(dbz: np.ndarray, zr_a: float, zr_b: float) -> np.ndarray:
    z_lin = np.power(10.0, dbz / 10.0, dtype="float32")
    rain = np.power(np.maximum(z_lin / float(zr_a), 0.0), 1.0 / float(zr_b), dtype="float32")
    rain = np.where(np.isfinite(dbz), rain, np.nan).astype("float32")
    return rain


def sample_grid_at_stations(grid: np.ndarray, stations: pd.DataFrame) -> pd.DataFrame:
    h, w = grid.shape
    rows = []

    for _, row in stations.iterrows():
        stid = row["station_id"]
        ix = int(row["ix"])
        iy = int(row["iy"])

        inside = (0 <= ix < w) and (0 <= iy < h)
        if inside:
            val = float(grid[iy, ix]) if np.isfinite(grid[iy, ix]) else np.nan
        else:
            val = np.nan

        rows.append({
            "station_id": stid,
            "ix": ix,
            "iy": iy,
            "value": val,
        })

    return pd.DataFrame(rows)


def process_day(day: dt.date, stations: pd.DataFrame, overwrite: bool = False):
    day_prefix = f"{RAW_PREFIX}/{day.year:04d}/{day.month:02d}/{day.day:02d}/"
    out_key = (
        f"{OUT_PREFIX}/{day.year:04d}/{day.month:02d}/{day.day:02d}/"
        f"radar_station_timeseries_{day.isoformat()}.parquet"
    )

    print("\n==============================")
    print(f" Processing day {day.isoformat()}")
    print("==============================")

    if (not overwrite) and s3_key_exists(RADAR_BUCKET, out_key):
        print(f"⏭ Already exists: s3://{RADAR_BUCKET}/{out_key}")
        return

    keys = [k for k in s3_list_keys(RADAR_BUCKET, day_prefix) if k.endswith(".h5")]
    keys = sorted(keys)

    if not keys:
        print("No raw radar HDF5 files found for this day.")
        return

    print(f"Found {len(keys)} raw radar file(s).")

    frames = []
    for key in keys:
        fname = key.split("/")[-1]

        try:
            radar_time = parse_radar_time_from_filename(fname)
        except Exception as e:
            print(f"  Skipping file with unparsable timestamp: {fname} ({e})")
            continue

        try:
            raw = s3_read_bytes(RADAR_BUCKET, key)
            dbzh_dbz, _, zr_a, zr_b = read_odim_dbzh_grid(raw)
            rainrate_mm_h = dbzh_to_rainrate(dbzh_dbz, zr_a, zr_b)

            df_dbz = sample_grid_at_stations(dbzh_dbz, stations).rename(columns={"value": "dbzh_dbz"})
            df_rr = sample_grid_at_stations(rainrate_mm_h, stations).rename(columns={"value": "rainrate_mm_h"})

            df = df_dbz.merge(df_rr[["station_id", "rainrate_mm_h"]], on="station_id", how="left")
            df["time"] = pd.Timestamp(radar_time)
            df["source_key"] = key

            frames.append(df)
            print(f"  Sampled {fname} -> {len(df)} station rows")

        except Exception as e:
            print(f"  Failed reading {key}: {e}")

    if not frames:
        print(" No valid station time-series rows produced for this day.")
        return

    out = pd.concat(frames, ignore_index=True)
    out = out[["station_id", "time", "dbzh_dbz", "rainrate_mm_h", "ix", "iy", "source_key"]]
    out = out.sort_values(["station_id", "time"]).reset_index(drop=True)

    print(f" Writing {len(out):,} rows -> s3://{RADAR_BUCKET}/{out_key}")
    s3_write_parquet(out, RADAR_BUCKET, out_key)
    print(" Done.")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python step9_radar_station_timeseries.py YYYY-MM-DD")
        print("  python step9_radar_station_timeseries.py YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)

    if len(sys.argv) == 2:
        start = end = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    else:
        start = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        end = dt.datetime.strptime(sys.argv[2], "%Y-%m-%d").date()

    stations = load_station_lookup()

    for day in daterange(start, end):
        process_day(day, stations, overwrite=False)


if __name__ == "__main__":
    main()