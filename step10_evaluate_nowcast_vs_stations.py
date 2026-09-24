#!/usr/bin/env python3

import re
from io import BytesIO
from typing import List

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =====================================================
# CONFIG
# =====================================================
REGION = "eu-north-1"
s3 = boto3.client("s3", region_name=REGION)

# Input = Step 9 output
WEATHER_BUCKET = "tica93-dmi-all-stations-regresion"
INPUT_SUFFIX = "_preproc_1min_radar"

# Output = Step 11 output
OUTPUT_SUFFIX = "_preproc_1min_radar_pysteps"

# PySTEPS daily station-nowcast files
PYSTEPS_BUCKET = "tica93-dmi-radar-cache"
PYSTEPS_PREFIX = "derived/pysteps_station_nowcasts"

# Matching tolerance for causal asof merge
PYSTEPS_TOLERANCE = pd.Timedelta("30s")

# Behavior when no PySTEPS exists for a month/file
SKIP_FILE_IF_NO_PYSTEPS = False

# If True, overwrite existing Step 11 output
OVERWRITE_EXISTING = True


# =====================================================
# S3 HELPERS
# =====================================================
def read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pq.read_table(BytesIO(obj["Body"].read())).to_pandas()


def write_parquet_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
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


# =====================================================
# NORMALIZATION HELPERS
# =====================================================
def ensure_utc_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def normalize_station_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\D+", "", regex=True)
    s = s.str.zfill(5)
    return s


def infer_days(df: pd.DataFrame) -> List[str]:
    days = df["time"].dt.strftime("%Y-%m-%d").dropna().unique().tolist()
    days.sort()
    return days


def pysteps_key_for_day(day_str: str) -> str:
    yyyy, mm, dd = day_str.split("-")
    return f"{PYSTEPS_PREFIX}/{yyyy}/{mm}/{dd}/pysteps_nowcast_{day_str}.parquet"


# =====================================================
# DISCOVERY
# =====================================================
def discover_weather_months():
    """
    Discover months available under:
      s3://WEATHER_BUCKET/{station_id}_preproc_1min_radar/year=YYYY/month=MM/*.parquet
    """
    months = set()
    token = None

    while True:
        kwargs = {"Bucket": WEATHER_BUCKET}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]

            if INPUT_SUFFIX not in key:
                continue
            if not key.endswith(".parquet"):
                continue
            if OUTPUT_SUFFIX in key:
                continue

            m = re.search(r"year=(\d{4})/month=(\d{2})/", key)
            if not m:
                continue

            year = int(m.group(1))
            month = int(m.group(2))
            months.add((year, month))

        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break

    return sorted(months)


def find_weather_keys_for_month(year: int, month: int) -> List[str]:
    keys = []
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
                and INPUT_SUFFIX in k
                and k.endswith(".parquet")
                and OUTPUT_SUFFIX not in k
            ):
                keys.append(k)

        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break

    return sorted(keys)


# =====================================================
# PYTSTEPS LOADING
# =====================================================
def load_pysteps_days(days: List[str]) -> pd.DataFrame:
    parts = []
    missing_days = []

    for d in days:
        key = pysteps_key_for_day(d)
        if not s3_key_exists(PYSTEPS_BUCKET, key):
            missing_days.append(d)
            continue

        df = read_parquet_s3(PYSTEPS_BUCKET, key)

        if "time" not in df.columns or "station_id" not in df.columns:
            print(f"  ⚠️ PySTEPS file missing required columns: s3://{PYSTEPS_BUCKET}/{key}")
            continue

        df = df.copy()
        df["time"] = ensure_utc_datetime(df["time"])
        df["station_id"] = normalize_station_id(df["station_id"])
        df = df.dropna(subset=["time", "station_id"]).copy()

        parts.append(df)

    if missing_days:
        preview = missing_days[:10]
        suffix = " ..." if len(missing_days) > 10 else ""
        print(f"  ⚠️ Missing PySTEPS days: {preview}{suffix}")

    if not parts:
        return pd.DataFrame(columns=["station_id", "time"])

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["station_id", "time"]).reset_index(drop=True)

    # Defensive dedupe on raw PySTEPS input
    before = len(out)
    out = out.drop_duplicates(subset=["station_id", "time"], keep="last").reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"  ⚠️ Dropped {dropped:,} duplicate raw PySTEPS rows")

    return out


def standardize_pysteps_columns(df_p: pd.DataFrame) -> pd.DataFrame:
    """
    Rename:
      time -> pysteps_time
      every other payload col -> pysteps_<col>
    while keeping station_id unchanged.
    """
    df_p = df_p.copy()

    rename_map = {"time": "pysteps_time"}

    for c in df_p.columns:
        if c in ("station_id", "time"):
            continue
        if c.startswith("pysteps_"):
            continue
        rename_map[c] = f"pysteps_{c}"

    df_p = df_p.rename(columns=rename_map)

    ordered = ["station_id", "pysteps_time"] + [
        c for c in df_p.columns if c not in ("station_id", "pysteps_time")
    ]
    return df_p[ordered]


# =====================================================
# MERGE
# =====================================================
def merge_pysteps_causal(
    df_weather: pd.DataFrame,
    df_pysteps: pd.DataFrame,
    tolerance: pd.Timedelta = PYSTEPS_TOLERANCE,
) -> pd.DataFrame:
    # ---------- normalize weather side ----------
    df_w = df_weather.copy()
    df_w["time"] = ensure_utc_datetime(df_w["time"])
    df_w["station_id"] = normalize_station_id(df_w["station_id"])
    df_w = df_w.dropna(subset=["time", "station_id"]).copy()

    # debug duplicate count in left side before dedupe
    dup_mask_w = df_w.duplicated(subset=["station_id", "time"], keep=False)
    n_dup_w = int(dup_mask_w.sum())
    print(f"  Duplicate rows already in weather input: {n_dup_w:,}")
    if n_dup_w:
        print(
            df_w.loc[dup_mask_w, ["station_id", "time"]]
            .sort_values(["station_id", "time"])
            .head(20)
            .to_string(index=False)
        )

    df_w = df_w.sort_values(["station_id", "time"]).reset_index(drop=True)

    before_w = len(df_w)
    df_w = df_w.drop_duplicates(subset=["station_id", "time"], keep="first").copy()
    dropped_w = before_w - len(df_w)
    if dropped_w:
        print(f"  ⚠️ Dropped {dropped_w:,} duplicate weather rows before PySTEPS merge")

    if df_pysteps.empty:
        out = df_w.copy()
        out["pysteps_time"] = pd.NaT
        out["pysteps_age_min"] = pd.NA
        return out

    # ---------- normalize pysteps side ----------
    df_p = standardize_pysteps_columns(df_pysteps)
    df_p["pysteps_time"] = ensure_utc_datetime(df_p["pysteps_time"])
    df_p["station_id"] = normalize_station_id(df_p["station_id"])
    df_p = df_p.dropna(subset=["pysteps_time", "station_id"]).copy()

    dup_mask_p = df_p.duplicated(subset=["station_id", "pysteps_time"], keep=False)
    n_dup_p = int(dup_mask_p.sum())
    print(f"  Duplicate rows already in PySTEPS input: {n_dup_p:,}")

    df_p = df_p.sort_values(["station_id", "pysteps_time"]).reset_index(drop=True)

    before_p = len(df_p)
    df_p = df_p.drop_duplicates(subset=["station_id", "pysteps_time"], keep="last").copy()
    dropped_p = before_p - len(df_p)
    if dropped_p:
        print(f"  ⚠️ Dropped {dropped_p:,} duplicate PySTEPS rows before merge")

    # IMPORTANT: for merge_asof, sort by merge key first
    df_w = df_w.sort_values(["time", "station_id"]).reset_index(drop=True)
    df_p = df_p.sort_values(["pysteps_time", "station_id"]).reset_index(drop=True)

    merged = pd.merge_asof(
        df_w,
        df_p,
        left_on="time",
        right_on="pysteps_time",
        by="station_id",
        direction="backward",
        allow_exact_matches=True,
        tolerance=tolerance,
    )

    merged["pysteps_age_min"] = (
        (merged["time"] - merged["pysteps_time"]).dt.total_seconds() / 60.0
    )

    return merged


# =====================================================
# QUALITY CHECKS
# =====================================================
def quality_checks(df: pd.DataFrame, key: str) -> None:
    total = len(df)
    matched = int(df["pysteps_time"].notna().sum()) if "pysteps_time" in df.columns else 0
    coverage = 100.0 * matched / total if total else 0.0

    print(f"  Rows: {total:,}")
    print(f"  PySTEPS matched: {matched:,} / {total:,} ({coverage:.2f}%)")

    if "pysteps_time" in df.columns:
        bad_future = int((df["pysteps_time"] > df["time"]).sum())
        if bad_future:
            print(f"  ❌ Future leakage rows: {bad_future:,}")
            ex = df.loc[df["pysteps_time"] > df["time"], ["station_id", "time", "pysteps_time"]].head(5)
            print(ex.to_string(index=False))
            raise RuntimeError(f"Future leakage detected in {key}")

        bad_neg_age = int((df["pysteps_age_min"] < -1e-9).sum())
        if bad_neg_age:
            print(f"  ❌ Negative age rows: {bad_neg_age:,}")
            ex = df.loc[df["pysteps_age_min"] < 0, ["station_id", "time", "pysteps_time", "pysteps_age_min"]].head(5)
            print(ex.to_string(index=False))
            raise RuntimeError(f"Negative pysteps_age_min detected in {key}")

    dup = int(df.duplicated(subset=["station_id", "time"]).sum())
    if dup:
        print(f"  ❌ Duplicate station_id,time rows after merge: {dup:,}")
        raise RuntimeError(f"Duplicate station/time rows detected in {key}")

    print("  ✅ Quality checks passed.")


# =====================================================
# PROCESSING
# =====================================================
def process_month(year: int, month: int) -> None:
    print("\n" + "=" * 70)
    print(f"📅 STEP 11 — Processing month {year}-{month:02d}")
    print("=" * 70)

    weather_keys = find_weather_keys_for_month(year, month)
    if not weather_keys:
        print("⚠️ No radar-enhanced weather files found for this month.")
        return

    print(f"🌦 Found {len(weather_keys)} radar-enhanced station-month files")

    for wkey in weather_keys:
        out_key = wkey.replace(INPUT_SUFFIX, OUTPUT_SUFFIX)

        if (not OVERWRITE_EXISTING) and s3_key_exists(WEATHER_BUCKET, out_key):
            print(f"\n⏭ Output already exists, skipping: s3://{WEATHER_BUCKET}/{out_key}")
            continue

        print(f"\n➡️ Reading weather+rader file: s3://{WEATHER_BUCKET}/{wkey}")
        wdf = read_parquet_s3(WEATHER_BUCKET, wkey)

        if "time" not in wdf.columns:
            print("  ⚠️ Missing 'time' column, skipping.")
            continue
        if "station_id" not in wdf.columns:
            print("  ⚠️ Missing 'station_id' column, skipping.")
            continue

        wdf = wdf.copy()
        wdf["time"] = ensure_utc_datetime(wdf["time"])
        wdf["station_id"] = normalize_station_id(wdf["station_id"])
        wdf = wdf.dropna(subset=["time", "station_id"]).copy()

        station_ids = sorted(wdf["station_id"].unique().tolist())
        days = infer_days(wdf)

        print(f"  Station ids: {station_ids[:10]}{' ...' if len(station_ids) > 10 else ''}")
        print(f"  Days in file: {len(days)}")

        df_pysteps = load_pysteps_days(days)

        if df_pysteps.empty:
            msg = "No PySTEPS found for these days."
            if SKIP_FILE_IF_NO_PYSTEPS:
                print(f"  ⚠️ {msg} Skipping file.")
                continue
            else:
                print(f"  ⚠️ {msg} Writing file with empty PySTEPS columns.")

        overlap = (
            sorted(set(station_ids).intersection(set(df_pysteps["station_id"].unique().tolist())))
            if not df_pysteps.empty
            else []
        )
        print(f"  PySTEPS station overlap: {overlap[:10]}{' ...' if len(overlap) > 10 else ''}")

        merged = merge_pysteps_causal(wdf, df_pysteps, tolerance=PYSTEPS_TOLERANCE)

        quality_checks(merged, wkey)

        print(f"  ⬆️ Writing merged file → s3://{WEATHER_BUCKET}/{out_key}")
        write_parquet_s3(merged, WEATHER_BUCKET, out_key)

    print(f"\n✅ Finished month {year}-{month:02d}")


# =====================================================
# MAIN
# =====================================================
def main():
    months = discover_weather_months()
    print(f"🔎 Found months under radar-enhanced weather files: {months}")

    if not months:
        print("⛔ No input months found. Exiting.")
        return

    for year, month in months:
        process_month(year, month)

    print("\n🎉 Step 10 finished — weather + radar + PySTEPS merged successfully.")


if __name__ == "__main__":
    main()