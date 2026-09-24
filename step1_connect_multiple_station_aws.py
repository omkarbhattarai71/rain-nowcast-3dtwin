#!/usr/bin/env python3
import boto3
import pandas as pd
import requests
import time
from io import BytesIO
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
API_KEY = "ff1f56f8-522a-4e00-9108-4f5cf651e084"
BASE_URL = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"

BUCKET_NAME = "tica93-dmi-all-stations"
REGION_NAME = "eu-north-1"

START_YEAR = 2020
END_YEAR_HIST = 2024
END_YEAR_NEW = 2025                 # NEW YEAR TO ADD

STATIONS = [
    # ... your full list of station IDs ...
]

PARAMS_10MIN = [
    "temp_dry",
    "precip_past10min",
    "wind_speed",
    "wind_dir",
    "humidity",
    "cloud_cover",
    "precip_past1min"
]

s3 = boto3.client("s3", region_name=REGION_NAME)


# =========================================================
# HELPERS
# =========================================================
def ensure_bucket(bucket):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"☑️ Bucket exists: {bucket}")
    except:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": REGION_NAME}
        )
        print(f"Created bucket {bucket}")


def fetch_param(station_id, param, start_date, end_date):
    """Fetch DMI observations for a single parameter."""
    params = {
        "api-key": API_KEY,
        "stationId": station_id,
        "parameterId": param,
        "datetime": f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
        "limit": 300000
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
        if r.status_code == 403:
            return None
        if r.status_code != 200:
            return None
        feats = r.json().get("features", [])
        if not feats:
            return []
        return [{"time": f["properties"]["observed"], param: f["properties"]["value"]}
                for f in feats]
    except:
        return None


def check_available_params(station_id):
    """Detect which parameters exist for this station."""
    available = []
    test_start = datetime(2020, 6, 1)
    test_end = datetime(2020, 6, 5)
    for p in PARAMS_10MIN:
        sample = fetch_param(station_id, p, test_start, test_end)
        if sample is None:
            continue
        if len(sample) > 0:
            available.append(p)
    return available


def load_existing_csv(station_id):
    """Load existing 2020–2024 CSV from S3 if exists."""
    key = f"{station_id}_10min_DMI_2020_2024.csv"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        df = pd.read_csv(obj["Body"])
        print(f"Found existing historical file: {key}")
        return df
    except:
        return None


def upload_csv(df, station_id):
    """Safe multipart upload to S3."""
    key = f"{station_id}_10min_DMI_2020_2025.csv"
    buf = BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)

    try:
        s3.upload_fileobj(buf, BUCKET_NAME, key)
        print(f"Uploaded merged 2020–2025 → s3://{BUCKET_NAME}/{key}")
    except Exception as e:
        print(f"Upload failed for station {station_id}: {e}")
        raise


# =========================================================
# MAIN LOOP
# =========================================================
ensure_bucket(BUCKET_NAME)

for station_id in STATIONS:
    print("\n" + "="*70)
    print(f"🌦 Checking station {station_id}")
    print("="*70)

    # ----------------------------------------------------------------------
    # 1. Station must have precip_past10min
    # ----------------------------------------------------------------------
    test = fetch_param(station_id, "precip_past10min",
                       datetime(2020, 6, 1), datetime(2020, 6, 30))
    if not test:
        print(f"No precip_past10min → skipping {station_id}")
        continue

    # ----------------------------------------------------------------------
    # 2. Detect available parameters
    # ----------------------------------------------------------------------
    print(f" Detecting available parameters for {station_id}...")
    available = check_available_params(station_id)
    if not available:
        print(f"No usable parameters → skipping {station_id}")
        continue

    print(f"{station_id} available parameters: {', '.join(available)}")

    # ----------------------------------------------------------------------
    # 3. Load existing 2020–2024 data
    # ----------------------------------------------------------------------
    df_hist = load_existing_csv(station_id)

    if df_hist is None:
        print(f"No historical CSV for {station_id}, skipping.")
        continue

    df_hist["time"] = pd.to_datetime(df_hist["time"])

    # ----------------------------------------------------------------------
    # 4. Download NEW data (2025 ONLY)
    # ----------------------------------------------------------------------
    print(f"Downloading 2025 data for {station_id}...")
    df_new_full = pd.DataFrame()

    for param in available:
        recs = fetch_param(
            station_id,
            param,
            datetime(END_YEAR_HIST + 1, 1, 1),
            datetime(END_YEAR_NEW, 12, 31)
        )
        time.sleep(0.15)

        if recs:
            df_p = pd.DataFrame(recs)
            df_p["time"] = pd.to_datetime(df_p["time"])
            df_p.set_index("time", inplace=True)
            if df_new_full.empty:
                df_new_full = df_p
            else:
                df_new_full = df_new_full.join(df_p, how="outer")

    if df_new_full.empty:
        print(f"No 2025 data for {station_id}, skip merge.")
        continue

    df_new_full.reset_index(inplace=True)

    # ----------------------------------------------------------------------
    # 5. MERGE 2020–2024 + 2025
    # ----------------------------------------------------------------------
    df_final = pd.concat([df_hist, df_new_full], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["time"])
    df_final.sort_values("time", inplace=True)

    # ----------------------------------------------------------------------
    # 6. Upload merged CSV
    # ----------------------------------------------------------------------
    upload_csv(df_final, station_id)

print("\n DONE — all stations processed.")


