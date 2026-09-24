#!/usr/bin/env python3

from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional, Tuple

import boto3
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# CONFIG
# =========================================================
REGION = "eu-north-1"
BUCKET = "tica93-dmi-all-stations-regresion"

PARQUET_SUFFIX = "__lightgbm_base.parquet"

TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
VAL_YEAR = 2025

TARGET_MM = "y_mm_t+1"

RESULTS_PREFIX = "model_outputs_regression_2025"
RESULTS_PREFIX_FILTERED = "model_outputs_regression_2025_filtered"

MIN_RAINY_MIN_PER_MONTH = 50

s3 = boto3.client("s3", region_name=REGION)


# =========================================================
# S3 HELPERS
# =========================================================
def save_csv(df: pd.DataFrame, key: str) -> None:
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=df.to_csv(index=False).encode("utf-8"),
        ContentType="text/csv",
    )
    print(f"  💾 Saved CSV     → s3://{BUCKET}/{key}   (rows={len(df):,})")


def save_parquet(df: pd.DataFrame, key: str) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=buf.read(),
        ContentType="application/octet-stream",
    )
    print(f"  Saved Parquet → s3://{BUCKET}/{key}   (rows={len(df):,})")


def read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def s3_key_exists(bucket: str, key: str) -> bool:
    r = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return "Contents" in r and len(r["Contents"]) > 0


# =========================================================
# DISCOVERY
# =========================================================
def list_parquets() -> List[dict]:
    """
    Scan BUCKET for station-only __lightgbm_base.parquet files.

    Keep only:
        {station_id}_preproc_1min/year=YYYY/month=MM/*__lightgbm_base.parquet

    Exclude:
        *_preproc_1min_radar/*
        *_preproc_1min_radar_pysteps/*
    """
    out = []
    token = None
    print(" Scanning regression bucket for station-only LGBM base parquets...")

    while True:
        kw = {"Bucket": BUCKET}
        if token:
            kw["ContinuationToken"] = token

        r = s3.list_objects_v2(**kw)

        for obj in r.get("Contents", []):
            key = obj["Key"]

            if not key.endswith(PARQUET_SUFFIX):
                continue

            # keep only station-only branch
            if "_preproc_1min/" not in key:
                continue

            # exclude radar branches
            if "_preproc_1min_radar/" in key:
                continue
            if "_preproc_1min_radar_pysteps/" in key:
                continue

            parts = key.split("/")
            if len(parts) < 4:
                continue

            try:
                station_id = parts[0].split("_")[0]
                year = int(parts[1].split("=")[1])
                month = int(parts[2].split("=")[1])
            except Exception:
                continue

            out.append(
                {
                    "key": key,
                    "station_id": station_id,
                    "year": year,
                    "month": month,
                }
            )

        if r.get("IsTruncated"):
            token = r.get("NextContinuationToken")
        else:
            break

    print(f" Found {len(out):,} station-only regression parquet shards")
    return out


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build regression feature matrix from one monthly shard.

    Drops:
      - time
      - station_id
      - all y_* targets
    """
    keep = df.copy()

    drop_cols = []
    for c in keep.columns:
        if c == "time" or c == "station_id" or c.startswith("y_"):
            drop_cols.append(c)

    X = keep.drop(columns=drop_cols, errors="ignore").copy()

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype("int8")
        elif str(X[c].dtype) == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.infer_objects(copy=False)
    X = X.fillna(0.0)

    for c in X.select_dtypes(include=["float64"]).columns:
        X[c] = X[c].astype("float32")
    for c in X.select_dtypes(include=["int64", "int32"]).columns:
        X[c] = X[c].astype("int32")

    y = pd.to_numeric(df[TARGET_MM], errors="coerce").values.astype(float)
    return X, y


# =========================================================
# METRICS
# =========================================================
def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0.0, None)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(y_pred.mean() - y_true.mean())

    mask_rain = y_true > 0.0
    if mask_rain.sum() > 0:
        y_t_r = y_true[mask_rain]
        y_p_r = y_pred[mask_rain]
        mse_r = mean_squared_error(y_t_r, y_p_r)
        rmse_r = float(np.sqrt(mse_r))
        mae_r = float(mean_absolute_error(y_t_r, y_p_r))
    else:
        rmse_r = float("nan")
        mae_r = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": bias,
        "rmse_rainy": rmse_r,
        "mae_rainy": mae_r,
    }


def station_regression_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in df.groupby("station_id"):
        y_t = g["y_true_mm"].values.astype(float)
        y_p = g["y_pred_mm"].values.astype(float)
        m = eval_regression(y_t, y_p)
        m.update(
            {
                "station_id": sid,
                "rows": int(len(g)),
                "n_rainy": int((y_t > 0.0).sum()),
            }
        )
        rows.append(m)

    return pd.DataFrame(rows)


# =========================================================
# EXISTING-MONTH REUSE
# =========================================================
def month_has_trained_preds(m: int) -> bool:
    key = f"{RESULTS_PREFIX}/month={m:02d}/preds_{VAL_YEAR}_regression.parquet"
    return s3_key_exists(BUCKET, key)


def load_existing_month(m: int) -> Optional[Tuple[dict, pd.DataFrame]]:
    mp = f"{RESULTS_PREFIX}/month={m:02d}"
    pred_key = f"{mp}/preds_{VAL_YEAR}_regression.parquet"
    if not s3_key_exists(BUCKET, pred_key):
        return None

    print(f"  ⏭ Reusing existing month {m:02d}: s3://{BUCKET}/{pred_key}")

    df_pred = read_parquet_s3(BUCKET, pred_key)

    metrics_key = f"{mp}/metrics_global_regression.csv"
    if s3_key_exists(BUCKET, metrics_key):
        obj = s3.get_object(Bucket=BUCKET, Key=metrics_key)
        gdf = pd.read_csv(BytesIO(obj["Body"].read()))
        g = gdf.iloc[0].to_dict()
    else:
        g = eval_regression(df_pred["y_true_mm"].values, df_pred["y_pred_mm"].values)
        g.update({"month": m, "best_iteration": np.nan, "filtered": 0})

    return g, df_pred


# =========================================================
# FILTERED METRICS
# =========================================================
def compute_and_save_filtered_metrics(
    m: int,
    df_pred: pd.DataFrame,
    station_metrics_all: pd.DataFrame,
) -> Optional[dict]:
    mp_f = f"{RESULTS_PREFIX_FILTERED}/month={m:02d}"

    comp = station_metrics_all.copy()
    good_mask = comp["n_rainy"] >= MIN_RAINY_MIN_PER_MONTH
    good_stations = comp.loc[good_mask, "station_id"].astype(str).tolist()
    comp["included"] = comp["station_id"].isin(good_stations).astype(int)

    save_csv(
        comp,
        f"{RESULTS_PREFIX_FILTERED}/station_comparison/month{m:02d}_comparison.csv",
    )

    if len(good_stations) == 0:
        print(f"  ⚠️ Month {m:02d}: no stations with >= {MIN_RAINY_MIN_PER_MONTH} rainy minutes.")
        return None

    df_f = df_pred[df_pred["station_id"].isin(good_stations)].copy()
    gf = eval_regression(df_f["y_true_mm"].values, df_f["y_pred_mm"].values)
    gf.update(
        {
            "month": m,
            "filtered": 1,
            "min_rainy_min_per_month": MIN_RAINY_MIN_PER_MONTH,
            "n_stations": len(good_stations),
        }
    )

    save_csv(pd.DataFrame([gf]), f"{mp_f}/metrics_global_filtered.csv")
    save_csv(
        station_metrics_all.loc[good_mask].copy(),
        f"{RESULTS_PREFIX_FILTERED}/station_metrics/month{m:02d}_filtered.csv",
    )
    save_parquet(df_f, f"{mp_f}/preds_{VAL_YEAR}_regression_filtered.parquet")
    save_csv(df_f, f"{mp_f}/preds_{VAL_YEAR}_regression_filtered.csv")

    return gf


# =========================================================
# TRAINING
# =========================================================
def train_month(m: int, meta: List[dict]) -> Optional[Tuple[dict, pd.DataFrame]]:
    """
    Train and evaluate one monthly regression model.

    Training:
        years 2020-2024, months {prev, m, next}
    Validation:
        year 2025, month m
    """
    print("\n==============================")
    print(f"  TRAINING REGRESSION FOR MONTH {m:02d}")
    print("==============================")

    prev = 12 if m == 1 else m - 1
    nxt = 1 if m == 12 else m + 1
    train_months = {prev, m, nxt}

    Xtr_list, ytr_list = [], []
    Xv_list, yv_list = [], []
    sid_v, time_v = [], []

    n_train_shards = 0
    n_val_shards = 0

    for s in meta:
        year = s["year"]
        mo = s["month"]

        if year not in TRAIN_YEARS and not (year == VAL_YEAR and mo == m):
            continue

        df = read_parquet_s3(BUCKET, s["key"])

        if TARGET_MM not in df.columns:
            continue

        df = df.dropna(subset=[TARGET_MM])
        if df.empty:
            continue

        X, y = extract_features_targets(df)

        if year in TRAIN_YEARS and mo in train_months:
            Xtr_list.append(X)
            ytr_list.append(y)
            n_train_shards += 1
        elif year == VAL_YEAR and mo == m:
            Xv_list.append(X)
            yv_list.append(y)
            sid_v.append(df["station_id"].astype(str).values)
            time_v.append(pd.to_datetime(df["time"], utc=True).values)
            n_val_shards += 1

    print(f"  Train months used: {sorted(train_months)}")
    print(f"  Number of train shards collected: {n_train_shards}")
    print(f"  Number of validation shards collected: {n_val_shards}")

    if not Xtr_list or not Xv_list:
        print(f" Month {m:02d}: insufficient data, skipped.")
        return None

    Xtr = pd.concat(Xtr_list, ignore_index=True)
    ytr = np.concatenate(ytr_list)
    Xv = pd.concat(Xv_list, ignore_index=True)
    yv = np.concatenate(yv_list)
    sid = np.concatenate(sid_v)
    t = np.concatenate(time_v)

    print(f"  Training samples:   {len(Xtr):,}")
    print(f"  Validation samples: {len(Xv):,}")
    print(f"  Feature count:      {Xtr.shape[1]:,}")

    dtr = lgb.Dataset(Xtr, label=ytr)
    dval = lgb.Dataset(Xv, label=yv)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 40,
        "lambda_l1": 1.0,
        "lambda_l2": 2.0,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        dtr,
        num_boost_round=5000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score (val rmse): {model.best_score['valid_0']['rmse']:.4f}")

    y_pred = model.predict(Xv, num_iteration=model.best_iteration)
    y_pred = np.clip(y_pred, 0.0, None)

    df_pred = pd.DataFrame(
        {
            "station_id": sid,
            "time": t,
            "y_true_mm": yv,
            "y_pred_mm": y_pred,
        }
    )
    df_pred["month"] = m

    global_metrics_all = eval_regression(yv, y_pred)
    global_metrics_all.update(
        {
            "month": m,
            "best_iteration": int(model.best_iteration),
            "filtered": 0,
            "n_features": int(Xtr.shape[1]),
        }
    )

    station_metrics_all = station_regression_metrics(df_pred)

    mp = f"{RESULTS_PREFIX}/month={m:02d}"

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{mp}/model_regression.txt",
        Body=model.model_to_string().encode("utf-8"),
    )

    fi = pd.DataFrame(
        {
            "feature": Xtr.columns,
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    save_csv(fi, f"{mp}/feature_importance.csv")

    save_csv(pd.DataFrame([global_metrics_all]), f"{mp}/metrics_global_regression.csv")
    save_csv(
        station_metrics_all,
        f"{RESULTS_PREFIX}/station_metrics_regression_month{m:02d}.csv",
    )
    save_parquet(df_pred, f"{mp}/preds_{VAL_YEAR}_regression.parquet")
    save_csv(df_pred, f"{mp}/preds_{VAL_YEAR}_regression.csv")

    compute_and_save_filtered_metrics(m, df_pred, station_metrics_all)

    return global_metrics_all, df_pred


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print(f"Using TRAIN_YEARS={TRAIN_YEARS}")
    print(f"Using VAL_YEAR={VAL_YEAR}")

    meta = list_parquets()
    if not meta:
        print(" No regression parquet shards found.")
        return

    all_metrics = []
    all_preds = []
    filtered_metrics_month = []

    for m in range(1, 13):
        print(f"\n==== Month {m:02d} ====")

        if month_has_trained_preds(m):
            reused = load_existing_month(m)
            if reused is None:
                print(f"   Month {m:02d}: expected existing preds but could not load.")
                continue
            g, df_month = reused

            station_metrics_all = station_regression_metrics(df_month)
            gf = compute_and_save_filtered_metrics(m, df_month, station_metrics_all)
            if gf is not None:
                filtered_metrics_month.append(gf)
        else:
            result = train_month(m, meta)
            if result is None:
                continue
            g, df_month = result

            station_metrics_all = station_regression_metrics(df_month)
            gf = compute_and_save_filtered_metrics(m, df_month, station_metrics_all)
            if gf is not None:
                filtered_metrics_month.append(gf)

        if "month" not in df_month.columns:
            df_month["month"] = m

        all_metrics.append(g)
        all_preds.append(df_month)

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics).sort_values("month").reset_index(drop=True)
        save_csv(df_metrics, f"{RESULTS_PREFIX}/all_metrics_regression.csv")

    if filtered_metrics_month:
        df_metrics_f = pd.DataFrame(filtered_metrics_month).sort_values("month").reset_index(drop=True)
        save_csv(df_metrics_f, f"{RESULTS_PREFIX_FILTERED}/all_metrics_regression_filtered.csv")

    if all_preds:
        df_all = pd.concat(all_preds, ignore_index=True)
        df_all["station_id"] = df_all["station_id"].astype(str).str.strip().str.zfill(5)
        df_all["time"] = pd.to_datetime(df_all["time"], utc=True, errors="coerce")
        df_all = df_all.sort_values(["station_id", "time"]).reset_index(drop=True)

        save_parquet(df_all, f"{RESULTS_PREFIX}/preds_{VAL_YEAR}_all_months_regression.parquet")
        save_csv(df_all, f"{RESULTS_PREFIX}/preds_{VAL_YEAR}_all_months_regression.csv")

        st_all = station_regression_metrics(df_all)
        save_csv(st_all, f"{RESULTS_PREFIX}/station_metrics_regression_all_months.csv")

        good_station_ids = (
            st_all.loc[st_all["n_rainy"] >= MIN_RAINY_MIN_PER_MONTH, "station_id"]
            .astype(str)
            .tolist()
        )

        comp = st_all.copy()
        comp["included"] = comp["station_id"].isin(good_station_ids).astype(int)
        save_csv(comp, f"{RESULTS_PREFIX_FILTERED}/station_comparison_all_months.csv")

        if good_station_ids:
            df_all_f = df_all[df_all["station_id"].isin(good_station_ids)].copy()
            g_all_f = eval_regression(df_all_f["y_true_mm"].values, df_all_f["y_pred_mm"].values)
            g_all_f.update(
                {
                    "filtered": 1,
                    "min_rainy_min_per_month": MIN_RAINY_MIN_PER_MONTH,
                    "n_stations": len(good_station_ids),
                }
            )
            save_csv(
                pd.DataFrame([g_all_f]),
                f"{RESULTS_PREFIX_FILTERED}/metrics_global_all_months_filtered.csv",
            )
            save_parquet(
                df_all_f,
                f"{RESULTS_PREFIX_FILTERED}/preds_{VAL_YEAR}_all_months_regression_filtered.parquet",
            )
            save_csv(
                df_all_f,
                f"{RESULTS_PREFIX_FILTERED}/preds_{VAL_YEAR}_all_months_regression_filtered.csv",
            )

    print("\n Station-only 2025 regression benchmark completed.")


if __name__ == "__main__":
    main()