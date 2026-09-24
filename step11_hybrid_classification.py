#!/usr/bin/env python3

import json
from io import BytesIO
from typing import Dict, List, Tuple

import boto3
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# =========================================================
# CONFIG
# =========================================================
REGION = "eu-north-1"
s3 = boto3.client("s3", region_name=REGION)

# Step 10 output bucket/prefix pattern
BUCKET = "tica93-dmi-all-stations-regresion"
INPUT_SUFFIX = "_preproc_1min_radar_pysteps"

# IMPORTANT:
# We only have 2025 data in your current Step 10.5 output,
# so we train on earlier 2025 months and validate on the target 2025 month.
TRAIN_YEARS = [2025]
VAL_YEAR = 2025
TARGET = "y_flag_t+1"

RESULTS_PREFIX = "model_outputs_hybrid_step11"

# filtered evaluation threshold
MIN_POS_PER_MONTH = 50

# optional: require PySTEPS columns to be present for pysteps-based configs
REQUIRE_NONEMPTY_PYSTEPS_FOR_PYSTEPS_MODELS = True


# =========================================================
# S3 HELPERS
# =========================================================
def read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def save_csv(df: pd.DataFrame, path: str) -> None:
    s3.put_object(
        Bucket=BUCKET,
        Key=path,
        Body=df.to_csv(index=False).encode("utf-8"),
        ContentType="text/csv",
    )
    print(f"  💾 Saved CSV → s3://{BUCKET}/{path}   (rows={len(df):,})")


def save_parquet(df: pd.DataFrame, path: str) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3.put_object(
        Bucket=BUCKET,
        Key=path,
        Body=buf.read(),
        ContentType="application/octet-stream",
    )
    print(f"  💾 Saved Parquet → s3://{BUCKET}/{path}   (rows={len(df):,})")


# =========================================================
# DISCOVERY
# =========================================================
def list_input_parquets() -> List[dict]:
    out = []
    token = None

    while True:
        kw = {"Bucket": BUCKET}
        if token:
            kw["ContinuationToken"] = token

        r = s3.list_objects_v2(**kw)
        for obj in r.get("Contents", []):
            k = obj["Key"]

            if INPUT_SUFFIX not in k:
                continue
            if not k.endswith(".parquet"):
                continue

            # IMPORTANT: skip derived model-output parquet shards
            if "__lightgbm_base" in k:
                continue

            parts = k.split("/")
            # expected like:
            # 05277_preproc_1min_radar_pysteps/year=2025/month=09/05277_preproc_2025-09.parquet
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
                    "key": k,
                    "station_id": station_id,
                    "year": year,
                    "month": month,
                }
            )

        if r.get("IsTruncated"):
            token = r["NextContinuationToken"]
        else:
            break

    return out


# =========================================================
# FEATURE DISCOVERY / PREP
# =========================================================
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    if "station_id" in df.columns:
        s = df["station_id"].astype(str).str.strip()
        s = s.str.replace(r"\.0$", "", regex=True)
        s = s.str.replace(r"\D+", "", regex=True)
        df["station_id"] = s.str.zfill(5)

    # Make sure target exists
    if TARGET not in df.columns:
        if "y_mm_t+1" in df.columns:
            df[TARGET] = (pd.to_numeric(df["y_mm_t+1"], errors="coerce").fillna(0) > 0).astype("int8")
        else:
            raise RuntimeError(f"Missing target column: {TARGET}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype("int8")

    # basic numeric coercion
    for c in df.columns:
        if c in ["time", "station_id"]:
            continue
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass

    return df


def build_feature_sets(example_df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = set(example_df.columns)

    base_candidates = [
        "humidity",
        "temp",
        "cloud_cover",
        "wind_speed",
        "wind_dir_sin",
        "wind_dir_cos",
        "precip_past1min",
        "precip_flag_1min",
        "gap_after_prev",
        "hour",
        "dow",
        "month",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]

    lag_roll_candidates = sorted(
        c for c in cols
        if (
            "lag" in c.lower()
            or "roll" in c.lower()
            or "rolling" in c.lower()
            or c.startswith("precip_")
            or c.startswith("temp_")
            or c.startswith("humidity_")
            or c.startswith("wind_")
        )
    )

    radar_candidates = sorted(
        c for c in cols
        if c.startswith("radar_") or "dbzh" in c.lower()
    )

    pysteps_candidates = sorted(
        c for c in cols
        if c.startswith("pysteps_")
    )

    # remove non-feature columns if they slipped in
    forbidden = {
        "time",
        "station_id",
        TARGET,
        "y_mm_t+1",
        "y_flag_t+5",
        "y_mm_t+5",
        "y_flag_t+10",
        "y_mm_t+10",
    }

    def clean_feature_list(lst: List[str]) -> List[str]:
        out = []
        seen = set()
        for c in lst:
            if c in forbidden:
                continue
            if c not in example_df.columns:
                continue
            if c in seen:
                continue
            # keep only numeric or boolean-ish columns
            if pd.api.types.is_numeric_dtype(example_df[c]) or pd.api.types.is_bool_dtype(example_df[c]):
                out.append(c)
                seen.add(c)
        return out

    base = clean_feature_list(base_candidates)
    lag_roll = clean_feature_list(lag_roll_candidates)
    radar = clean_feature_list(radar_candidates)
    pysteps = clean_feature_list(pysteps_candidates)

    feature_sets = {
        "base_only": clean_feature_list(base),
        "base_plus_lags": clean_feature_list(base + lag_roll),
        "base_plus_radar": clean_feature_list(base + lag_roll + radar),
        "base_plus_pysteps": clean_feature_list(base + lag_roll + pysteps),
        "hybrid_all": clean_feature_list(base + lag_roll + radar + pysteps),
    }

    print("\n🔎 Feature set summary:")
    for k, v in feature_sets.items():
        print(f"  {k}: {len(v)} features")

    return feature_sets


def select_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_cols].copy()

    # force numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype("int8").values
    return X, y


# =========================================================
# METRICS
# =========================================================
def eval_global(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "threshold": float(thr),
        "rows": int(len(y_true)),
        "n_pos": int(np.sum(y_true)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc) if pd.notna(pr_auc) else np.nan,
        "pod": float(pod),
        "far": float(far),
        "csi": float(csi),
    }


def compute_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5

    thresholds = np.linspace(0.05, 0.95, 37)
    best_thr = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)

    return best_thr


def station_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sid, g in df.groupby("station_id"):
        y = g["y_true_flag"].values.astype(int)
        yhat = g["y_pred_flag"].values.astype(int)

        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        n_pos = tp + fn

        rows.append(
            {
                "station_id": sid,
                "rows": len(g),
                "n_pos": int(n_pos),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": float(precision_score(y, yhat, zero_division=0)),
                "recall": float(recall_score(y, yhat, zero_division=0)),
                "f1": float(f1_score(y, yhat, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# DATA ASSEMBLY
# =========================================================
def load_month_dataset(meta: List[dict], years: List[int], month: int) -> pd.DataFrame:
    keys = [m["key"] for m in meta if m["year"] in years and m["month"] == month]
    if not keys:
        return pd.DataFrame()

    parts = []
    for k in sorted(keys):
        print(f"  Loading {k}")
        try:
            df = read_parquet_s3(BUCKET, k)
            df = prepare_df(df)
            parts.append(df)
        except Exception as e:
            print(f"  Failed to load {k}: {e}")

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["station_id", "time"], keep="first").reset_index(drop=True)
    return df


def load_month_dataset_up_to(meta: List[dict], year: int, max_month_exclusive: int) -> pd.DataFrame:
    """
    Load all shards from one year for months strictly earlier than target month.
    Example: for target month 10, load months 06-09 if available.
    """
    keys = [
        m["key"]
        for m in meta
        if m["year"] == year and m["month"] < max_month_exclusive
    ]
    if not keys:
        return pd.DataFrame()

    parts = []
    for k in sorted(keys):
        print(f"   Loading train shard {k}")
        try:
            df = read_parquet_s3(BUCKET, k)
            df = prepare_df(df)
            parts.append(df)
        except Exception as e:
            print(f"   Failed to load {k}: {e}")

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["station_id", "time"], keep="first").reset_index(drop=True)
    return df


# =========================================================
# TRAIN
# =========================================================
def train_one_config(
    model_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    month: int,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    print(f"\n  ---- Training config: {model_name}")
    print(f"  Features: {len(feature_cols)}")

    if not feature_cols:
        raise RuntimeError(f"No features available for config: {model_name}")

    # optional guard for pysteps configs
    if REQUIRE_NONEMPTY_PYSTEPS_FOR_PYSTEPS_MODELS and "pysteps" in model_name:
        pcols = [c for c in feature_cols if c.startswith("pysteps_")]
        if not pcols:
            raise RuntimeError(f"No PySTEPS feature columns found for config: {model_name}")

    X_train, y_train = select_xy(train_df, feature_cols)
    X_val, y_val = select_xy(val_df, feature_cols)

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training target has only one class.")
    if len(np.unique(y_val)) < 2:
        raise RuntimeError("Validation target has only one class.")

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    p_val = model.predict(X_val, num_iteration=model.best_iteration)
    thr = compute_best_threshold(y_val, p_val)
    yhat = (p_val >= thr).astype(int)

    pred_df = val_df[["time", "station_id"]].copy()
    pred_df["month"] = month
    pred_df["model_name"] = model_name
    pred_df["y_true_flag"] = y_val
    pred_df["proba"] = p_val
    pred_df["y_pred_flag"] = yhat

    # helpful diagnostics
    if "pysteps_time" in val_df.columns:
        pred_df["has_pysteps"] = val_df["pysteps_time"].notna().astype(int).values
    if "pysteps_age_min" in val_df.columns:
        pred_df["pysteps_age_min"] = pd.to_numeric(val_df["pysteps_age_min"], errors="coerce").values

    metrics = eval_global(y_val, yhat, p_val, thr)
    metrics.update(
        {
            "month": month,
            "model_name": model_name,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_features": int(len(feature_cols)),
            "n_pos_train": int(y_train.sum()),
            "n_pos_val": int(y_val.sum()),
        }
    )

    # feature importance
    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
            "month": month,
            "model_name": model_name,
        }
    ).sort_values("importance_gain", ascending=False)

    return metrics, pred_df, fi


# =========================================================
# MONTH LOOP
# =========================================================
def run_month(month: int, meta: List[dict]) -> None:
    print("\n========================================================")
    print(f"STEP 11 HYBRID CLASSIFICATION — MONTH {month:02d}")
    print("========================================================")

    # If train and validation year are the same, train on all earlier months.
    if VAL_YEAR in TRAIN_YEARS:
        train_df = load_month_dataset_up_to(meta, VAL_YEAR, month)
    else:
        train_df = load_month_dataset(meta, TRAIN_YEARS, month)

    val_df = load_month_dataset(meta, [VAL_YEAR], month)

    if train_df.empty:
        print("  No training data for this month")
        return
    if val_df.empty:
        print("  No validation data for this month")
        return

    print(f"  Train rows: {len(train_df):,}")
    print(f"  Val rows:   {len(val_df):,}")

    feature_sets = build_feature_sets(
        pd.concat([train_df.head(1000), val_df.head(1000)], ignore_index=True)
    )

    all_metrics = []
    all_station_metrics = []
    all_preds = []
    all_fi = []

    for model_name, feature_cols in feature_sets.items():
        try:
            metrics, pred_df, fi_df = train_one_config(
                model_name, feature_cols, train_df, val_df, month
            )

            st = station_metrics(pred_df)
            st["month"] = month
            st["model_name"] = model_name

            # filtered version
            st_good = st.loc[st["n_pos"] >= MIN_POS_PER_MONTH].copy()
            if not st_good.empty:
                keep_ids = set(st_good["station_id"])
                pred_f = pred_df[pred_df["station_id"].isin(keep_ids)].copy()

                m_f = eval_global(
                    pred_f["y_true_flag"].values.astype(int),
                    pred_f["y_pred_flag"].values.astype(int),
                    pred_f["proba"].values.astype(float),
                    metrics["threshold"],
                )
                m_f.update(
                    {
                        "month": month,
                        "model_name": model_name,
                        "subset": "filtered",
                        "stations_kept": int(len(keep_ids)),
                    }
                )
                all_metrics.append(pd.DataFrame([m_f]))

            metrics["subset"] = "all"
            all_metrics.append(pd.DataFrame([metrics]))
            all_station_metrics.append(st)
            all_preds.append(pred_df)
            all_fi.append(fi_df)

        except Exception as e:
            print(f"  Failed config {model_name}: {e}")

    if not all_metrics:
        print("  No models completed for this month")
        return

    df_metrics = pd.concat(all_metrics, ignore_index=True)
    df_station = pd.concat(all_station_metrics, ignore_index=True) if all_station_metrics else pd.DataFrame()
    df_preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    df_fi = pd.concat(all_fi, ignore_index=True) if all_fi else pd.DataFrame()

    mp = f"{RESULTS_PREFIX}/month={month:02d}"

    save_csv(df_metrics, f"{mp}/metrics_global.csv")
    if not df_station.empty:
        save_csv(df_station, f"{mp}/metrics_station.csv")
    if not df_preds.empty:
        save_parquet(df_preds, f"{mp}/preds_{VAL_YEAR}.parquet")
    if not df_fi.empty:
        save_csv(df_fi, f"{mp}/feature_importance.csv")


# =========================================================
# MAIN
# =========================================================
def main():
    meta = list_input_parquets()
    print(f" Found {len(meta):,} Step 10 merged parquet shards")

    if not meta:
        print(" No Step 10 merged input files found.")
        return

    months = sorted(set(m["month"] for m in meta))
    print(f"Months found: {months}")

    summary_rows = []
    for month in months:
        run_month(month, meta)

        # try reading global metrics back for summary
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=f"{RESULTS_PREFIX}/month={month:02d}/metrics_global.csv")
            g = pd.read_csv(BytesIO(obj["Body"].read()))
            summary_rows.append(g)
        except Exception:
            pass

    if summary_rows:
        df_summary = pd.concat(summary_rows, ignore_index=True)
        save_csv(df_summary, f"{RESULTS_PREFIX}/summary_all_months.csv")

        best = (
            df_summary[df_summary["subset"] == "all"]
            .sort_values(["month", "pr_auc"], ascending=[True, False])
            .groupby("month")
            .head(1)
            .reset_index(drop=True)
        )
        save_csv(best, f"{RESULTS_PREFIX}/best_model_per_month.csv")

    print("\n Step 11 finished — hybrid classification benchmark completed.")


if __name__ == "__main__":
    main()