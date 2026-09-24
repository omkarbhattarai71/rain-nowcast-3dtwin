#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# CONFIG
# =========================================================
REGION = "eu-north-1"
BUCKET = "tica93-dmi-all-stations-regresion"
STEP5_FILE_TEMPLATE = "model_outputs_regression_2025_filtered/month={month:02d}/preds_2025_regression_filtered.parquet"

OUTPUT_DIR = Path(
    r"C:\Users\Tijana Devaja\PycharmProjects\PythonProject1Tijana\results\step13_ssm_hybrid_improved"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- basic SSM ----------
DT = 1.0
PROCESS_NOISE_Q = 0.005
OBSERVATION_NOISE_R = 0.01
INITIAL_STATE_COV = 1.0

# ---------- residual SSM ----------
RESIDUAL_AR = 0.90
RESIDUAL_PROCESS_NOISE_Q = 0.0005
RESIDUAL_OBSERVATION_NOISE_R = 0.002
RESIDUAL_INITIAL_STATE_COV = 1.0

# ---------- residual lag benchmark ----------
RESIDUAL_LAGS = [1, 2, 3, 5, 10, 15]

# ---------- weighted fusion search ----------
FUSION_ALPHAS = [0.70, 0.75, 0.80, 0.85, 0.90]

# ---------- rainfall evaluation ----------
RAIN_THRESHOLD_MM = 0.1
INTENSITY_BINS = [
    ("zero", -np.inf, 0.0),
    ("light", 0.0, 0.1),
    ("moderate", 0.1, 1.0),
    ("heavy", 1.0, np.inf),
]

# ---------- attenuation placeholders ----------
ITU_K = 0.15
ITU_ALPHA = 1.0
LINK_LENGTH_KM = 1.0

# ---------- plotting ----------
DPI = 300
N_PLOT_POINTS = 500
MAX_SCATTER_POINTS = 100000

s3 = boto3.client("s3", region_name=REGION)


# =========================================================
# METRICS
# =========================================================
@dataclass
class ModelMetrics:
    model: str
    rmse: float
    mae: float
    r2: float
    bias: float
    rmse_rain_only: float
    mae_rain_only: float
    r2_rain_only: float


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def rain_mask(y_true: np.ndarray, thr: float = RAIN_THRESHOLD_MM) -> np.ndarray:
    return np.asarray(y_true, dtype=float) > thr


def skill_score_rmse(model_rmse: float, persistence_rmse: float) -> float:
    if persistence_rmse == 0 or not np.isfinite(persistence_rmse):
        return np.nan
    return float(1.0 - (model_rmse / persistence_rmse))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.allclose(np.nanstd(y_true), 0.0):
        return np.nan
    return float(r2_score(y_true, y_pred))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> ModelMetrics:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]

    if len(y_true_v) == 0:
        return ModelMetrics(model_name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    mask_r = rain_mask(y_true_v)
    if mask_r.sum() >= 2:
        y_true_r = y_true_v[mask_r]
        y_pred_r = y_pred_v[mask_r]
        rmse_r = rmse(y_true_r, y_pred_r)
        mae_r = float(mean_absolute_error(y_true_r, y_pred_r))
        r2_r = safe_r2(y_true_r, y_pred_r)
    else:
        rmse_r = np.nan
        mae_r = np.nan
        r2_r = np.nan

    return ModelMetrics(
        model=model_name,
        rmse=rmse(y_true_v, y_pred_v),
        mae=float(mean_absolute_error(y_true_v, y_pred_v)),
        r2=safe_r2(y_true_v, y_pred_v),
        bias=bias(y_true_v, y_pred_v),
        rmse_rain_only=rmse_r,
        mae_rain_only=mae_r,
        r2_rain_only=r2_r,
    )


# =========================================================
# LOAD FILTERED STEP 5 MONTHLY OUTPUTS FROM S3
# =========================================================
def load_step5_predictions_from_s3() -> pd.DataFrame:
    monthly_dfs: List[pd.DataFrame] = []

    print("🔍 Loading filtered Step 5 monthly regression predictions from S3...")

    for m in range(1, 13):
        key = STEP5_FILE_TEMPLATE.format(month=m)
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            df_m = pd.read_parquet(BytesIO(obj["Body"].read()))
            print(f"  Loaded month {m:02d}: {len(df_m):,} rows")
            if "month" not in df_m.columns:
                df_m["month"] = m
            monthly_dfs.append(df_m)
        except Exception as e:
            print(f"  Month {m:02d} not available: {key} ({e})")

    if not monthly_dfs:
        raise ValueError("No filtered Step 5 monthly regression outputs were found in S3.")

    df = pd.concat(monthly_dfs, ignore_index=True)

    required = {"station_id", "time", "y_true_mm", "y_pred_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Loaded Step 5 predictions are missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["station_id"] = df["station_id"].astype(str).str.strip().str.zfill(5)
    df["y_true_mm"] = pd.to_numeric(df["y_true_mm"], errors="coerce")
    df["y_pred_mm"] = pd.to_numeric(df["y_pred_mm"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")

    df = df.dropna(subset=["time", "station_id", "y_true_mm", "y_pred_mm"])
    df = df.sort_values(["station_id", "time"]).reset_index(drop=True)

    df = df.rename(columns={
        "y_true_mm": "rain_obs",
        "y_pred_mm": "ml_pred",
    })

    print(
        f" Combined filtered Step 5 data loaded: {len(df):,} rows across "
        f"{df['station_id'].nunique()} stations"
    )
    return df


# =========================================================
# BASELINES
# =========================================================
def add_persistence_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["persistence_pred"] = out.groupby("station_id")["rain_obs"].shift(1).fillna(0.0)
    return out


# =========================================================
# BASIC SSM
# =========================================================
class RainfallStateSpaceModel:
    def __init__(
        self,
        dt: float = 1.0,
        process_noise_q: float = 0.005,
        observation_noise_r: float = 0.01,
        initial_state_cov: float = 1.0,
    ) -> None:
        self.A = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        self.H = np.array([[1.0, 0.0]], dtype=float)
        self.Q = process_noise_q * np.eye(2, dtype=float)
        self.R = np.array([[observation_noise_r]], dtype=float)
        self.x = np.zeros((2, 1), dtype=float)
        self.P = initial_state_cov * np.eye(2, dtype=float)

    def predict(self) -> None:
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y_obs: float) -> None:
        y = np.array([[float(y_obs)]], dtype=float)
        innovation = y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(2) - K @ self.H) @ self.P

    def step(self, y_obs: float) -> Dict[str, float]:
        self.predict()
        prior_forecast = float(self.x[0, 0])
        self.update(y_obs)
        posterior_estimate = float(self.x[0, 0])
        trend_estimate = float(self.x[1, 0])
        return {
            "prior_forecast": prior_forecast,
            "posterior_estimate": posterior_estimate,
            "trend_estimate": trend_estimate,
        }


# =========================================================
# RESIDUAL SSM
# =========================================================
class ResidualStateSpaceModel:
    def __init__(
        self,
        a: float = 0.90,
        process_noise_q: float = 0.0005,
        observation_noise_r: float = 0.002,
        initial_state_cov: float = 1.0,
    ) -> None:
        self.A = np.array([[a]], dtype=float)
        self.H = np.array([[1.0]], dtype=float)
        self.Q = np.array([[process_noise_q]], dtype=float)
        self.R = np.array([[observation_noise_r]], dtype=float)
        self.x = np.zeros((1, 1), dtype=float)
        self.P = initial_state_cov * np.eye(1, dtype=float)

    def predict(self) -> None:
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, residual_obs: float) -> None:
        y = np.array([[float(residual_obs)]], dtype=float)
        innovation = y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(1) - K @ self.H) @ self.P

    def step(self, residual_obs: float) -> Dict[str, float]:
        self.predict()
        prior_forecast = float(self.x[0, 0])
        self.update(residual_obs)
        posterior_estimate = float(self.x[0, 0])
        return {
            "prior_forecast": prior_forecast,
            "posterior_estimate": posterior_estimate,
        }


# =========================================================
# RUN MODELS PER STATION
# =========================================================
def run_basic_ssm_per_station(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    n_stations = df["station_id"].nunique()

    for i, (station_id, g) in enumerate(df.groupby("station_id", sort=False), start=1):
        print(f"⚙️ Basic SSM for station {station_id} ({i}/{n_stations})")
        g = g.sort_values("time").reset_index(drop=True).copy()

        model = RainfallStateSpaceModel(
            dt=DT,
            process_noise_q=PROCESS_NOISE_Q,
            observation_noise_r=OBSERVATION_NOISE_R,
            initial_state_cov=INITIAL_STATE_COV,
        )

        basic_forecasts = []
        basic_filtered = []
        basic_trend = []

        for y_obs in g["rain_obs"].values:
            y_obs = max(float(y_obs), 0.0)
            res = model.step(y_obs)
            basic_forecasts.append(res["prior_forecast"])
            basic_filtered.append(res["posterior_estimate"])
            basic_trend.append(res["trend_estimate"])

        g["ssm_basic_forecast"] = np.clip(basic_forecasts, 0.0, None)
        g["ssm_basic_filtered"] = np.clip(basic_filtered, 0.0, None)
        g["ssm_basic_trend"] = basic_trend
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["station_id", "time"]).reset_index(drop=True)
    return out


def run_residual_ssm_per_station(df: pd.DataFrame) -> pd.DataFrame:
    if "ml_pred" not in df.columns:
        raise ValueError("ml_pred column is required for residual SSM hybrid.")

    parts = []
    n_stations = df["station_id"].nunique()

    for i, (station_id, g) in enumerate(df.groupby("station_id", sort=False), start=1):
        print(f"🔧 Residual SSM for station {station_id} ({i}/{n_stations})")
        g = g.sort_values("time").reset_index(drop=True).copy()

        model = ResidualStateSpaceModel(
            a=RESIDUAL_AR,
            process_noise_q=RESIDUAL_PROCESS_NOISE_Q,
            observation_noise_r=RESIDUAL_OBSERVATION_NOISE_R,
            initial_state_cov=RESIDUAL_INITIAL_STATE_COV,
        )

        residual_forecasts = []
        residual_filtered = []
        hybrid_preds = []
        residual_series = g["rain_obs"].values - g["ml_pred"].values

        for ml_p, e_obs in zip(g["ml_pred"].values, residual_series):
            res = model.step(float(e_obs))
            e_fore = res["prior_forecast"]
            e_filt = res["posterior_estimate"]
            residual_forecasts.append(e_fore)
            residual_filtered.append(e_filt)
            hybrid_pred = float(ml_p) + e_fore
            hybrid_preds.append(max(hybrid_pred, 0.0))

        g["residual_ssm_forecast"] = residual_forecasts
        g["residual_ssm_filtered"] = residual_filtered
        g["hybrid_residual_ssm_pred"] = hybrid_preds
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["station_id", "time"]).reset_index(drop=True)
    return out


# =========================================================
# RESIDUAL LAG BENCHMARK
# =========================================================
def run_residual_lag_benchmark(df: pd.DataFrame, residual_lags: List[int]) -> pd.DataFrame:
    parts = []
    n_stations = df["station_id"].nunique()

    for i, (station_id, g) in enumerate(df.groupby("station_id", sort=False), start=1):
        print(f"📐 Residual-lag benchmark for station {station_id} ({i}/{n_stations})")
        g = g.sort_values("time").reset_index(drop=True).copy()

        g["resid_obs"] = g["rain_obs"] - g["ml_pred"]

        for lag in residual_lags:
            g[f"resid_lag_{lag}"] = g["resid_obs"].shift(lag)

        g["ml_pred_current"] = g["ml_pred"].astype(float)
        g["is_rain_ml"] = (g["ml_pred"] > RAIN_THRESHOLD_MM).astype(int)
        g["ml_pred_x_rainflag"] = g["ml_pred_current"] * g["is_rain_ml"]

        feature_cols = [f"resid_lag_{lag}" for lag in residual_lags] + [
            "ml_pred_current",
            "is_rain_ml",
            "ml_pred_x_rainflag",
        ]

        fit_df = g.dropna(subset=feature_cols + ["resid_obs"]).copy()

        g["hybrid_residual_lag_pred"] = g["ml_pred"].clip(lower=0.0)
        g["residual_lag_forecast"] = np.nan

        min_rows = max(30, len(feature_cols) + 10)

        if len(fit_df) >= min_rows:
            X = fit_df[feature_cols].values
            y = fit_df["resid_obs"].values
            reg = LinearRegression()
            reg.fit(X, y)
            pred_resid = reg.predict(X)
            fit_df["resid_hat"] = pred_resid

            g.loc[fit_df.index, "residual_lag_forecast"] = fit_df["resid_hat"].values
            g.loc[fit_df.index, "hybrid_residual_lag_pred"] = np.clip(
                g.loc[fit_df.index, "ml_pred"].values + fit_df["resid_hat"].values,
                0.0,
                None,
            )

        drop_cols = ["resid_obs"] + [f"resid_lag_{lag}" for lag in residual_lags]
        g = g.drop(columns=[c for c in drop_cols if c in g.columns])
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["station_id", "time"]).reset_index(drop=True)
    return out


# =========================================================
# REGIME SWITCH
# =========================================================
def add_regime_switch_hybrid(df: pd.DataFrame, rain_threshold: float = RAIN_THRESHOLD_MM) -> pd.DataFrame:
    out = df.copy()

    required_cols = ["ml_pred", "hybrid_residual_ssm_pred", "hybrid_residual_lag_pred"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for regime switch hybrid: {missing}")

    rain_mask_local = out["ml_pred"] > rain_threshold
    out["hybrid_regime_switch"] = np.where(
        rain_mask_local,
        out["hybrid_residual_ssm_pred"],
        out["hybrid_residual_lag_pred"],
    )
    out["hybrid_regime_switch"] = out["hybrid_regime_switch"].clip(lower=0.0)
    return out


# =========================================================
# FUSION
# =========================================================
def add_weighted_fusions(df: pd.DataFrame, alphas: List[float]) -> pd.DataFrame:
    out = df.copy()
    for alpha in alphas:
        col = f"hybrid_weighted_pred_{int(round(alpha * 100)):03d}"
        out[col] = (alpha * out["ml_pred"] + (1.0 - alpha) * out["ssm_basic_forecast"]).clip(lower=0.0)
    return out


# =========================================================
# MODEL COLUMN REGISTRY
# =========================================================
def get_model_cols(df: pd.DataFrame) -> List[Tuple[str, str]]:
    model_cols = [
        ("Persistence", "persistence_pred"),
        ("StateSpace_Basic", "ssm_basic_forecast"),
        ("ML_Step5", "ml_pred"),
        ("Hybrid_ResidualSSM", "hybrid_residual_ssm_pred"),
    ]

    if "hybrid_residual_lag_pred" in df.columns:
        model_cols.append(("Hybrid_ResidualLag", "hybrid_residual_lag_pred"))

    if "hybrid_regime_switch" in df.columns:
        model_cols.append(("Hybrid_RegimeSwitch", "hybrid_regime_switch"))

    for alpha in FUSION_ALPHAS:
        col = f"hybrid_weighted_pred_{int(round(alpha * 100)):03d}"
        if col in df.columns:
            model_cols.append((f"Hybrid_WeightedFusion_{alpha:.2f}", col))

    return model_cols


# =========================================================
# EVALUATION
# =========================================================
def evaluate_all_models(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["rain_obs"].to_numpy(dtype=float)
    model_cols = get_model_cols(df)

    metrics = []
    for model_name, pred_col in model_cols:
        metrics.append(evaluate_predictions(y_true, df[pred_col].to_numpy(dtype=float), model_name))

    metrics_df = pd.DataFrame([m.__dict__ for m in metrics])

    if "Persistence" in metrics_df["model"].values:
        persistence_rmse = float(metrics_df.loc[metrics_df["model"] == "Persistence", "rmse"].iloc[0])
        metrics_df["skill_vs_persistence"] = metrics_df["rmse"].apply(lambda x: skill_score_rmse(x, persistence_rmse))
    else:
        metrics_df["skill_vs_persistence"] = np.nan

    metrics_df = metrics_df.sort_values("rmse").reset_index(drop=True)
    return metrics_df


def evaluate_by_month(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_cols = get_model_cols(df)

    for month, g in df.groupby("month"):
        y_true = g["rain_obs"].to_numpy(dtype=float)
        for model_name, pred_col in model_cols:
            m = evaluate_predictions(y_true, g[pred_col].to_numpy(dtype=float), model_name)
            rows.append({
                "month": int(month),
                "model": m.model,
                "rmse": m.rmse,
                "mae": m.mae,
                "r2": m.r2,
                "bias": m.bias,
                "rmse_rain_only": m.rmse_rain_only,
                "mae_rain_only": m.mae_rain_only,
                "r2_rain_only": m.r2_rain_only,
                "rows": len(g),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["month", "rmse"]).reset_index(drop=True)
    return out


def evaluate_by_station(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_cols = get_model_cols(df)

    for station_id, g in df.groupby("station_id"):
        y_true = g["rain_obs"].to_numpy(dtype=float)
        for model_name, pred_col in model_cols:
            m = evaluate_predictions(y_true, g[pred_col].to_numpy(dtype=float), model_name)
            rows.append({
                "station_id": station_id,
                "model": m.model,
                "rmse": m.rmse,
                "mae": m.mae,
                "r2": m.r2,
                "bias": m.bias,
                "rmse_rain_only": m.rmse_rain_only,
                "mae_rain_only": m.mae_rain_only,
                "r2_rain_only": m.r2_rain_only,
                "rows": len(g),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["station_id", "rmse"]).reset_index(drop=True)
    return out


def evaluate_rain_only(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["rain_obs"].to_numpy(dtype=float) > RAIN_THRESHOLD_MM
    if mask.sum() == 0:
        return pd.DataFrame()

    y_true = df.loc[mask, "rain_obs"].to_numpy(dtype=float)
    rows = []
    for model_name, pred_col in get_model_cols(df):
        y_pred = df.loc[mask, pred_col].to_numpy(dtype=float)
        m = evaluate_predictions(y_true, y_pred, model_name)
        rows.append({
            "model": m.model,
            "rmse": m.rmse,
            "mae": m.mae,
            "r2": m.r2,
            "bias": m.bias,
            "rmse_rain_only": m.rmse_rain_only,
            "mae_rain_only": m.mae_rain_only,
            "r2_rain_only": m.r2_rain_only,
            "n_rows": int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def evaluate_by_intensity_bin(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_cols = get_model_cols(df)
    y = df["rain_obs"].to_numpy(dtype=float)

    for subset_name, lo, hi in INTENSITY_BINS:
        mask = (y > lo) & (y <= hi)
        if mask.sum() < 2:
            continue
        y_true = y[mask]
        for model_name, pred_col in model_cols:
            y_pred = df.loc[mask, pred_col].to_numpy(dtype=float)
            m = evaluate_predictions(y_true, y_pred, model_name)
            rows.append({
                "subset": subset_name,
                "n_rows": int(mask.sum()),
                "model": m.model,
                "rmse": m.rmse,
                "mae": m.mae,
                "r2": m.r2,
                "bias": m.bias,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["subset", "rmse"]).reset_index(drop=True)
    return out


# =========================================================
# RESIDUAL ACF
# =========================================================
def autocorr(x: np.ndarray, lag: int) -> float:
    if lag >= len(x):
        return np.nan
    x0 = x[:-lag] if lag > 0 else x
    x1 = x[lag:] if lag > 0 else x
    if np.nanstd(x0) == 0 or np.nanstd(x1) == 0:
        return np.nan
    return float(np.corrcoef(x0, x1)[0, 1])


def compute_stationwise_residual_acf(df: pd.DataFrame, pred_col: str, max_lag: int = 60) -> pd.DataFrame:
    lags = list(range(1, max_lag + 1))
    vals = []

    for lag in lags:
        acs = []
        for _, g in df.groupby("station_id"):
            g = g.sort_values("time")
            resid = g["rain_obs"].to_numpy(dtype=float) - g[pred_col].to_numpy(dtype=float)
            val = autocorr(resid, lag)
            if np.isfinite(val):
                acs.append(val)
        vals.append({"lag": lag, "acf": float(np.mean(acs)) if acs else np.nan, "pred_col": pred_col})

    return pd.DataFrame(vals)


# =========================================================
# ATTENUATION
# =========================================================
def compute_specific_attenuation(rain_rate_mm: pd.Series, k: float, alpha: float) -> pd.Series:
    rain = pd.to_numeric(rain_rate_mm, errors="coerce").fillna(0.0).clip(lower=0.0)
    return k * np.power(rain, alpha)


def add_attenuation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rain_cols = [
        "rain_obs",
        "persistence_pred",
        "ssm_basic_forecast",
        "ml_pred",
        "hybrid_residual_ssm_pred",
    ]

    if "hybrid_residual_lag_pred" in out.columns:
        rain_cols.append("hybrid_residual_lag_pred")
    if "hybrid_regime_switch" in out.columns:
        rain_cols.append("hybrid_regime_switch")

    for alpha in FUSION_ALPHAS:
        col = f"hybrid_weighted_pred_{int(round(alpha * 100)):03d}"
        if col in out.columns:
            rain_cols.append(col)

    for rain_col in rain_cols:
        gamma = compute_specific_attenuation(out[rain_col], k=ITU_K, alpha=ITU_ALPHA)
        out[f"{rain_col}_attenuation_db"] = gamma * LINK_LENGTH_KM

    return out


# =========================================================
# PLOTS
# =========================================================
def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_time_series_comparison(df: pd.DataFrame, out_path: Path, n_points: int = N_PLOT_POINTS) -> None:
    if df.empty:
        return

    station_totals = df.groupby("station_id")["rain_obs"].sum().sort_values(ascending=False)
    station_id = station_totals.index[0]
    plot_df = df[df["station_id"] == station_id].sort_values("time").head(n_points).copy()

    plt.figure(figsize=(12, 5.5))
    plt.plot(plot_df["time"], plot_df["rain_obs"], label="Observed", linewidth=2.2)
    plt.plot(plot_df["time"], plot_df["persistence_pred"], label="Persistence", linestyle="--", linewidth=1.5)
    plt.plot(plot_df["time"], plot_df["ssm_basic_forecast"], label="SSM Basic", linestyle="--", linewidth=1.7)
    plt.plot(plot_df["time"], plot_df["ml_pred"], label="ML Step 5", linewidth=1.7)
    plt.plot(plot_df["time"], plot_df["hybrid_residual_ssm_pred"], label="Hybrid Residual SSM", linewidth=1.7)

    if "hybrid_residual_lag_pred" in plot_df.columns:
        plt.plot(plot_df["time"], plot_df["hybrid_residual_lag_pred"], label="Residual Lag", linewidth=1.7)
    if "hybrid_regime_switch" in plot_df.columns:
        plt.plot(plot_df["time"], plot_df["hybrid_regime_switch"], label="Regime Switch", linewidth=1.7)

    best_alpha = 0.80
    best_col = f"hybrid_weighted_pred_{int(round(best_alpha * 100)):03d}"
    if best_col in plot_df.columns:
        plt.plot(plot_df["time"], plot_df[best_col], label=f"Weighted Fusion {best_alpha:.2f}", linewidth=1.7)

    plt.title(f"Rainfall Prediction Comparison (Station {station_id})")
    plt.xlabel("Time")
    plt.ylabel("Rainfall (mm)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, ncol=2)
    _save_fig(out_path)


def plot_scatter(df: pd.DataFrame, pred_col: str, title: str, out_path: Path) -> None:
    work = df[["rain_obs", pred_col]].copy().dropna()
    if work.empty:
        return
    if len(work) > MAX_SCATTER_POINTS:
        work = work.sample(MAX_SCATTER_POINTS, random_state=42)

    x = work["rain_obs"].to_numpy(dtype=float)
    y = work[pred_col].to_numpy(dtype=float)
    vmax = max(x.max(), y.max(), 1e-6)

    plt.figure(figsize=(6.5, 6.0))
    plt.scatter(x, y, alpha=0.25, s=8)
    plt.plot([0, vmax], [0, vmax], linestyle="--", linewidth=1.5)
    plt.xlabel("Observed rainfall (mm)")
    plt.ylabel("Predicted rainfall (mm)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, vmax)
    plt.ylim(0, vmax)
    _save_fig(out_path)


def plot_monthly_metric(monthly_metrics: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    if monthly_metrics.empty:
        return
    pivot = monthly_metrics.pivot(index="month", columns="model", values=metric).sort_index()

    plt.figure(figsize=(9, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", linewidth=2, label=col)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(metric.upper())
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    _save_fig(out_path)


def plot_residual_acf(acf_df: pd.DataFrame, title: str, out_path: Path) -> None:
    if acf_df.empty:
        return
    plt.figure(figsize=(9, 4.8))
    plt.bar(acf_df["lag"], acf_df["acf"], edgecolor="black")
    plt.title(title)
    plt.xlabel("Lag (minutes)")
    plt.ylabel("Autocorrelation")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    _save_fig(out_path)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print("Step 13 — Improved state-space and hybrid benchmarking")

    df = load_step5_predictions_from_s3()
    df = add_persistence_baseline(df)
    df = run_basic_ssm_per_station(df)
    df = run_residual_ssm_per_station(df)
    df = run_residual_lag_benchmark(df, residual_lags=RESIDUAL_LAGS)
    df = add_regime_switch_hybrid(df)
    df = add_weighted_fusions(df, alphas=FUSION_ALPHAS)

    metrics_df = evaluate_all_models(df)
    monthly_metrics_df = evaluate_by_month(df)
    station_metrics_df = evaluate_by_station(df)
    rainonly_metrics_df = evaluate_rain_only(df)
    intensity_metrics_df = evaluate_by_intensity_bin(df)

    df_att = add_attenuation_columns(df)

    acf_ml = compute_stationwise_residual_acf(df, pred_col="ml_pred", max_lag=60)
    acf_hybrid = compute_stationwise_residual_acf(df, pred_col="hybrid_residual_ssm_pred", max_lag=60)

    # Save outputs
    df.to_csv(OUTPUT_DIR / "step13_hybrid_forecasts.csv", index=False)
    df.to_parquet(OUTPUT_DIR / "step13_hybrid_forecasts.parquet", index=False)

    metrics_df.to_csv(OUTPUT_DIR / "step13_hybrid_metrics_overall.csv", index=False)
    monthly_metrics_df.to_csv(OUTPUT_DIR / "step13_hybrid_metrics_by_month.csv", index=False)
    station_metrics_df.to_csv(OUTPUT_DIR / "step13_hybrid_metrics_by_station.csv", index=False)
    rainonly_metrics_df.to_csv(OUTPUT_DIR / "step13_hybrid_metrics_rainonly.csv", index=False)
    intensity_metrics_df.to_csv(OUTPUT_DIR / "step13_hybrid_metrics_by_intensity.csv", index=False)

    acf_ml.to_csv(OUTPUT_DIR / "step13_ml_residual_acf.csv", index=False)
    acf_hybrid.to_csv(OUTPUT_DIR / "step13_hybrid_residual_acf.csv", index=False)

    df_att.to_csv(OUTPUT_DIR / "step13_hybrid_with_attenuation.csv", index=False)
    df_att.to_parquet(OUTPUT_DIR / "step13_hybrid_with_attenuation.parquet", index=False)

    # Plots
    plot_time_series_comparison(df, OUTPUT_DIR / "fig_step13_hybrid_timeseries.png")
    plot_scatter(df, "ssm_basic_forecast", "Observed vs Predicted Rainfall (SSM Basic)", OUTPUT_DIR / "fig_scatter_ssm_basic.png")
    plot_scatter(df, "ml_pred", "Observed vs Predicted Rainfall (ML Step 5)", OUTPUT_DIR / "fig_scatter_ml.png")
    plot_scatter(df, "hybrid_residual_ssm_pred", "Observed vs Predicted Rainfall (Hybrid Residual SSM)", OUTPUT_DIR / "fig_scatter_hybrid_residual_ssm.png")

    if "hybrid_residual_lag_pred" in df.columns:
        plot_scatter(df, "hybrid_residual_lag_pred", "Observed vs Predicted Rainfall (Hybrid Residual Lag)", OUTPUT_DIR / "fig_scatter_hybrid_residual_lag.png")

    if "hybrid_regime_switch" in df.columns:
        plot_scatter(df, "hybrid_regime_switch", "Observed vs Predicted Rainfall (Hybrid Regime Switch)", OUTPUT_DIR / "fig_scatter_hybrid_regime_switch.png")

    weighted_rows = metrics_df[metrics_df["model"].str.startswith("Hybrid_WeightedFusion_")].copy()
    if not weighted_rows.empty:
        best_weighted_model = weighted_rows.sort_values("rmse").iloc[0]["model"]
        alpha_str = best_weighted_model.split("_")[-1]
        alpha_val = float(alpha_str)
        pred_col = f"hybrid_weighted_pred_{int(round(alpha_val * 100)):03d}"
        if pred_col in df.columns:
            plot_scatter(df, pred_col, f"Observed vs Predicted Rainfall ({best_weighted_model})", OUTPUT_DIR / "fig_scatter_hybrid_weighted_best.png")

    plot_monthly_metric(monthly_metrics_df, "rmse", "Monthly RMSE Comparison", OUTPUT_DIR / "fig_monthly_rmse.png")
    plot_monthly_metric(monthly_metrics_df, "mae", "Monthly MAE Comparison", OUTPUT_DIR / "fig_monthly_mae.png")
    plot_monthly_metric(monthly_metrics_df, "r2", "Monthly R² Comparison", OUTPUT_DIR / "fig_monthly_r2.png")

    plot_residual_acf(acf_ml, "Autocorrelation of ML Residuals", OUTPUT_DIR / "fig_residual_acf_ml.png")
    plot_residual_acf(acf_hybrid, "Autocorrelation of Hybrid Residuals", OUTPUT_DIR / "fig_residual_acf_hybrid.png")

    print("\nOverall metrics:")
    print(metrics_df)

    available_months = sorted(df["month"].dropna().astype(int).unique().tolist())
    print(f"\nAvailable months used: {available_months}")
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

