"""Causal per-minute input channels.

Everything here at minute t uses only observations made at or before t. The same functions
are used offline (training) and online (deploy/service.py), so there is no train/serve skew.

Input of the station-level functions is a minute-grid frame with the raw observation columns
produced by truth.build_truth (p1_obs, p10_obs, temp_dry, humidity, wind_speed, wind_dir,
cloud_cover). The validated target `precip` is NEVER used here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .stations import neighbour_table

QUADRANTS = ["N", "E", "S", "W"]


def _age_since_obs(mask: np.ndarray, cap: int) -> np.ndarray:
    """Minutes since the last True in mask (cap when never seen)."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    age = np.arange(n) - last
    age[last < 0] = cap
    return np.minimum(age, cap).astype("float32")


def _asof(s: pd.Series, max_age: int) -> pd.Series:
    return s.ffill(limit=max_age)


def rain_signal(rain_rt: pd.Series, p10_last: pd.Series, p10_age: pd.Series) -> pd.Series:
    """Best real-time estimate of rain over the last 10 minutes (mm), used for neighbour features."""
    roll10 = rain_rt.rolling(10, min_periods=1).sum()
    p10_recent = p10_last.where(p10_age <= 9, 0.0).fillna(0.0)
    return np.maximum(roll10, p10_recent).astype("float32")


def station_channels(obs: pd.DataFrame, cfg, lat: float, lon: float) -> pd.DataFrame:
    """Station-only causal channels on the minute grid of `obs`."""
    max_age = int(cfg.channels.met_max_age_min)
    out = pd.DataFrame(index=obs.index)
    p1 = obs["p1_obs"]
    out["rain_rt"] = p1.fillna(0.0).astype("float32")
    out["rain_avail"] = p1.notna().astype("float32")
    p10_mask = obs["p10_obs"].notna().to_numpy()
    out["p10_last"] = obs["p10_obs"].ffill(limit=30).fillna(0.0).astype("float32")
    out["p10_age"] = _age_since_obs(p10_mask, 60)
    out["rain_signal"] = rain_signal(out["rain_rt"], out["p10_last"], out["p10_age"])

    temp = _asof(obs["temp_dry"], max_age)
    rh = _asof(obs["humidity"], max_age)
    ws = _asof(obs["wind_speed"], max_age)
    wd = _asof(obs["wind_dir"], max_age)
    cc = _asof(obs["cloud_cover"], max_age)
    out["temp"] = temp
    out["rh"] = rh
    out["wind_speed"] = ws
    rad = np.radians(wd)
    out["wind_u"] = -ws * np.sin(rad)          # direction the air moves towards (east +)
    out["wind_v"] = -ws * np.cos(rad)          # (north +)
    out["cloud"] = cc / 100.0
    # dew-point depression (Magnus formula); small depression = saturated air
    a, b = 17.62, 243.12
    gamma = np.log(np.clip(rh, 1, 100) / 100.0) + a * temp / (b + temp)
    out["dewpoint_dep"] = temp - b * gamma / (a - gamma)
    met_mask = obs[["temp_dry", "humidity", "wind_speed"]].notna().any(axis=1).to_numpy()
    out["met_age"] = _age_since_obs(met_mask, 120)

    t = obs.index
    hour = t.hour + t.minute / 60.0
    doy = t.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["lat"] = np.float32(lat)
    out["lon"] = np.float32(lon)
    return out.astype("float32")


def neighbour_channels(
    sid: str,
    signal: pd.DataFrame,
    wind_u: pd.DataFrame,
    wind_v: pd.DataFrame,
    nb: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Spatial channels from other stations.

    signal / wind_u / wind_v: minute x station matrices (causal, from station_channels).
    nb: neighbour table rows for this station (neighbour_id, dist_km, bearing_deg).
    """
    r_in, r_out = cfg.channels.neighbour_rings_km
    idx = signal.index
    out = pd.DataFrame(index=idx)
    nb = nb[nb["neighbour_id"].isin(signal.columns)]
    if nb.empty:
        for name in _nb_names():
            out[name] = np.float32(0.0)
        return out
    S = signal[nb["neighbour_id"].tolist()].to_numpy(dtype="float32")
    S = np.nan_to_num(S)
    d = nb["dist_km"].to_numpy()
    b = nb["bearing_deg"].to_numpy()
    q = (((b + 45.0) // 90.0) % 4).astype(int)
    rings = [(0.0, r_in), (r_in, r_out)]
    for ri, (lo, hi) in enumerate(rings):
        for qi, qn in enumerate(QUADRANTS):
            m = (d >= lo) & (d < hi) & (q == qi) if ri == 0 else (d >= lo) & (d <= hi) & (q == qi)
            out[f"nb_r{ri}_{qn}"] = S[:, m].mean(axis=1) if m.any() else 0.0
    out["nb_max"] = S.max(axis=1)
    out["nb_mean"] = S.mean(axis=1)
    out["nb_wet_frac"] = (S > 0).mean(axis=1)
    out["nb_nearest"] = S[:, int(np.argmin(d))]
    out["nb_max_trend10"] = out["nb_max"] - out["nb_max"].shift(10).fillna(0.0)

    # regional wind (mean over stations within the wind radius, incl. the station itself)
    wcols = [c for c in [sid] + nb.loc[nb["dist_km"] <= cfg.channels.wind_radius_km, "neighbour_id"].tolist()
             if c in wind_u.columns]
    if wcols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows -> NaN (no wind data)
            u = np.nanmean(wind_u[wcols].to_numpy(dtype="float32"), axis=1)
            v = np.nanmean(wind_v[wcols].to_numpy(dtype="float32"), axis=1)
    else:
        u = v = np.full(len(idx), np.nan, dtype="float32")
    has_wind = ~(np.isnan(u) | np.isnan(v)) & ((np.nan_to_num(u) ** 2 + np.nan_to_num(v) ** 2) > 0.01)
    # wind blows towards atan2(u, v); rain comes FROM the opposite direction
    wind_from = (np.degrees(np.arctan2(np.nan_to_num(u), np.nan_to_num(v))) + 180.0) % 360.0
    for ri, (lo, hi) in enumerate(rings):
        m = (d >= lo) & (d <= hi)
        if not m.any():
            out[f"nb_upwind_r{ri}"] = 0.0
            continue
        w = np.cos(np.radians(b[m][None, :] - wind_from[:, None]))
        w = np.clip(w, 0.0, None) * has_wind[:, None]
        ws = w.sum(axis=1)
        val = (w * S[:, m]).sum(axis=1) / np.where(ws > 0, ws, 1.0)
        out[f"nb_upwind_r{ri}"] = np.where(ws > 0, val, S[:, m].mean(axis=1))
    out["regional_wind_u"] = np.nan_to_num(u)
    out["regional_wind_v"] = np.nan_to_num(v)
    return out.astype("float32")


def _nb_names() -> list[str]:
    names = [f"nb_r{ri}_{q}" for ri in range(2) for q in QUADRANTS]
    names += ["nb_max", "nb_mean", "nb_wet_frac", "nb_nearest", "nb_max_trend10",
              "nb_upwind_r0", "nb_upwind_r1", "regional_wind_u", "regional_wind_v"]
    return names


def neighbour_inputs(obs: pd.DataFrame, cfg) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Cheap per-station series needed by other stations: rain signal and wind components."""
    max_age = int(cfg.channels.met_max_age_min)
    rain_rt = obs["p1_obs"].fillna(0.0)
    p10_last = obs["p10_obs"].ffill(limit=30).fillna(0.0)
    p10_age = pd.Series(_age_since_obs(obs["p10_obs"].notna().to_numpy(), 60), index=obs.index)
    sig = rain_signal(rain_rt, p10_last, p10_age)
    ws = _asof(obs["wind_speed"], max_age)
    rad = np.radians(_asof(obs["wind_dir"], max_age))
    return sig, (-ws * np.sin(rad)).astype("float32"), (-ws * np.cos(rad)).astype("float32")


def build_all_channels(obs_by_station: dict[str, pd.DataFrame], targets: list[str], stab: pd.DataFrame, cfg):
    """Channels for `targets`, using every station in obs_by_station as a potential neighbour.

    Returns {sid: DataFrame(channels)} on the common minute grid.
    """
    idx = None
    for df in obs_by_station.values():
        idx = df.index if idx is None else idx.union(df.index)
    sig, wu, wv = {}, {}, {}
    for sid, obs in obs_by_station.items():
        sig[sid], wu[sid], wv[sid] = neighbour_inputs(obs.reindex(idx), cfg)
    signal = pd.DataFrame(sig, index=idx)
    wu = pd.DataFrame(wu, index=idx)
    wv = pd.DataFrame(wv, index=idx)
    nbt = neighbour_table(stab, list(obs_by_station), max_km=max(cfg.channels.neighbour_rings_km))
    out = {}
    for sid in targets:
        r = stab.loc[sid]
        base = station_channels(obs_by_station[sid].reindex(idx), cfg, float(r.lat), float(r.lon))
        nbc = neighbour_channels(sid, signal, wu, wv, nbt[nbt.station_id == sid], cfg)
        out[sid] = pd.concat([base, nbc], axis=1)
    return out


STATION_CHANNELS = [
    "rain_rt", "rain_avail", "p10_last", "p10_age", "rain_signal",
    "temp", "rh", "wind_speed", "wind_u", "wind_v", "cloud", "dewpoint_dep", "met_age",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "lat", "lon",
]
NEIGHBOUR_CHANNELS = _nb_names()
ALL_CHANNELS = STATION_CHANNELS + NEIGHBOUR_CHANNELS
