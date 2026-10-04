"""Tabular (hand-crafted) features computed from the causal channels.

Feature groups (used for ablations):
  P  rain history at the station       M  meteorology
  T  time / location                   N  neighbour stations (spatial)
  R  radar (benchmark B)               S  radar extrapolation nowcast (PySTEPS-style, benchmark B)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .channels import NEIGHBOUR_CHANNELS


def _since_last(mask: np.ndarray, cap: int) -> np.ndarray:
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.arange(n) - last
    out[last < 0] = cap
    return np.minimum(out, cap)


def _run_length(mask: np.ndarray, cap: int) -> np.ndarray:
    """Length of the current run of True values ending at each position."""
    n = len(mask)
    idx = np.where(~mask, np.arange(n), -1)
    last_false = np.maximum.accumulate(idx)
    return np.minimum(np.where(mask, np.arange(n) - last_false, 0), cap)


def build_features(ch: pd.DataFrame, cfg) -> pd.DataFrame:
    """All tabular features on the channel index (causal: row t uses channels <= t)."""
    f = {}
    r = ch["rain_rt"]
    rv = r.to_numpy()
    wet = rv > 0
    # --- P: station rain history
    for k in range(cfg.features.lags):
        f[f"P_lag{k}"] = r.shift(k)
    for w in cfg.features.roll_windows:
        f[f"P_sum{w}"] = r.rolling(w, min_periods=1).sum()
        f[f"P_wetfrac{w}"] = pd.Series(wet, index=r.index).rolling(w, min_periods=1).mean()
    f["P_max10"] = r.rolling(10, min_periods=1).max()
    f["P_max30"] = r.rolling(30, min_periods=1).max()
    f["P_mins_since_rain"] = _since_last(wet, 1440)
    run = _run_length(wet, 600)
    f["P_wet_run"] = run
    f["P_dry_run"] = _run_length(~wet, 1440)
    spell_id = np.cumsum(~wet)
    f["P_spell_accum"] = pd.Series(np.where(wet, rv, 0.0)).groupby(spell_id).cumsum().to_numpy()
    f["P_slope3"] = r - r.shift(3)
    f["P_slope10"] = f["P_sum5"] - f["P_sum5"].shift(5)
    # NOTE: rain_avail (was a 1-min value reported?) is deliberately NOT a feature: DMI reports 1-min
    # values mainly during rain in some years and every minute in others, so it is a non-stationary
    # proxy for rain that broke generalisation from 2020 to 2025.
    f["P_p10_last"] = ch["p10_last"]
    f["P_p10_age"] = ch["p10_age"]
    f["P_p10_prev"] = ch["p10_last"].shift(10)
    f["P_signal"] = ch["rain_signal"]
    # --- M: meteorology
    for c in ["temp", "rh", "wind_speed", "wind_u", "wind_v", "cloud", "dewpoint_dep", "met_age"]:
        f[f"M_{c}"] = ch[c]
    for c in ["temp", "rh", "wind_speed", "dewpoint_dep"]:
        f[f"M_{c}_d30"] = ch[c] - ch[c].shift(30)
        f[f"M_{c}_d60"] = ch[c] - ch[c].shift(60)
    # --- T: time & location
    for c in ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "lat", "lon"]:
        f[f"T_{c}"] = ch[c]
    # --- N: neighbours
    for c in NEIGHBOUR_CHANNELS:
        if c in ch:
            f[f"N_{c}"] = ch[c]
    if "nb_max" in ch:
        f["N_nb_max_roll30"] = ch["nb_max"].rolling(30, min_periods=1).max()
        f["N_nb_wet_frac_roll30"] = ch["nb_wet_frac"].rolling(30, min_periods=1).mean()
        f["N_upwind_r1_d10"] = ch["nb_upwind_r1"] - ch["nb_upwind_r1"].shift(10)
    # --- R / S: radar (only present in benchmark-B channel frames)
    for c in ch.columns:
        if c.startswith("radar_"):
            f[f"R_{c}"] = ch[c]
        elif c.startswith("nowcast_"):
            f[f"S_{c}"] = ch[c]
    out = pd.DataFrame(f, index=ch.index)
    return out.astype("float32")


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in columns:
        g = c.split("_", 1)[0]
        groups.setdefault(g, []).append(c)
    return groups


def select_groups(columns: list[str], groups: list[str]) -> list[str]:
    return [c for c in columns if c.split("_", 1)[0] in groups]


# Feature set mirroring the supervisors' step 3 (lags, rolling sums, calendar; no neighbours).
SUPERVISOR_FEATURES = [
    "P_lag0", "P_lag1", "P_lag2", "P_lag3", "P_lag5", "P_lag9",
    "P_sum10", "P_sum30", "P_sum60",
    "M_temp", "M_rh", "M_cloud", "M_wind_speed", "M_wind_u", "M_wind_v",
    "M_temp_d30", "M_rh_d30", "T_hour_sin", "T_hour_cos", "T_lat", "T_lon",
]
