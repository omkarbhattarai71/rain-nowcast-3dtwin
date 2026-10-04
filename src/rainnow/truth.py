"""Validated 1-minute rainfall ground truth.

DMI convention (verified on the data: 100 % exact match at offset 0):
    precip_past10min observed at T  ==  sum of precip_past1min observed at T-9 ... T.

Both series have 0.1 mm (tipping-bucket) resolution in recent years and are separate products,
so "add up" means agreement within two tips (0.2 mm) or 20 %.

Rules per 10-minute window (T-9 .. T):
  * 10-min total == 0 and the 1-min values sum to <= one tip      -> valid (1-min values kept)
  * 10-min total  > 0 and the available 1-min values add up to it -> valid; unreported minutes are 0
  * no 10-min total, but all ten 1-min values present             -> valid
  * anything else                                                  -> invalid (target = NaN)

Missing data is never turned into "no rain". This replaces the supervisors' step 2, which
filled missing minutes with 0 and spread 10-minute totals uniformly (target leakage).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MET_COLS = ["temp_dry", "humidity", "wind_speed", "wind_dir", "cloud_cover"]
TRUTH_COLS = ["p1_obs", "p10_obs", "precip", "valid"] + MET_COLS

_RANGES = {
    "temp_dry": (-45.0, 45.0),
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, 60.0),
    "wind_dir": (0.0, 360.0),
    "cloud_cover": (0.0, 100.0),
}


def read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["time"])
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # duplicate timestamps: take the first non-null value per column
    return df.groupby("time").first().sort_index()


def read_overlap(dir_: Path) -> pd.Series | None:
    """Dense 1-min precipitation (explicit zeros) from ml_pipeline/00_raw/station_1min_overlap."""
    files = sorted(Path(dir_).rglob("*.csv")) if dir_ and Path(dir_).exists() else []
    if not files:
        return None
    parts = [pd.read_csv(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed", errors="coerce")
    s = pd.to_numeric(df["precip_past1min"], errors="coerce")
    s.index = df["time"]
    s = s[s.index.notna()]
    return s.groupby(level=0).first().sort_index()


def clean_raw(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    if "precip_past1min" in df:
        p = df["precip_past1min"]
        df["precip_past1min"] = p.where((p >= 0) & (p <= cfg.truth.p1_max_mm))
    if "precip_past10min" in df:
        p = df["precip_past10min"]
        df["precip_past10min"] = p.where((p >= 0) & (p <= cfg.truth.p10_max_mm))
    if "cloud_cover" in df:
        # DMI code 112 = sky obscured (fog/precipitation) -> treat as overcast
        df["cloud_cover"] = df["cloud_cover"].where(df["cloud_cover"] != 112, 100.0)
    for c, (lo, hi) in _RANGES.items():
        if c in df:
            df[c] = df[c].where((df[c] >= lo) & (df[c] <= hi))
    return df


def harmonise_tips(p1: pd.Series, tip: float) -> pd.Series:
    """Re-express 1-min rain as whole tipping-bucket tips of `tip` mm.

    Some gauges reported 0.01 mm in 2021-2023 while all other years use 0.1 mm. A tip is
    emitted whenever the cumulative rain crosses a multiple of `tip` (exactly how a tipping
    bucket works); for data already in whole tips this is the identity.
    """
    v = p1.to_numpy(dtype="float64")
    obs = ~np.isnan(v)
    cum = np.cumsum(np.nan_to_num(v))
    tips = np.floor(cum / tip + 1e-6)
    out = np.diff(tips, prepend=0.0) * tip
    return pd.Series(np.where(obs, out, np.nan), index=p1.index)


def build_truth(raw: pd.DataFrame, cfg, overlap: pd.Series | None = None) -> pd.DataFrame:
    """Return a 1-minute UTC grid with p1_obs, p10_obs, precip (validated truth), valid, met columns."""
    raw = clean_raw(raw, cfg)
    start = raw.index.min().floor("10min")
    end = raw.index.max().ceil("10min")
    grid = pd.date_range(start, end, freq="1min", tz="UTC", name="time")
    n = len(grid)

    p1 = raw["precip_past1min"].reindex(grid) if "precip_past1min" in raw else pd.Series(np.nan, index=grid)
    if overlap is not None and len(overlap):
        ov = overlap.where((overlap >= 0) & (overlap <= cfg.truth.p1_max_mm))
        p1 = p1.combine_first(ov.reindex(grid))
    p10 = raw["precip_past10min"] if "precip_past10min" in raw else pd.Series(dtype=float)
    p10 = p10[(p10.index.minute % 10 == 0) & (p10.index.second == 0)].reindex(grid)

    tip = float(cfg.truth.get("tip_mm", 0.0) or 0.0)
    if tip > 0:
        p1 = harmonise_tips(p1, tip)
        p10 = (np.round(p10 / tip) * tip).where(p10.notna())
    p1v = p1.to_numpy(dtype="float64")
    has1 = ~np.isnan(p1v)
    pos = np.arange(n)
    win = (pos + 9) // 10                       # grid starts on a 10-minute boundary
    nwin = win[-1] + 1
    n1 = np.bincount(win, weights=has1.astype(float), minlength=nwin)
    s1 = np.bincount(win, weights=np.nan_to_num(p1v), minlength=nwin)
    end_pos = np.minimum(np.arange(nwin) * 10, n - 1)
    p10w = p10.to_numpy(dtype="float64")[end_pos]
    p10w[np.arange(nwin) * 10 > n - 1] = np.nan

    tol = np.maximum(cfg.truth.mass_tol_abs_mm, cfg.truth.mass_tol_rel * np.nan_to_num(p10w))
    dry_ok = (p10w == 0) & (s1 <= cfg.truth.dry_tol_mm)
    wet_ok = (p10w > 0) & (n1 >= 1) & (np.abs(s1 - p10w) <= tol)
    # without a 10-min total, a complete window only counts where the 1-min stream is dense
    # (in sparse-reporting years DMI sends 1-min values mainly during rain -> selection bias)
    dense_regime = pd.Series(has1.astype(float)).rolling(121, center=True, min_periods=1).mean().to_numpy()
    dense_ok = np.isnan(p10w) & (n1 >= 10) & (dense_regime[np.minimum(np.arange(nwin) * 10, n - 1)] >= 0.9)
    valid_w = dry_ok | wet_ok | dense_ok

    valid = valid_w[win]
    # where the two gauge products agree, the 1-min series is the truth (unreported minutes = 0)
    precip = np.where(valid, np.nan_to_num(p1v), np.nan)

    out = pd.DataFrame(index=grid)
    out["p1_obs"] = p1v.astype("float32")
    out["p10_obs"] = p10.to_numpy(dtype="float32")
    out["precip"] = precip.astype("float32")
    out["valid"] = valid.astype("int8")
    for c in MET_COLS:
        out[c] = raw[c].reindex(grid).to_numpy(dtype="float32") if c in raw else np.float32(np.nan)
    return out


def quality_by_year(tr: pd.DataFrame) -> pd.DataFrame:
    """Per-year quality numbers used to select benchmark station-years."""
    rows = []
    for year, g in tr.groupby(tr.index.year):
        p10 = g["p10_obs"].dropna()
        rain_windows = p10[p10 > 0]
        if len(rain_windows):
            # a rain window is covered if its end minute is valid
            cov = float(g.loc[rain_windows.index, "valid"].mean())
        else:
            cov = np.nan
        p10_sum = float(p10.sum())
        rows.append(
            {
                "year": int(year),
                "minutes": len(g),
                "valid_frac": float(g["valid"].mean()),
                "p1_frac": float(g["p1_obs"].notna().mean()),
                "rain_windows": int(len(rain_windows)),
                "rain_coverage": cov,
                "mass_ratio": float(g["precip"].sum() / p10_sum) if p10_sum > 0 else np.nan,
                "wet_minutes": int((g["precip"] > 0).sum()),
                "precip_mm": float(g["precip"].sum()),
                "met_frac": float(g["temp_dry"].notna().mean() * 10),  # met is 10-minute data
            }
        )
    return pd.DataFrame(rows)


def station_year_ok(q: pd.Series, cfg) -> bool:
    lo, hi = cfg.truth.mass_ratio_range
    return bool(
        q.valid_frac >= cfg.truth.min_valid_frac
        and q.rain_windows >= cfg.truth.min_rain_windows
        and (q.rain_coverage if q.rain_coverage == q.rain_coverage else 0) >= cfg.truth.min_rain_coverage
        and (lo <= q.mass_ratio <= hi if q.mass_ratio == q.mass_ratio else False)
    )


def truth_path(cfg, sid: str) -> Path:
    return cfg.path("interim", "truth", f"{sid}.parquet")


def load_truth(cfg, sid: str, start=None, end=None, columns: list[str] | None = None) -> pd.DataFrame:
    filters = []
    if start is not None:
        filters.append(("time", ">=", pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)))
    if end is not None:
        filters.append(("time", "<", pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)))
    cols = None if columns is None else ["time"] + [c for c in columns if c != "time"]
    df = pd.read_parquet(truth_path(cfg, sid), columns=cols, filters=filters or None)
    return df.set_index("time")
