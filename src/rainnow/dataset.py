"""Benchmark station-year selection, split loading, evaluation indices and training-row sampling."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .truth import station_year_ok

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Part:
    """One station-year slice of a split."""

    sid: str
    year: int
    start: pd.Timestamp
    end: pd.Timestamp


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def period(cfg, bench: str, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    s, e = cfg.splits[bench][split]
    return _ts(s), _ts(e)


def load_quality(cfg) -> pd.DataFrame:
    q = pd.read_csv(cfg.path("interim", "truth", "quality.csv"), dtype={"station_id": str})
    q["ok"] = [station_year_ok(r, cfg) for r in q.itertuples()]
    return q


def radar_stations(cfg) -> set[str] | None:
    f = cfg.path("processed", "radar", "stations_covered.json")
    if f.exists():
        return set(json.loads(f.read_text()))
    return None


def split_parts(cfg, bench: str, split: str) -> list[Part]:
    """Station-year slices of a split that pass the quality gate (and radar coverage for B)."""
    q = load_quality(cfg)
    ok = q[q.ok]
    start, end = period(cfg, bench, split)
    covered = radar_stations(cfg) if bench == "B" else None
    parts = []
    for r in ok.sort_values(["station_id", "year"]).itertuples():
        ys, ye = _ts(f"{r.year}-01-01"), _ts(f"{r.year + 1}-01-01")
        s, e = max(start, ys), min(end, ye)
        if s >= e:
            continue
        if covered is not None and r.station_id not in covered:
            continue
        cp = channel_path(cfg, r.station_id, r.year)
        if not cp.exists():
            continue
        # a station can stop reporting before the split period (e.g. data ending in August while
        # the split is October): such parts would yield empty frames downstream, so drop them here
        if _valid_rows(str(cp), cp.stat().st_mtime, s.value, e.value) < int(cfg.get("min_part_rows", 60)):
            continue
        parts.append(Part(r.station_id, int(r.year), s, e))
    if not parts:
        log.warning("bench %s split %s: no station-year has valid target rows in %s - %s", bench, split, start, end)
    return parts


@lru_cache(maxsize=4096)
def _valid_rows(path: str, mtime: float, start_ns: int, end_ns: int) -> int:
    """Number of minutes with a valid target inside [start, end) of a channel file (cached)."""
    df = pd.read_parquet(path, columns=["time", "target_valid"])
    t = pd.DatetimeIndex(df["time"]).as_unit("ns").asi8
    m = (t >= start_ns) & (t < end_ns)
    return int((df["target_valid"].to_numpy()[m] > 0).sum())


def channel_path(cfg, sid: str, year: int) -> Path:
    return cfg.path("processed", "channels", sid, f"{year}.parquet")


def radar_minute_path(cfg, sid: str) -> Path:
    return cfg.path("processed", "radar", "minute", f"{sid}.parquet")


def load_part(cfg, part: Part, bench: str, pad: int | None = None) -> pd.DataFrame:
    """Channels (+ target) for a part, with `pad` minutes of history before part.start.

    Adds a boolean column `in_period` marking the rows that belong to the part itself.
    """
    pad = int(cfg.channels.pad_minutes if pad is None else pad)
    df = pd.read_parquet(channel_path(cfg, part.sid, part.year))
    df = df.set_index("time") if "time" in df.columns else df
    lo = part.start - pd.Timedelta(minutes=pad)
    df = df[(df.index >= lo) & (df.index < part.end)]
    if bench == "B":
        rp = radar_minute_path(cfg, part.sid)
        if rp.exists():
            rad = pd.read_parquet(rp)
            rad = rad.set_index("time") if "time" in rad.columns else rad
            df = df.join(rad, how="left")
        else:
            log.warning("no radar channels for %s", part.sid)
    df["in_period"] = df.index >= part.start
    return df


def eval_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["in_period"] & (df["target_valid"] > 0)).to_numpy()


def _seed_for(cfg, part: Part, split: str) -> int:
    h = hashlib.md5(f"{cfg.seed}-{part.sid}-{part.year}-{split}".encode()).hexdigest()
    return int(h[:8], 16)


def sample_weights(df: pd.DataFrame, cfg, part: Part, split: str) -> np.ndarray:
    """Training-row selection: keep every row near rain, plus a random share of dry rows.

    Returns a weight per row (0 = not used). Kept dry rows get weight 1/keep_frac so that
    the weighted data has the true class balance (calibrated probabilities).
    """
    ctx = int(cfg.sampling.context_minutes)
    sig = df["rain_signal"].to_numpy()
    nb = df["nb_max"].to_numpy() if "nb_max" in df else np.zeros(len(df))
    near_now = (np.nan_to_num(sig) > 0) | (np.nan_to_num(nb) > 0)
    near = pd.Series(near_now).rolling(ctx, min_periods=1).max().to_numpy() > 0
    near |= np.nan_to_num(df["target"].to_numpy()) > 0
    rng = np.random.default_rng(_seed_for(cfg, part, split))
    keep_frac = float(cfg.sampling.dry_keep_frac)
    dry_keep = rng.random(len(df)) < keep_frac
    w = np.where(near, 1.0, np.where(dry_keep, 1.0 / keep_frac, 0.0))
    return (w * eval_mask(df)).astype("float32")


def index_path(cfg, bench: str, split: str) -> Path:
    return cfg.path(f"bench{bench}", "index", f"{split}.parquet", results=True, mkdir=True)


def preds_path(cfg, bench: str, model: str, split: str) -> Path:
    return cfg.path(f"bench{bench}", "preds", model, f"{split}.parquet", results=True, mkdir=True)


def model_dir(cfg, bench: str, model: str) -> Path:
    p = cfg.path(f"bench{bench}", "models", model, results=True)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_preds(cfg, bench: str, model: str, split: str, frames: list[pd.DataFrame]) -> None:
    """Standard prediction file: station_id, time, p_rain, y_hat, q90 (NaN if not provided)."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        log.warning("no predictions for %s/%s", model, split)
        return
    df = pd.concat(frames, ignore_index=True)
    for c in ("p_rain", "y_hat", "q90"):
        if c not in df:
            df[c] = np.nan
    df["y_hat"] = df["y_hat"].clip(lower=0)
    df = df[["station_id", "time", "p_rain", "y_hat", "q90"]]
    df[["p_rain", "y_hat", "q90"]] = df[["p_rain", "y_hat", "q90"]].astype("float32")
    df.to_parquet(preds_path(cfg, bench, model, split), index=False)
    log.info("wrote %s preds %-28s %-5s rows=%d", bench, model, split, len(df))


def pred_frame(df: pd.DataFrame, sid: str, mask: np.ndarray, **cols) -> pd.DataFrame:
    out = pd.DataFrame({"station_id": sid, "time": df.index[mask]})
    for k, v in cols.items():
        out[k] = np.asarray(v)
    return out
