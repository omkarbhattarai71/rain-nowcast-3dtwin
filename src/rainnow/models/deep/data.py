"""Window store for sequence models.

Each station-year part is kept as one contiguous float32 array X[T, C] (history padding included),
plus target y[T] and a row weight w[T]. A training example is the window X[t-L+1 : t+1] with
target y[t] (= rain in minute t+1). Windows are gathered with fancy indexing, so no copies of the
dataset are made per epoch.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ...channels import ALL_CHANNELS
from ...dataset import Part, eval_mask, load_part, sample_weights

log = logging.getLogger(__name__)

# channels transformed with log1p(100 * x / div) / 3 instead of z-scoring
RAIN_LIKE = {"rain_rt": 1, "p10_last": 10, "rain_signal": 10}
_NB_RAIN = {c for c in ALL_CHANNELS if c.startswith("nb_r") or c in
            ("nb_max", "nb_mean", "nb_nearest", "nb_upwind_r0", "nb_upwind_r1")}


EXCLUDED = {"rain_avail"}   # reporting artefact, not a weather signal (see features.py)


def deep_channels(bench: str, sample=None) -> list[str]:
    """Input channels; for benchmark B also every radar/nowcast column found in the sample frame(s)."""
    chans = [c for c in ALL_CHANNELS if c not in EXCLUDED]
    if bench == "B" and sample is not None:
        frames = sample if isinstance(sample, (list, tuple)) else [sample]
        extra = sorted({c for f in frames for c in f.columns if c.startswith(("radar_", "nowcast_"))})
        if not extra:
            log.warning("benchmark B: no radar/nowcast columns found - radar variant equals station-only")
        chans += extra
    return chans


def _is_rain_like(c: str) -> bool:
    return c in RAIN_LIKE or c in _NB_RAIN or c.startswith(("radar_r", "nowcast_"))


def fit_scaler(frames: list[pd.DataFrame], channels: list[str]) -> dict:
    """Mean/std for non-rain channels from training frames."""
    sc = {}
    # reindex: a station without radar files lacks the radar columns (filled with NaN here)
    cat = pd.concat([f.reindex(columns=channels).sample(min(len(f), 50000), random_state=0)
                     for f in frames if len(f)])
    for c in channels:
        if _is_rain_like(c):
            sc[c] = {"type": "rain", "div": RAIN_LIKE.get(c, 10 if c.startswith("nb_") else 1)}
        else:
            v = cat[c].to_numpy(dtype="float64")
            v = v[np.isfinite(v)]
            mu, sd = (v.mean(), v.std()) if len(v) else (0.0, 1.0)
            sc[c] = {"type": "z", "mean": float(0 if np.isnan(mu) else mu), "std": float(sd if sd and sd > 1e-6 else 1.0)}
    return sc


def transform(df: pd.DataFrame, channels: list[str], scaler: dict) -> np.ndarray:
    out = np.zeros((len(df), len(channels)), dtype="float32")
    for j, c in enumerate(channels):
        v = df[c].to_numpy(dtype="float32") if c in df else np.zeros(len(df), "float32")
        s = scaler[c]
        if s["type"] == "rain":
            if c.startswith("radar_r") or c.startswith("nowcast_"):
                v = v / 60.0                       # mm/h -> mm/min
            v = np.log1p(100.0 * np.clip(np.nan_to_num(v), 0, None) / s["div"]) / 3.0
        else:
            v = (v - s["mean"]) / s["std"]
        out[:, j] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return out


class WindowStore:
    def __init__(self, window: int):
        self.L = window
        self.X: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        self.w: list[np.ndarray] = []
        self.times: list[pd.DatetimeIndex] = []
        self.sids: list[str] = []
        self.rows: np.ndarray = np.zeros((0, 2), dtype="int64")   # (part_idx, position)
        self.weights: np.ndarray = np.zeros(0, dtype="float32")

    @classmethod
    def build(cls, cfg, parts: list[Part], bench: str, split: str, channels, scaler, train_like: bool,
              frames: list[pd.DataFrame] | None = None):
        st = cls(int(cfg.deep.window))
        rows, wts = [], []
        dtype = cfg.deep.store_dtype
        for i, part in enumerate(parts):
            df = frames[i] if frames is not None else load_part(cfg, part, bench)
            w = sample_weights(df, cfg, part, split) if train_like else eval_mask(df).astype("float32")
            pos = np.where(w > 0)[0]
            pos = pos[pos >= st.L - 1]
            st.X.append(transform(df, channels, scaler).astype(dtype))
            st.y.append(np.nan_to_num(df["target"].to_numpy(dtype="float32")))
            st.times.append(df.index)
            st.sids.append(part.sid)
            rows.append(np.c_[np.full(len(pos), len(st.X) - 1), pos])
            wts.append(w[pos])
        if rows:
            st.rows = np.concatenate(rows).astype("int64")
            st.weights = np.concatenate(wts).astype("float32")
        log.info("WindowStore %s/%s: %d parts, %d windows", bench, split, len(parts), len(st.rows))
        return st

    def __len__(self):
        return len(self.rows)

    def gather(self, idx: np.ndarray):
        """Return X (B, L, C), y (B,), w (B,) for row indices idx."""
        sel = self.rows[idx]
        offs = np.arange(-self.L + 1, 1)
        C = self.X[0].shape[1]
        Xb = np.empty((len(idx), self.L, C), dtype="float32")
        yb = np.empty(len(idx), dtype="float32")
        for p in np.unique(sel[:, 0]):
            m = sel[:, 0] == p
            pos = sel[m, 1]
            Xb[m] = self.X[p][pos[:, None] + offs[None, :]]
            yb[m] = self.y[p][pos]
        return Xb, yb, self.weights[idx]

    def keys(self, idx: np.ndarray) -> pd.DataFrame:
        sel = self.rows[idx]
        sid = np.empty(len(idx), dtype=object)
        ns = np.empty(len(idx), dtype="int64")
        for p in np.unique(sel[:, 0]):
            m = sel[:, 0] == p
            sid[m] = self.sids[p]
            ns[m] = self.times[p].as_unit("ns").asi8[sel[m, 1]]
        return pd.DataFrame({"station_id": sid, "time": pd.to_datetime(ns, utc=True)})
