"""Step 05 - tabular feature files per split and the evaluation index shared by all models.

Writes data/processed/tabular/<bench>/<split>/<sid>_<year>.parquet with columns
  station_id, time, <features>, y (target = rain in t+1), w (training weight), rain_now
and results/bench<bench>/index/{val,test}.parquet (the rows every model is scored on).
"""
from __future__ import annotations

import logging

import pandas as pd

from ..dataset import SPLITS, eval_mask, index_path, load_part, sample_weights, split_parts
from ..features import build_features

log = logging.getLogger(__name__)


def tab_dir(cfg, bench, split):
    return cfg.path("processed", "tabular", bench, split)


def run(cfg, args) -> None:
    for bench in args.bench:
        for split in SPLITS:
            parts = split_parts(cfg, bench, split)
            if not parts:
                log.warning("bench %s split %s: no station-years pass the quality gate", bench, split)
                continue
            d = tab_dir(cfg, bench, split)
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*.parquet"):
                old.unlink()
            idx_rows, n_rows = [], 0
            for part in parts:
                df = load_part(cfg, part, bench)
                feats = build_features(df.drop(columns=["target", "target_valid", "in_period"]), cfg)
                ev = eval_mask(df)
                variants = {}
                if split == "train":
                    w = sample_weights(df, cfg, part, split)
                    variants[""] = (w > 0, w)
                else:
                    variants[""] = (ev, ev.astype("float32"))
                    if split == "val":
                        w = sample_weights(df, cfg, part, split)
                        variants["_sampled"] = (w > 0, w)
                for suffix, (m, w) in variants.items():
                    if not m.any():
                        continue                        # nothing to train/score in this part
                    out = feats[m].copy()
                    out.insert(0, "station_id", part.sid)
                    out["y"] = df["target"].to_numpy()[m]
                    out["w"] = w[m]
                    out["rain_now"] = df["rain_rt"].to_numpy()[m]
                    out.index.name = "time"
                    out.reset_index().to_parquet(d / f"{part.sid}_{part.year}{suffix}.parquet", index=False)
                    if suffix == "":
                        n_rows += int(m.sum())
                if split != "train" and ev.any():
                    idx_rows.append(pd.DataFrame({
                        "station_id": part.sid, "time": df.index[ev],
                        "y_true": df["target"].to_numpy()[ev], "rain_now": df["rain_rt"].to_numpy()[ev],
                    }))
            if idx_rows:
                idx = pd.concat(idx_rows, ignore_index=True)
                idx["day"] = idx["time"].dt.strftime("%Y-%m-%d")
                idx.to_parquet(index_path(cfg, bench, split), index=False)
            log.info("bench %s %-5s: %d parts, %d rows", bench, split, len(parts), n_rows)


def load_tabular(cfg, bench: str, split: str, sampled: bool = False, max_rows: int | None = None,
                 columns: list[str] | None = None) -> pd.DataFrame:
    d = tab_dir(cfg, bench, split)
    if sampled:
        files = sorted(d.glob("*_sampled.parquet"))
    else:
        files = sorted(f for f in d.glob("*.parquet") if not f.stem.endswith("_sampled"))
    if not files:
        raise FileNotFoundError(f"no tabular files in {d}; run step05")
    frames = [x for x in (pd.read_parquet(f, columns=columns) for f in files) if len(x)]
    if not frames:
        raise FileNotFoundError(f"all tabular files in {d} are empty")
    df = pd.concat(frames, ignore_index=True)
    if max_rows and len(df) > max_rows:
        # uniform subsample: relative weights (and hence calibration) are unchanged
        df = df.sample(max_rows, random_state=int(cfg.seed))
    df = df.set_index("time")
    return df


def iter_tabular(cfg, bench: str, split: str):
    """Yield (file_stem, frame) for prediction without loading a whole split into memory."""
    for f in sorted(tab_dir(cfg, bench, split).glob("*.parquet")):
        if f.stem.endswith("_sampled"):
            continue
        df = pd.read_parquet(f)
        if len(df):
            yield f.stem, df.set_index("time")


def xyw(df: pd.DataFrame):
    feats = [c for c in df.columns if c not in ("station_id", "y", "w", "rain_now")]
    return df[feats], df["y"].to_numpy(dtype="float64"), df["w"].to_numpy(dtype="float64")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("station_id", "y", "w", "rain_now")]
