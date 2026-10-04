"""Step 03 - causal per-minute channels (+ target) for every benchmark station-year."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..channels import build_all_channels
from ..dataset import _ts, channel_path, load_quality
from ..stations import load_station_table
from ..truth import MET_COLS, load_truth

log = logging.getLogger(__name__)


def needed_station_years(cfg) -> pd.DataFrame:
    q = load_quality(cfg)
    years = set()
    for bench in cfg.splits:
        for split in cfg.splits[bench]:
            s, e = _ts(cfg.splits[bench][split][0]), _ts(cfg.splits[bench][split][1])
            years.update(range(s.year, (e - pd.Timedelta(minutes=1)).year + 1))
    return q[q.ok & q.year.isin(sorted(years))]


def run(cfg, args) -> None:
    stab = load_station_table(cfg)
    q = load_quality(cfg)
    all_sids = sorted(q.station_id.unique())
    need = needed_station_years(cfg)
    pad = pd.Timedelta(minutes=int(cfg.channels.pad_minutes))
    log.info("channels for %d station-years", len(need))
    for year, grp in need.groupby("year"):
        start, end = _ts(f"{year}-01-01") - pad, _ts(f"{year + 1}-01-01")
        obs = {}
        for sid in all_sids:
            try:
                df = load_truth(cfg, sid, start, end, ["p1_obs", "p10_obs"] + MET_COLS)
            except Exception as exc:  # noqa: BLE001
                log.warning("truth %s: %s", sid, exc)
                continue
            if len(df):
                obs[sid] = df
        grid = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left", name="time")
        obs = {s: d.reindex(grid) for s, d in obs.items()}
        targets = [s for s in grp.station_id if s in obs]
        chans = build_all_channels(obs, targets, stab, cfg)
        for sid in targets:
            tr = load_truth(cfg, sid, start, end + pd.Timedelta(minutes=1), ["precip", "valid"]).reindex(
                grid.append(pd.DatetimeIndex([end])))
            ch = chans[sid]
            ch["target"] = tr["precip"].shift(-1).iloc[:-1].to_numpy(dtype="float32")
            ch["target_valid"] = tr["valid"].shift(-1).iloc[:-1].fillna(0).to_numpy(dtype="int8")
            p = channel_path(cfg, sid, int(year))
            p.parent.mkdir(parents=True, exist_ok=True)
            ch.reset_index().to_parquet(p, index=False)
        log.info("year %d: %d stations (neighbour pool %d)", year, len(targets), len(obs))
        del obs, chans
    np.save(cfg.path("processed", "channels", "_done.npy"), np.array([1]))
