"""Step 04 - radar features for benchmark B (rebuilt from raw ODIM pseudo-CAPPI files).

1. audit of the supervisors' station lookup against each radar site's own grid
2. per-day: 5-site nearest-radar composite, motion, extrapolation nowcast, station features
3. causal minute-level merge per station (scan time + latency <= t)
4. radar-vs-gauge sanity statistics
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd

from .. import io_s3
from ..config import Cfg
from ..dataset import _ts, load_quality
from ..radar import FNAME_RE, minute_channels, process_day, read_odim
from ..stations import load_station_table
from ..truth import load_truth

log = logging.getLogger(__name__)


def _day_job(args):
    cfg_dict, day, stab = args
    cfg = Cfg.wrap(cfg_dict)
    out = cfg.path("processed", "radar", "scans", f"{day:%Y-%m-%d}.parquet")
    if out.exists():
        return str(out)
    df = process_day(day, cfg, stab)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return str(out)


def lookup_audit(cfg, stab: pd.DataFrame, day) -> pd.DataFrame:
    """Pixel error of the old single lookup (ekxr grid) on each radar's own grid."""
    from pyproj import Transformer

    lk = pd.read_csv(cfg.path("raw", "meta", "stations_xy_lookup.csv"))
    seen, rows = set(), []
    keys = []
    for i in range(14):                                   # some radars have outages: scan up to 2 weeks
        if len(seen | {k.rsplit("/", 1)[-1][:4] for k in keys}) >= len(cfg.radar.sites):
            break
        pre = f"{cfg.s3.radar_prefix}/{day + timedelta(days=i):%Y/%m/%d}/"
        keys += [o["Key"] for o in io_s3.list_keys(cfg.s3.radar_bucket, pre, cfg.s3.region)]
    for key in keys:
        m = FNAME_RE.match(key.rsplit("/", 1)[-1])
        if not m or m.group(1) in seen:
            continue
        o = {"Key": key}
        seen.add(m.group(1))
        _, where = read_odim(io_s3.read_bytes(cfg.s3.radar_bucket, o["Key"], cfg.s3.region))
        tr = Transformer.from_crs("EPSG:4326", where["projdef"], always_xy=True)
        x0, y0 = tr.transform(float(where["LL_lon"]), float(where["LL_lat"]))
        x, y = tr.transform(lk["lon"].to_numpy(), lk["lat"].to_numpy())
        ix = np.floor((x - x0) / float(where["xscale"]))
        iy = (int(where["ysize"]) - 1) - np.floor((y - y0) / float(where["yscale"]))
        err_px = np.hypot(ix - lk["ix"], iy - lk["iy"])
        rows.append({"site": m.group(1), "median_error_px": float(np.median(err_px)),
                     "median_error_km": float(np.median(err_px) * float(where["xscale"]) / 1000),
                     "max_error_km": float(err_px.max() * float(where["xscale"]) / 1000),
                     "stations_inside_site_grid": int(((ix >= 0) & (ix < 960) & (iy >= 0) & (iy < 960)).sum())})
    return pd.DataFrame(rows)


def run(cfg, args) -> None:
    if "B" not in args.bench:
        log.info("benchmark B not requested; skipping radar")
        return
    start = min(_ts(v[0]) for v in cfg.splits.B.values())
    end = max(_ts(v[1]) for v in cfg.splits.B.values())
    q = load_quality(cfg)
    stab = load_station_table(cfg)
    sids = sorted(set(q.station_id) & set(stab.index))
    stab = stab.loc[sids, ["lon", "lat"]]
    out_dir = cfg.path("radar", results=True, mkdir=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        lookup_audit(cfg, stab, start.date()).to_csv(out_dir / "lookup_audit.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("lookup audit failed: %s", exc)

    days = [start.date() + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if _ts(d) < end]
    workers = int(args.workers or cfg.radar.workers)
    log.info("radar: %d days, %d stations, %d workers", len(days), len(sids), workers)
    with ProcessPoolExecutor(workers) as ex:
        for i, p in enumerate(ex.map(_day_job, [(cfg.to_dict(), d, stab) for d in days]), 1):
            if i % 10 == 0 or i == len(days):
                log.info("  radar days done %d/%d", i, len(days))
    files = sorted(cfg.path("processed", "radar", "scans").glob("*.parquet"))
    scans = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    scans = scans[(scans.scan_time >= start - timedelta(hours=1)) & (scans.scan_time < end)]
    log.info("scan rows: %d", len(scans))

    pad = pd.Timedelta(minutes=int(cfg.channels.pad_minutes))
    minutes = pd.date_range(start - pad, end, freq="1min", tz="UTC", inclusive="left")
    covered, qc_rows = [], []
    mdir = cfg.path("processed", "radar", "minute")
    mdir.mkdir(parents=True, exist_ok=True)
    for sid, g in scans.groupby("station_id"):
        cov = float(g["radar_r_pix"].notna().mean())
        if cov >= 0.8:
            covered.append(sid)
        mc = minute_channels(g, minutes, cfg)
        mc.to_parquet(mdir / f"{sid}.parquet", index=False)
        qc_rows.append(_gauge_check(cfg, sid, g, start, end) | {"station_id": sid, "coverage": cov})
    (cfg.path("processed", "radar", "stations_covered.json")).write_text(json.dumps(sorted(covered)))
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(out_dir / "radar_vs_gauge.csv", index=False)
    log.info("radar covers %d/%d stations; median corr(radar, gauge 10-min) = %.3f",
             len(covered), len(qc), qc["corr_mean3_vs_gauge10"].median())


def _gauge_check(cfg, sid, g, start, end) -> dict:
    """Correlation of radar rain rate with the gauge's 10-min rain (both in mm/h)."""
    try:
        tr = load_truth(cfg, sid, start, end, ["precip"])
    except Exception:  # noqa: BLE001
        return {}
    gauge10 = tr["precip"].rolling(10, min_periods=8).sum().shift(-9) * 6.0   # mm/h over scan + 10 min
    s = g.set_index("scan_time")
    j = pd.concat([s["radar_r_mean3"], gauge10.reindex(s.index).rename("gauge")], axis=1).dropna()
    if len(j) < 50 or j["gauge"].std() == 0:
        return {}
    wet = (j["gauge"] > 0) | (j["radar_r_mean3"] > 0.1)
    pod = float(((j["radar_r_mean3"] > 0.1) & (j["gauge"] > 0)).sum() / max((j["gauge"] > 0).sum(), 1))
    return {"corr_mean3_vs_gauge10": float(j.corr().iloc[0, 1]),
            "bias_mmh_wet": float((j.loc[wet, "radar_r_mean3"] - j.loc[wet, "gauge"]).mean()),
            "radar_pod": pod, "n_scans": len(j)}
