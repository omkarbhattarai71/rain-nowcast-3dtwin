"""Step 01 - read-only inventory of S3 and local cache of the raw data.

Downloads: station CSVs, station metadata, radar lookup, dense 2025 1-min data, and the
supervisors' published metric tables. Writes results/inventory/manifest.csv.
(Replaces the original step 1: no bucket creation, no hard-coded API key.)
"""
from __future__ import annotations

import logging

import pandas as pd

from .. import io_s3
from ..stations import norm_id

log = logging.getLogger(__name__)


def run(cfg, args) -> None:
    s3c, region, workers = cfg.s3, cfg.s3.region, int(cfg.s3.workers)
    raw = cfg.path("raw", mkdir=True)

    # station CSVs
    objs = io_s3.list_keys(s3c.station_bucket, "", region, delimiter="/")
    csvs = [o for o in objs if o["Key"].endswith(s3c.station_csv_suffix)]
    if cfg.stations.include:
        csvs = [o for o in csvs if o["Key"][:5] in set(cfg.stations.include)]
    log.info("station CSVs: %d (%.2f GB)", len(csvs), sum(o["Size"] for o in csvs) / 1e9)
    io_s3.download_many(s3c.station_bucket, [(o["Key"], raw / "stations" / o["Key"]) for o in csvs], region, workers)

    # metadata + lookup
    io_s3.download(s3c.station_bucket, s3c.metadata_key, raw / "meta" / "stations_metObs_v1.csv", region)
    io_s3.download(s3c.radar_bucket, s3c.lookup_key, raw / "meta" / "stations_xy_lookup.csv", region)

    # dense 1-minute data (2025-05 .. 2025-10) from the ml_pipeline area
    ov = io_s3.list_keys(s3c.radar_bucket, s3c.overlap_prefix + "/", region)
    items = []
    for o in ov:
        k = o["Key"]
        if "station_id=" not in k or not k.endswith(".csv"):
            continue
        sid_raw, rel = k.split("station_id=", 1)[1].split("/", 1)
        sid = norm_id(sid_raw)
        if cfg.stations.include and sid not in cfg.stations.include:
            continue
        items.append((k, raw / "overlap" / sid / rel))
    log.info("dense 1-min overlap files: %d", len(items))
    io_s3.download_many(s3c.radar_bucket, items, region, workers)

    # published supervisor metrics (for the "as published" comparison)
    pub = cfg.path("inventory", "published", results=True, mkdir=True)
    pub.mkdir(parents=True, exist_ok=True)
    for bucket, key in s3c.published_metrics:
        try:
            io_s3.download(bucket, key, pub / key.replace("/", "__"), region)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not fetch %s/%s: %s", bucket, key, exc)

    # manifest
    rows = []
    for o in csvs:
        p = raw / "stations" / o["Key"]
        head = pd.read_csv(p, nrows=1)
        t = pd.read_csv(p, usecols=["time"])["time"]
        rows.append({
            "station_id": o["Key"][:5], "size_mb": round(o["Size"] / 1e6, 2), "columns": "|".join(head.columns[1:]),
            "has_1min": "precip_past1min" in head.columns, "first": t.iloc[0], "last": t.iloc[-1], "rows": len(t),
            "overlap_days": len(list((raw / "overlap" / o["Key"][:5]).rglob("*.csv"))),
        })
    man = pd.DataFrame(rows)
    out = cfg.path("inventory", "manifest.csv", results=True, mkdir=True)
    man.to_csv(out, index=False)
    log.info("manifest: %d stations, %d with 1-min column -> %s", len(man), int(man.has_1min.sum()), out)
