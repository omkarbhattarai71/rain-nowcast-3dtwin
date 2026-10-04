"""Latency / memory benchmark of the in-process service (run on a laptop and on the edge device).

  python src/deploy/bench_latency.py --config smoke --model lgbm_hurdle --repeats 50
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rainnow.config import load_config, setup_logging  # noqa: E402
from rainnow.dataset import load_quality  # noqa: E402
from rainnow.serving import NowcastService, records_from_truth  # noqa: E402
from rainnow.truth import MET_COLS, load_truth  # noqa: E402


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--start", default="2025-07-10T06:00")
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tracemalloc.start()
    svc = NowcastService(cfg, model=args.model)
    start = pd.Timestamp(args.start, tz="UTC")
    q = load_quality(cfg)
    stations = sorted(q[(q.year == start.year) & q.ok].station_id.unique())
    hist = int(cfg.deploy.history_minutes)
    for s in stations:
        d = load_truth(cfg, s, start - pd.Timedelta(minutes=hist), start + pd.Timedelta(minutes=1), ["p1_obs", "p10_obs"] + MET_COLS)
        svc.ingest(records_from_truth(d, s))
    lat_all, lat_one = [], []
    for _ in range(args.repeats):
        t = time.perf_counter()
        svc.forecast(None, start)
        lat_all.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        svc.forecast([stations[0]], start)
        lat_one.append((time.perf_counter() - t) * 1000)
    cur, peak = tracemalloc.get_traced_memory()
    res = {
        "machine": platform.platform(), "processor": platform.processor() or platform.machine(),
        "model": svc.model_name, "stations": len(stations),
        "all_stations_ms_p50": float(np.percentile(lat_all, 50)), "all_stations_ms_p99": float(np.percentile(lat_all, 99)),
        "one_station_ms_p50": float(np.percentile(lat_one, 50)), "one_station_ms_p99": float(np.percentile(lat_one, 99)),
        "python_heap_peak_mb": peak / 1e6,
    }
    out = cfg.path("deploy", f"latency_{platform.machine()}_{svc.model_name}.json", results=True, mkdir=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
