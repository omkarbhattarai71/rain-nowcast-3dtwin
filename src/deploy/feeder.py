"""Feed observations to the nowcast service and log forecasts.

Replay (recorded test data, minute by minute):
  python src/deploy/feeder.py replay --start 2025-07-10T06:00 --minutes 180 --speed 60
      [--api http://localhost:8000]   # omit --api to run the service in-process
Live (DMI open data API, polls every minute):
  python src/deploy/feeder.py live --api http://localhost:8000 --stations 05065 05075

Replay prints a running comparison of the forecast with the observation one minute later.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rainnow.config import load_config, setup_logging  # noqa: E402
from rainnow.dataset import load_quality  # noqa: E402
from rainnow.serving import NowcastService, records_from_truth  # noqa: E402
from rainnow.truth import MET_COLS, load_truth  # noqa: E402

log = logging.getLogger("feeder")
DMI_PARAMS = ["precip_past1min", "precip_past10min", "temp_dry", "humidity", "wind_speed", "wind_dir", "cloud_cover"]


class Client:
    """Same interface for HTTP and in-process use."""

    def __init__(self, api: str | None, cfg):
        self.api = api.rstrip("/") if api else None
        self.svc = None if api else NowcastService(cfg)

    def post(self, recs):
        if self.svc:
            return self.svc.ingest(recs)
        import requests

        return requests.post(f"{self.api}/observations", json=recs, timeout=30).json()

    def forecast(self, now=None):
        if self.svc:
            return self.svc.forecast(None, now)
        import requests

        return requests.get(f"{self.api}/forecast", params={"now": now} if now else None, timeout=30).json()


def replay(cfg, args):
    client = Client(args.api, cfg)
    q = load_quality(cfg)
    start = pd.Timestamp(args.start, tz="UTC")
    stations = args.stations or sorted(q[(q.year == start.year) & q.ok].station_id.unique())
    hist = int(cfg.deploy.history_minutes)
    t0, t1 = start - pd.Timedelta(minutes=hist), start + pd.Timedelta(minutes=args.minutes + 1)
    data = {s: load_truth(cfg, s, t0, t1, ["p1_obs", "p10_obs", "precip", "valid"] + MET_COLS) for s in stations}
    # warm-up: send the history in one batch
    warm = [r for s, d in data.items() for r in records_from_truth(d[d.index < start], s)]
    client.post(warm)
    log.info("warm-up: %d records for %d stations", len(warm), len(stations))
    rows = []
    for i in range(args.minutes):
        now = start + pd.Timedelta(minutes=i)
        recs = [r for s, d in data.items() for r in records_from_truth(d.loc[[now]] if now in d.index else d.iloc[:0], s)]
        tic = time.perf_counter()
        client.post(recs)
        fc = client.forecast(now.isoformat())
        ms = (time.perf_counter() - tic) * 1000
        for f in fc:
            nxt = now + pd.Timedelta(minutes=1)
            d = data[f["station_id"]]
            obs = d.loc[nxt, "precip"] if nxt in d.index and d.loc[nxt, "valid"] > 0 else np.nan
            rows.append({**{k: f[k] for k in ("station_id", "issue_time", "p_rain", "rain_mm_next_minute")},
                         "observed_next_minute": obs, "latency_ms": ms})
        if fc and (i % 10 == 0 or i == args.minutes - 1):
            wet = [f for f in fc if f["p_rain"] > 0.5]
            log.info("%s  %d stations  %.0f ms  P(rain)>0.5 at %d stations", now, len(fc), ms, len(wet))
        if args.speed > 0:
            time.sleep(max(0.0, 60.0 / args.speed - (time.perf_counter() - tic)))
    df = pd.DataFrame(rows)
    out = cfg.path("deploy", "replay_log.csv", results=True, mkdir=True)
    df.to_csv(out, index=False)
    ok = df.dropna(subset=["observed_next_minute"])
    summary = {
        "forecasts": len(df), "scored": len(ok),
        "mae_mm": float((ok.rain_mm_next_minute - ok.observed_next_minute).abs().mean()) if len(ok) else None,
        "latency_ms_p50": float(df.latency_ms.median()), "latency_ms_p99": float(df.latency_ms.quantile(0.99)),
    }
    (out.parent / "replay_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("replay summary: %s", summary)


def fetch_dmi(cfg, station: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Observations from the DMI metObs API (open data; api-key added if DMI_API_KEY is set)."""
    import requests

    params = {"stationId": station, "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}", "limit": 10000}
    if os.environ.get("DMI_API_KEY"):
        params["api-key"] = os.environ["DMI_API_KEY"]
    r = requests.get(cfg.deploy.dmi_url, params=params, timeout=30)
    r.raise_for_status()
    by_time: dict[str, dict] = {}
    for f in r.json().get("features", []):
        p = f["properties"]
        if p.get("parameterId") in DMI_PARAMS:
            rec = by_time.setdefault(p["observed"], {"station_id": station, "time": p["observed"]})
            rec[p["parameterId"]] = p["value"]
    return list(by_time.values())


def live(cfg, args):
    client = Client(args.api, cfg)
    hist = int(cfg.deploy.history_minutes)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    for s in args.stations:
        client.post(fetch_dmi(cfg, s, now - pd.Timedelta(minutes=hist), now))
    log.info("warm-up done for %s", args.stations)
    while True:
        now = pd.Timestamp.now(tz="UTC").floor("min")
        for s in args.stations:
            try:
                client.post(fetch_dmi(cfg, s, now - pd.Timedelta(minutes=15), now))
            except Exception as exc:  # noqa: BLE001
                log.warning("DMI fetch failed for %s: %s", s, exc)
        for f in client.forecast():
            log.info("%s %s P(rain)=%.2f  %.2f mm/h  38GHz/1km: %.2f dB", f["valid_time"], f["station_id"], f["p_rain"],
                     f["rain_rate_mmh"], f["attenuation_db"].get("38GHz_1km_V", float("nan")))
        time.sleep(max(1.0, 60 - (pd.Timestamp.now(tz="UTC") - now).total_seconds()))


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["replay", "live"])
    ap.add_argument("--config", default=os.environ.get("RAINNOW_CONFIG"))
    ap.add_argument("--api", default=None)
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--start", default="2025-07-10T06:00")
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--speed", type=float, default=0, help="minutes per real minute (0 = as fast as possible)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    (replay if args.mode == "replay" else live)(cfg, args)


if __name__ == "__main__":
    main()
