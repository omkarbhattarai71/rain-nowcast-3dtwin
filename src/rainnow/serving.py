"""Real-time nowcasting service core (used by deploy/app.py and deploy/feeder.py).

Raw observations arrive per station and minute; the service keeps a rolling buffer and, on
request, recomputes the causal channels with *exactly the same code* used for training
(channels.build_all_channels -> features.build_features / deep transform). No training/serving skew.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .attenuation import scenario_attenuation
from .channels import build_all_channels
from .features import build_features
from .stations import load_station_table, norm_id
from .truth import MET_COLS

log = logging.getLogger(__name__)

OBS_COLS = ["p1_obs", "p10_obs"] + MET_COLS
FIELD_MAP = {"precip_past1min": "p1_obs", "precip_past10min": "p10_obs"}


class NowcastService:
    def __init__(self, cfg, model: str | None = None, bench: str | None = None):
        self.cfg = cfg
        self.bench = bench or cfg.deploy.benchmark
        self.stab = load_station_table(cfg)
        self.history = int(cfg.deploy.history_minutes)
        self.model_name = model or self._pick_model()
        self.kind, self.model, self.meta = self._load(self.model_name)
        self.buffers: dict[str, pd.DataFrame] = {}
        self.lock = threading.Lock()
        self.latencies = deque(maxlen=5000)
        log.info("serving model %s (%s) for benchmark %s", self.model_name, self.kind, self.bench)

    # ------------------------------------------------------------------ model
    def _pick_model(self) -> str:
        want = self.cfg.deploy.model
        if want and want != "auto":
            return want
        f = self.cfg.path(f"bench{self.bench}", "eval", "best_model.json", results=True)
        if f.exists():
            best = json.loads(f.read_text()).get("best_deployable")
            if best:
                return best
        return "lgbm_hurdle"

    def _load(self, name: str):
        d = self.cfg.path(f"bench{self.bench}", "models", name, results=True)
        if (d / "model.pt").exists():
            import torch

            from .models.deep.train import load_model

            torch.set_num_threads(2)
            torch.set_flush_denormal(True)        # avoid subnormal-float slowdowns in the SSM scan
            net, meta = load_model(d)
            return "deep", net, meta
        if (d / "model.joblib").exists():
            from .models.base import TabularModel

            return "tabular", TabularModel.load(d), {}
        raise FileNotFoundError(f"no trained model in {d}")

    # ------------------------------------------------------------------ ingest
    def ingest(self, records: list[dict]) -> int:
        """records: {station_id, time, precip_past1min?, precip_past10min?, temp_dry?, humidity?, ...}"""
        if not records:
            return 0
        df = pd.DataFrame(records).rename(columns=FIELD_MAP)
        df["station_id"] = norm_id(df["station_id"].astype(str))
        df = df[df["station_id"].isin(self.stab.index)]
        df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed").dt.floor("min")
        for c in OBS_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
        with self.lock:
            for sid, g in df.groupby("station_id"):
                new = g.groupby("time")[OBS_COLS].last().astype("float32")
                old = self.buffers.get(sid)
                buf = new if old is None else new.combine_first(old)
                cutoff = buf.index.max() - pd.Timedelta(minutes=self.history + 30)
                self.buffers[sid] = buf[buf.index >= cutoff].sort_index()
        return int(len(df))

    # ------------------------------------------------------------------ forecast
    def forecast(self, station_ids: list[str] | None = None, now=None) -> list[dict]:
        t0 = time.perf_counter()
        with self.lock:
            if not self.buffers:
                return []
            now = pd.Timestamp(now) if now is not None else max(b.index.max() for b in self.buffers.values())
            now = (now.tz_localize("UTC") if now.tzinfo is None else now).floor("min")
            grid = pd.date_range(now - pd.Timedelta(minutes=self.history - 1), now, freq="1min", tz="UTC", name="time")
            obs = {s: b.reindex(grid).astype("float32") for s, b in self.buffers.items()}
        targets = [norm_id(s) for s in station_ids] if station_ids else sorted(obs)
        targets = [s for s in targets if s in obs]
        if not targets:
            return []
        chans = build_all_channels(obs, targets, self.stab.loc[sorted(obs)], self.cfg)
        out = []
        for sid in targets:
            ch = chans[sid]
            p, y, q = self._predict_one(ch)
            last_p1 = obs[sid]["p1_obs"].last_valid_index()
            rec = {
                "station_id": sid, "issue_time": now.isoformat(),
                "valid_time": (now + pd.Timedelta(minutes=1)).isoformat(),
                "p_rain": round(float(p), 4), "rain_mm_next_minute": round(float(y), 4),
                "rain_rate_mmh": round(60 * float(y), 3),
                "rain_mm_q90": None if q is None or not np.isfinite(q) else round(float(q), 4),
                "attenuation_db": {s["name"]: round(float(scenario_attenuation(
                    y, s, float(self.stab.loc[sid, "lat"]), self.cfg.attenuation.rain_height_km)), 3)
                    for s in list(self.cfg.attenuation.terrestrial) + list(self.cfg.attenuation.earth_space)},
                "input_age_min": None if last_p1 is None else int((now - last_p1).total_seconds() // 60),
                "model": self.model_name,
            }
            if rec["input_age_min"] is None or rec["input_age_min"] > 10:
                rec["warning"] = "no recent 1-minute rain observation; forecast relies on 10-min/neighbour data"
            out.append(rec)
        self.latencies.append((time.perf_counter() - t0) * 1000)
        return out

    def _predict_one(self, ch: pd.DataFrame):
        if self.kind == "tabular":
            X = build_features(ch, self.cfg).iloc[[-1]]
            r = self.model.predict(X).iloc[0]
            return r["p_rain"], r["y_hat"], r["q90"]
        import torch

        from .models.deep.blocks import head_outputs
        from .models.deep.data import transform

        L = int(self.meta["window"])
        X = transform(ch.iloc[-L:], self.meta["channels"], self.meta["scaler"])[None]
        with torch.no_grad():
            p, y, q = head_outputs(*self.model(torch.from_numpy(X)))
        return float(p[0]), float(y[0]), float(q[0])

    def stats(self) -> dict:
        lat = np.array(self.latencies)
        return {"model": self.model_name, "kind": self.kind, "stations_buffered": len(self.buffers),
                "requests": len(self.latencies),
                "latency_ms_p50": round(float(np.percentile(lat, 50)), 2) if len(lat) else None,
                "latency_ms_p99": round(float(np.percentile(lat, 99)), 2) if len(lat) else None}


def records_from_truth(tr: pd.DataFrame, sid: str) -> list[dict]:
    """Turn stored raw observations (truth parquet columns) into API records (for replay)."""
    recs = []
    for t, r in tr.iterrows():
        rec = {"station_id": sid, "time": t.isoformat()}
        if pd.notna(r.get("p1_obs")):
            rec["precip_past1min"] = float(r["p1_obs"])
        if pd.notna(r.get("p10_obs")):
            rec["precip_past10min"] = float(r["p10_obs"])
        for c in MET_COLS:
            if pd.notna(r.get(c)):
                rec[c] = float(r[c])
        recs.append(rec)
    return recs


def default_model_dir(cfg, bench: str, name: str) -> Path:
    return cfg.path(f"bench{bench}", "models", name, results=True)
