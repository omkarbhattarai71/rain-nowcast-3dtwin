"""FastAPI service: real-time 1-minute-ahead rain + attenuation forecasts.

Run:  uvicorn deploy.app:app --app-dir src --host 0.0.0.0 --port 8000
Env:  RAINNOW_CONFIG (e.g. smoke), RAINNOW_MODEL (model name; default = best deployable from evaluation),
      RAINNOW_RESULTS / RAINNOW_DATA (artifact locations)

Endpoints
  GET  /health                  model + buffer status
  POST /observations            list of raw station observations (DMI parameter names)
  GET  /forecast[?station_id=]  next-minute forecast for one or all buffered stations
  GET  /metrics                 request count and latency percentiles
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from rainnow.config import load_config, setup_logging  # noqa: E402
from rainnow.serving import NowcastService  # noqa: E402

setup_logging()
app = FastAPI(title="rainnow - 1-minute rainfall nowcast", version="1.0.0")
_service: NowcastService | None = None


class Observation(BaseModel):
    station_id: str
    time: str
    precip_past1min: float | None = None
    precip_past10min: float | None = None
    temp_dry: float | None = None
    humidity: float | None = None
    wind_speed: float | None = None
    wind_dir: float | None = None
    cloud_cover: float | None = None


def service() -> NowcastService:
    global _service
    if _service is None:
        cfg = load_config(os.environ.get("RAINNOW_CONFIG"))
        _service = NowcastService(cfg, model=os.environ.get("RAINNOW_MODEL"))
    return _service


@app.on_event("startup")
def _startup() -> None:
    service()


@app.get("/health")
def health():
    s = service()
    return {"status": "ok", **s.stats()}


@app.post("/observations")
def observations(obs: list[Observation]):
    n = service().ingest([o.model_dump(exclude_none=True) for o in obs])
    return {"accepted": n}


@app.get("/forecast")
def forecast(station_id: str | None = None, now: str | None = None):
    res = service().forecast([station_id] if station_id else None, now)
    if station_id and not res:
        raise HTTPException(404, f"no buffered observations for station {station_id}")
    return res


@app.get("/metrics")
def metrics():
    return service().stats()
