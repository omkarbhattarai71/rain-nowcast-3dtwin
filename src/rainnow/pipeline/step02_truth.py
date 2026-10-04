"""Step 02 - validated 1-minute ground truth and station-year quality report."""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from ..config import Cfg
from ..stations import raw_station_ids, select_stations
from ..truth import build_truth, quality_by_year, read_overlap, read_raw_csv, station_year_ok, truth_path

log = logging.getLogger(__name__)


def _one(args):
    cfg_dict, sid = args
    cfg = Cfg.wrap(cfg_dict)
    raw_file = cfg.path("raw", "stations", f"{sid}{cfg.s3.station_csv_suffix}")
    raw = read_raw_csv(raw_file)
    ov = read_overlap(cfg.path("raw", "overlap", sid))
    tr = build_truth(raw, cfg, ov)
    p = truth_path(cfg, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    tr.reset_index().to_parquet(p, index=False)
    q = quality_by_year(tr)
    q.insert(0, "station_id", sid)
    return q


def run(cfg, args) -> None:
    ids = raw_station_ids(cfg.path("raw", "stations"), cfg.s3.station_csv_suffix)
    sids = select_stations(cfg, ids)
    log.info("building truth for %d stations", len(sids))
    workers = max(1, min(int(getattr(args, "workers", 0) or 6), len(sids)))
    with ProcessPoolExecutor(workers) as ex:
        qs = list(ex.map(_one, [(cfg.to_dict(), s) for s in sids]))
    q = pd.concat(qs, ignore_index=True)
    q["ok"] = [station_year_ok(r, cfg) for r in q.itertuples()]
    q.to_csv(cfg.path("interim", "truth", "quality.csv"), index=False)
    out = cfg.path("data_quality", results=True, mkdir=True)
    out.mkdir(parents=True, exist_ok=True)
    q.to_csv(out / "quality_station_year.csv", index=False)
    summary = q.groupby("year").agg(
        stations=("station_id", "nunique"), ok_station_years=("ok", "sum"),
        median_rain_coverage=("rain_coverage", "median"), median_mass_ratio=("mass_ratio", "median"),
        wet_minutes=("wet_minutes", "sum"),
    ).reset_index()
    summary.to_csv(out / "quality_by_year.csv", index=False)
    log.info("quality summary:\n%s", summary.to_string(index=False))
    _plot(q, out)


def _plot(q: pd.DataFrame, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = q.pivot_table(index="station_id", columns="year", values="rain_coverage")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.18 * len(piv))))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=6)
    ax.set_title("Share of rainy 10-min windows with valid 1-min truth")
    fig.colorbar(im, ax=ax, label="rain coverage")
    fig.tight_layout()
    fig.savefig(out / "rain_coverage_heatmap.png", dpi=150)
    plt.close(fig)
