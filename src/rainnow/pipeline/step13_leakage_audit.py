"""Step 13 - quantify how the supervisors' pipeline inflates the reported skill.

1. Ground truth: rebuild the supervisors' step-2 target (missing 1-min -> 0; 10-min-only stations ->
   uniform spreading) for the benchmark-A test station-years and compare it with ours.
2. Target leakage: persistence on a uniformly spread 10-min target vs. on real 1-min data.
3. Test-set tuning: LightGBM early-stopped on the *test* split (as in step 5/11) vs. on validation.
4. Radar lookup audit (from step 04), if available.
"""
from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..dataset import period, split_parts
from ..metrics import categorical_scores, contingency, tune_threshold
from ..truth import clean_raw, load_truth, read_raw_csv
from .step05_tabular import load_tabular, xyw

log = logging.getLogger(__name__)


def supervisor_truth(raw: pd.DataFrame, grid: pd.DatetimeIndex, use_1min: bool) -> pd.Series:
    """The original step-2 logic: fill missing 1-min with 0, or spread 10-min totals uniformly."""
    if use_1min and "precip_past1min" in raw and raw["precip_past1min"].notna().any():
        return raw["precip_past1min"].reindex(grid).fillna(0.0)
    s = pd.Series(0.0, index=grid)
    p10 = raw["precip_past10min"].dropna()
    for t, v in p10.items():
        block = pd.date_range(t - pd.Timedelta(minutes=9), t, freq="1min", tz="UTC").intersection(grid)
        if len(block):
            s.loc[block] = float(v) / len(block)
    return s


def _persistence_scores(y_next: np.ndarray, y_now: np.ndarray) -> dict:
    sc = categorical_scores(contingency(y_next > 0, y_now > 0))
    return {"csi": sc["csi"], "pod": sc["pod"], "far": sc["far"],
            "rmse": float(np.sqrt(np.mean((y_next - y_now) ** 2)))}


def run(cfg, args) -> None:
    out = cfg.path("leakage_audit", results=True, mkdir=True)
    out.mkdir(parents=True, exist_ok=True)
    s, e = period(cfg, "A", "test")
    rows = []
    for part in split_parts(cfg, "A", "test"):
        raw = clean_raw(read_raw_csv(cfg.path("raw", "stations", f"{part.sid}{cfg.s3.station_csv_suffix}")), cfg)
        raw = raw[(raw.index >= part.start - pd.Timedelta(minutes=10)) & (raw.index <= part.end + pd.Timedelta(minutes=10))]
        tr = load_truth(cfg, part.sid, part.start, part.end + pd.Timedelta(minutes=1), ["precip", "valid", "p1_obs"])
        grid = tr.index
        sup = supervisor_truth(raw, grid, use_1min=True).to_numpy()
        spread = supervisor_truth(raw, grid, use_1min=False).to_numpy()
        ours = tr["precip"].to_numpy()
        valid = tr["valid"].to_numpy() > 0
        ok = valid[1:] & valid[:-1]
        wet_ours = valid & (np.nan_to_num(ours) > 0)
        rows.append({
            "station_id": part.sid, "minutes": len(grid), "valid_frac": float(valid.mean()),
            "our_wet_minutes": int(wet_ours.sum()),
            "wet_minutes_labelled_dry_by_supervisor_truth": int((wet_ours & (sup == 0)).sum()),
            "supervisor_minutes_filled_with_zero": int(tr["p1_obs"].isna().sum()),
            "mm_ours": float(np.nansum(ours[valid])), "mm_supervisor": float(sup.sum()),
            **{f"pers_ours_{k}": v for k, v in _persistence_scores(ours[1:][ok], ours[:-1][ok]).items()},
            **{f"pers_supervisor_{k}": v for k, v in _persistence_scores(sup[1:], sup[:-1]).items()},
            **{f"pers_spread10_{k}": v for k, v in _persistence_scores(spread[1:], spread[:-1]).items()},
        })
    truth_df = pd.DataFrame(rows)
    truth_df.to_csv(out / "truth_comparison.csv", index=False)
    summ = {
        "stations": int(len(truth_df)),
        "share_of_real_wet_minutes_labelled_dry": float(truth_df["wet_minutes_labelled_dry_by_supervisor_truth"].sum()
                                                       / max(truth_df["our_wet_minutes"].sum(), 1)),
        "persistence_csi_ours": float(truth_df["pers_ours_csi"].median()),
        "persistence_csi_supervisor_truth": float(truth_df["pers_supervisor_csi"].median()),
        "persistence_csi_uniform_10min_spread": float(truth_df["pers_spread10_csi"].median()),
    }
    summ |= _test_tuning_effect(cfg)
    rad = cfg.path("radar", "lookup_audit.csv", results=True)
    if rad.exists():
        summ["radar_lookup_audit"] = pd.read_csv(rad).to_dict(orient="records")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    log.info("leakage audit:\n%s", json.dumps(summ, indent=1, default=float))


def _test_tuning_effect(cfg) -> dict:
    """Early stopping on test (supervisor protocol) vs. on validation: optimistic bias of test CSI."""
    try:
        tr = load_tabular(cfg, "A", "train", max_rows=min(int(cfg.sampling.max_train_rows), 5_000_000))
        va = load_tabular(cfg, "A", "val", sampled=True)
        te = load_tabular(cfg, "A", "test")
    except FileNotFoundError:
        return {}
    X, y, w = xyw(tr)
    Xv, yv, wv = xyw(va)
    Xt, yt, _ = xyw(te)
    params = {**cfg.models.gbm.params, "objective": "binary", "seed": cfg.seed}
    res = {}
    for name, (Xe, ye, we) in {"val": (Xv, yv, wv), "test": (Xt, yt, np.ones(len(yt)))}.items():
        b = lgb.train(params, lgb.Dataset(X, (y > 0).astype(float), weight=w), int(cfg.models.gbm.num_boost_round),
                      valid_sets=[lgb.Dataset(Xe, (ye > 0).astype(float), weight=we)],
                      callbacks=[lgb.early_stopping(int(cfg.models.gbm.early_stopping), verbose=False)])
        p = b.predict(Xt, num_iteration=b.best_iteration)
        if name == "test":
            # supervisor protocol: threshold also picked on the evaluation data
            csi = max(categorical_scores(contingency(yt > 0, p >= t))["csi"] for t in np.linspace(0.05, 0.95, 37))
        else:
            vf = load_tabular(cfg, "A", "val")                    # full validation rows, as in step 11
            Xf, yf, _ = xyw(vf)
            thr = tune_threshold(b.predict(Xf, num_iteration=b.best_iteration), yf > 0)
            csi = categorical_scores(contingency(yt > 0, p >= thr))["csi"]
        res[f"lgbm_csi_test_when_tuned_on_{name}"] = float(csi)
        res[f"lgbm_iterations_when_tuned_on_{name}"] = int(b.best_iteration)
    return res
