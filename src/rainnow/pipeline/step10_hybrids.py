"""Step 10 - hybrid models built from other models' predictions (all fitted out-of-sample).

The validation period is split in two halves by day: hybrids are *fitted* on the first half,
their validation predictions (used later for threshold tuning) come from the second half, and
the test split is never used for fitting.

* hyb_gbm_kalman   GBM + residual Kalman (AR(1)+noise, MLE) - the supervisors' residual-SSM idea, fixed
* hyb_regime       regime switch: best detector when dry now, best intensity model when wet now
* hyb_stack        stacking: logistic regression on calibrated probabilities, NNLS on amounts
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression

from ..dataset import index_path, preds_path, write_preds
from ..metrics import categorical_scores, contingency, tune_threshold
from ..models.base import ProbCalibrator
from ..models.ssm_classic import fit_residual_ar

log = logging.getLogger(__name__)

BASE_GBM = {"A": "lgbm_hurdle", "B": "lgbm_B_all"}
CANDIDATES = {
    "A": ["lgbm_hurdle", "samba", "kalman_x", "hmm_switch", "gru", "mamba", "persistence"],
    "B": ["lgbm_B_all", "samba_B_radar", "lgbm_B_station", "samba_B_station", "A__lgbm_hurdle", "A__samba"],
}


def _load(cfg, bench, model, split, idx):
    p = preds_path(cfg, bench, model, split)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return idx.merge(df, on=["station_id", "time"], how="inner")


def run(cfg, args) -> None:
    for bench in args.bench:
        try:
            idx = {s: pd.read_parquet(index_path(cfg, bench, s)) for s in ("val", "test")}
        except FileNotFoundError:
            continue
        days = np.sort(idx["val"]["day"].unique())
        fit_days = set(days[: len(days) // 2])
        cands = {}
        for m in CANDIDATES[bench]:
            v, t = _load(cfg, bench, m, "val", idx["val"]), _load(cfg, bench, m, "test", idx["test"])
            if v is not None and t is not None:
                cands[m] = (v, t)
        if not cands:
            log.warning("bench %s: no candidate predictions for hybrids", bench)
            continue
        log.info("bench %s hybrid candidates: %s", bench, list(cands))
        base = BASE_GBM[bench]
        if base in cands:
            _gbm_kalman(cfg, bench, *cands[base], fit_days)
        _regime_and_stack(cfg, bench, cands, fit_days)


def _resid_corrections(df: pd.DataFrame, model) -> np.ndarray:
    """Kalman forecast of the next residual for each row (causal: uses residuals up to t-1)."""
    out = np.zeros(len(df))
    for sid, g in df.groupby("station_id"):
        g = g.sort_values("time")
        full = pd.date_range(g["time"].min(), g["time"].max(), freq="1min", tz="UTC")
        r = (g.set_index("time")["y_true"] - g.set_index("time")["y_hat"]).reindex(full).fillna(0.0)
        pred_next = model.score(r.to_numpy())
        corr = pd.Series(np.r_[0.0, pred_next[:-1]], index=full)
        out[g.index.to_numpy()] = corr.reindex(g["time"]).to_numpy()
    return out


def _gbm_kalman(cfg, bench, val, test, fit_days):
    vf = val[val["day"].isin(fit_days)].reset_index(drop=True)
    vs = val[~val["day"].isin(fit_days)].reset_index(drop=True)
    blocks = []
    for _, g in vf.groupby("station_id"):
        g = g.sort_values("time")
        full = pd.date_range(g["time"].min(), g["time"].max(), freq="1min", tz="UTC")
        blocks.append((g.set_index("time")["y_true"] - g.set_index("time")["y_hat"]).reindex(full).fillna(0).to_numpy())
    model = fit_residual_ar(blocks)
    cal = None
    out = {}
    for name, df in (("fit", vf), ("val", vs), ("test", test.reset_index(drop=True))):
        yh = np.clip(df["y_hat"].to_numpy() + _resid_corrections(df, model), 0, None)
        if cal is None:
            cal = ProbCalibrator().fit(yh, df["y_true"].to_numpy() > 0)
        out[name] = df.assign(y_hat=yh, p_rain=cal(yh))
    for split in ("val", "test"):
        write_preds(cfg, bench, "hyb_gbm_kalman", split, [out[split][["station_id", "time", "p_rain", "y_hat"]]])


def _regime_and_stack(cfg, bench, cands, fit_days):
    names = list(cands)
    val = {m: v for m, (v, _) in cands.items()}
    test = {m: t for m, (_, t) in cands.items()}
    keys = ["station_id", "time"]
    base_v = val[names[0]][keys + ["y_true", "rain_now", "day"]]
    base_t = test[names[0]][keys + ["y_true", "rain_now", "day"]]
    for m in names:
        base_v = base_v.merge(val[m][keys + ["p_rain", "y_hat"]].rename(columns={"p_rain": f"{m}__p", "y_hat": f"{m}__y"}), on=keys)
        base_t = base_t.merge(test[m][keys + ["p_rain", "y_hat"]].rename(columns={"p_rain": f"{m}__p", "y_hat": f"{m}__y"}), on=keys)
    fit = base_v[base_v["day"].isin(fit_days)]
    sel = base_v[~base_v["day"].isin(fit_days)]
    cals = {m: ProbCalibrator().fit(fit[f"{m}__p"].fillna(0), fit["y_true"] > 0) for m in names}

    # regime switch
    dry_f, wet_f = fit["rain_now"] == 0, fit["rain_now"] > 0
    best_dry, best_wet, sd, sw = names[0], names[0], -1, np.inf
    for m in names:
        s = cals[m](fit.loc[dry_f, f"{m}__p"].fillna(0))
        obs = fit.loc[dry_f, "y_true"] > 0
        thr = tune_threshold(s, obs)
        csi = categorical_scores(contingency(obs, s >= thr))["csi"]
        rmse = float(np.sqrt(np.mean((fit.loc[wet_f, f"{m}__y"] - fit.loc[wet_f, "y_true"]) ** 2)))
        if csi > sd:
            best_dry, sd = m, csi
        if rmse < sw:
            best_wet, sw = m, rmse
    log.info("regime switch: dry-now -> %s (CSI %.3f), wet-now -> %s (RMSE %.4f)", best_dry, sd, best_wet, sw)

    def regime(df):
        wet = df["rain_now"].to_numpy() > 0
        p = np.where(wet, cals[best_wet](df[f"{best_wet}__p"].fillna(0)), cals[best_dry](df[f"{best_dry}__p"].fillna(0)))
        y = np.where(wet, df[f"{best_wet}__y"], df[f"{best_dry}__y"])
        return df[keys].assign(p_rain=p, y_hat=y)

    # stacking
    def zp(df):
        cols = [cals[m](df[f"{m}__p"].fillna(0)) for m in names] + [(df["rain_now"] > 0).astype(float)]
        return np.column_stack(cols)

    def zy(df):
        return np.column_stack([df[f"{m}__y"].fillna(0) for m in names])

    lr = LogisticRegression(max_iter=500).fit(zp(fit), fit["y_true"] > 0)
    coef, _ = nnls(zy(fit), fit["y_true"].to_numpy())
    log.info("stacking NNLS weights: %s", dict(zip(names, np.round(coef, 3))))

    def stack(df):
        return df[keys].assign(p_rain=lr.predict_proba(zp(df))[:, 1], y_hat=zy(df) @ coef)

    for split, df in (("val", sel), ("test", base_t)):
        write_preds(cfg, bench, "hyb_regime", split, [regime(df)])
        write_preds(cfg, bench, "hyb_stack", split, [stack(df)])
