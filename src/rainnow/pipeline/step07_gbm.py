"""Step 07 - LightGBM models, feature-group ablations and SHAP feature importance.

Benchmark A: lgbm_hurdle, lgbm_tweedie, lgbm_supervisor_features, gbm_abl_* (groups P/PT/PM/PN)
Benchmark B: the same hurdle model with station-only / +radar / +nowcast / all features (RQ3).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..dataset import model_dir
from ..features import feature_groups
from ..models import gbm
from . import Isolated
from ._tabular_common import fit_and_predict, load_train_val
from .step05_tabular import load_tabular, xyw

log = logging.getLogger(__name__)

B_VARIANTS = {
    "lgbm_B_station": ["P", "M", "T", "N"],
    "lgbm_B_radar": ["P", "M", "T", "N", "R"],
    "lgbm_B_nowcast": ["P", "M", "T", "N", "S"],
    "lgbm_B_all": ["P", "M", "T", "N", "R", "S"],
}


def run(cfg, args) -> None:
    iso = Isolated("step07")
    for bench in args.bench:
        try:
            tr, va = load_train_val(cfg, bench)
        except FileNotFoundError as exc:
            log.warning("bench %s: %s", bench, exc)
            continue
        if bench == "A":
            names = list(cfg.models.gbm.variants) + list(cfg.models.gbm.ablation_groups)
            models = {n: gbm.build(n, cfg) for n in names}
        else:
            models = {}
            for n, groups in B_VARIANTS.items():
                m = gbm.LGBMHurdle(cfg, groups=groups, quantile=(n == "lgbm_B_all"))
                m.name = n
                models[n] = m
        if args.models:
            models = {k: v for k, v in models.items() if k in args.models}
        for name, model in models.items():
            with iso.model(f"{bench}/{name}"):
                fit_and_predict(cfg, bench, name, model, tr, va)
            if name in ("lgbm_hurdle", "lgbm_B_all") and f"{bench}/{name}" not in iso.failed:
                try:
                    shap_report(cfg, bench, name, model)
                except Exception:  # noqa: BLE001  (explanations are optional)
                    log.exception("SHAP report for %s/%s failed", bench, name)
        del tr, va
    iso.finish()


def shap_report(cfg, bench: str, name: str, model) -> None:
    """Mean |SHAP| per feature and per group, separately for onset (dry now) and wet-now rows."""
    te = load_tabular(cfg, bench, "test")
    n = min(len(te), int(cfg.models.gbm.shap_rows))
    # oversample wet / onset rows so both regimes are represented
    wet_next = te["y"] > 0
    pick = pd.concat([te[wet_next].sample(min(wet_next.sum(), n // 2), random_state=0),
                      te[~wet_next].sample(min((~wet_next).sum(), n // 2), random_state=0)])
    X, y, _ = xyw(pick)
    contrib = model.contributions(X)[:, :-1]
    feats = model.features
    regimes = {"all": np.ones(len(pick), bool), "dry_now": pick["rain_now"].to_numpy() == 0,
               "wet_now": pick["rain_now"].to_numpy() > 0}
    rows = []
    for reg, m in regimes.items():
        if not m.any():
            continue
        imp = np.abs(contrib[m]).mean(0)
        rows += [{"regime": reg, "feature": f, "mean_abs_shap": float(v)} for f, v in zip(feats, imp)]
    df = pd.DataFrame(rows)
    groups = feature_groups(feats)
    inv = {f: g for g, fs in groups.items() for f in fs}
    df["group"] = df["feature"].map(inv)
    out = model_dir(cfg, bench, name)
    df.sort_values(["regime", "mean_abs_shap"], ascending=[True, False]).to_csv(out / "shap_features.csv", index=False)
    df.groupby(["regime", "group"])["mean_abs_shap"].sum().reset_index().to_csv(out / "shap_groups.csv", index=False)
    log.info("SHAP top features (all):\n%s",
             df[df.regime == "all"].nlargest(12, "mean_abs_shap")[["feature", "mean_abs_shap"]].to_string(index=False))
