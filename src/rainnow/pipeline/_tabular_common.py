"""Fit / predict loop shared by all tabular model steps."""
from __future__ import annotations

import logging
import time

import numpy as np

from ..dataset import model_dir, pred_frame, write_preds
from .step05_tabular import iter_tabular, load_tabular, tab_dir, xyw

log = logging.getLogger(__name__)


def load_train_val(cfg, bench):
    tr = load_tabular(cfg, bench, "train", max_rows=int(cfg.sampling.max_train_rows))
    va = load_tabular(cfg, bench, "val", sampled=True)
    return tr, va


def predict_split(cfg, model, bench_data: str, split: str):
    frames = []
    for stem, df in iter_tabular(cfg, bench_data, split):
        if df.empty:
            continue
        X, _, _ = xyw(df)
        out = model.predict(X)
        mask = np.ones(len(df), bool)
        frames.append(pred_frame(df, stem.split("_", 1)[0], mask,
                                 p_rain=out["p_rain"], y_hat=out["y_hat"], q90=out["q90"]))
    return frames


def fit_and_predict(cfg, bench: str, name: str, model, tr, va) -> None:
    t0 = time.time()
    X, y, w = xyw(tr)
    Xv, yv, wv = xyw(va)
    model.fit(X, y, w, Xv, yv, wv)
    model.save(model_dir(cfg, bench, name))
    for split in ("val", "test"):
        write_preds(cfg, bench, name, split, predict_split(cfg, model, bench, split))
    # models trained on the long station record (A) are also scored on the radar period (B)
    if bench == "A":
        for split in ("val", "test"):
            if tab_dir(cfg, "B", split).exists() and any(tab_dir(cfg, "B", split).glob("*.parquet")):
                write_preds(cfg, "B", f"A__{name}", split, predict_split(cfg, model, "B", split))
    log.info("%s/%s trained + predicted in %.1f min", bench, name, (time.time() - t0) / 60)
    return model
