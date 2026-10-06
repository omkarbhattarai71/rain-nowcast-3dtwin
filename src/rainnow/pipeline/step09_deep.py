"""Step 09 - deep sequence models (priority track B: SAMBA; comparison: Mamba, S4D, GRU, TCN, PatchTST).

Benchmark A: every model in cfg.deep.models on station + neighbour channels (also scored on B).
Benchmark B: cfg.deep.models_B, each trained with station-only channels and with radar channels.
"""
from __future__ import annotations

import gc
import json
import logging

from ..config import set_seed
from ..dataset import load_part, model_dir, preds_path, split_parts, write_preds
from ..models.deep.data import WindowStore, deep_channels, fit_scaler
from . import Isolated
from ..models.deep.train import predict_store, train_model

log = logging.getLogger(__name__)


def _frames(cfg, parts, bench):
    return [load_part(cfg, p, bench) for p in parts]


def run(cfg, args) -> None:
    iso = Isolated("step09")
    for bench in args.bench:
        parts = {s: split_parts(cfg, bench, s) for s in ("train", "val", "test")}
        if not all(parts.values()):
            log.warning("bench %s: missing split parts %s", bench, {k: len(v) for k, v in parts.items()})
            continue
        frames = {s: _frames(cfg, parts[s], bench) for s in parts}
        if bench == "A":
            variants = {"": deep_channels("A")}
            names = list(cfg.deep.models)
        else:
            variants = {"_B_station": deep_channels("A"), "_B_radar": deep_channels("B", frames["train"])}
            names = list(cfg.deep.models_B)
        if args.models:
            names = [n for n in names if n in args.models]
        if not names:
            continue
        for suffix, channels in variants.items():
            scaler = fit_scaler(frames["train"], channels)
            st_tr = WindowStore.build(cfg, parts["train"], bench, "train", channels, scaler, True, frames["train"])
            st_vs = WindowStore.build(cfg, parts["val"], bench, "val", channels, scaler, True, frames["val"])
            st_ve = WindowStore.build(cfg, parts["val"], bench, "val", channels, scaler, False, frames["val"])
            st_te = WindowStore.build(cfg, parts["test"], bench, "test", channels, scaler, False, frames["test"])
            transfer = _transfer_stores(cfg, channels, scaler) if bench == "A" else {}
            for name in names:
                set_seed(int(cfg.seed))
                full = f"{name}{suffix}"
                done = [preds_path(cfg, bench, full, "test")] + [preds_path(cfg, "B", f"A__{full}", sp) for sp in transfer]
                if all(p.exists() for p in done) and not getattr(args, "force", False):
                    log.info("%s/%s already has predictions - skipping (use --force to retrain)", bench, full)
                    continue
                with iso.model(f"{bench}/{full}"):
                    if len(st_tr) == 0 or len(st_vs) == 0:
                        raise RuntimeError(f"no training/validation windows ({len(st_tr)}/{len(st_vs)})")
                    out = model_dir(cfg, bench, full)
                    meta = {"bench": bench, "channels": channels, "scaler": scaler, "window": int(cfg.deep.window)}
                    net = train_model(name, cfg, st_tr, st_vs, out, meta)
                    write_preds(cfg, bench, full, "val", [predict_store(net, st_ve, cfg, name)])
                    write_preds(cfg, bench, full, "test", [predict_store(net, st_te, cfg, name)])
                    for split, st in transfer.items():
                        write_preds(cfg, "B", f"A__{full}", split, [predict_store(net, st, cfg, name)])
                    del net
                gc.collect()
            del st_tr, st_vs, st_ve, st_te, transfer
            gc.collect()
        (cfg.path(f"bench{bench}", "models", results=True, mkdir=True) / "deep_channels.json").write_text(
            json.dumps({k: v for k, v in variants.items()}, indent=1))
    iso.finish()


def _transfer_stores(cfg, channels, scaler) -> dict:
    out = {}
    for split in ("val", "test"):
        parts = split_parts(cfg, "B", split)
        if parts:
            out[split] = WindowStore.build(cfg, parts, "B", split, channels, scaler, False)
    return out
