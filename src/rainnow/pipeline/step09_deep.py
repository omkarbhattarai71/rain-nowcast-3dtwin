"""Step 09 - deep sequence models (priority track B: SAMBA; comparison: Mamba, S4D, GRU, TCN, PatchTST).

Benchmark A: every model in cfg.deep.models on station + neighbour channels (also scored on B).
Benchmark B: cfg.deep.models_B, each trained with station-only channels (_B_station) and with radar
channels (_B_radar). RAINNOW_VARIANT=_B_station or _B_radar restricts a job to one variant, so the two
variants can train in parallel on separate GPUs.

Built for 12 h cluster jobs: training is deadline-aware and resumable, and predictions are written per
station-year (see models/deep/train.py). A job stopped by the deadline exits with an error; submitting
it again continues where it stopped. Finished models are skipped before any data is loaded.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil

from ..config import set_seed
from ..dataset import load_part, model_dir, preds_path, split_parts, write_preds
from ..models.deep.data import WindowStore, deep_channels, fit_scaler
from ..models.deep.train import DeadlineReached, predict_store, train_model
from . import Isolated

log = logging.getLogger(__name__)


def _frames(cfg, parts, bench):
    return [load_part(cfg, p, bench) for p in parts]


def _outputs(cfg, bench, full):
    out = [preds_path(cfg, bench, full, "val"), preds_path(cfg, bench, full, "test")]
    if bench == "A" and split_parts(cfg, "B", "test"):
        out += [preds_path(cfg, "B", f"A__{full}", "val"), preds_path(cfg, "B", f"A__{full}", "test")]
    return out


def _prepare_model_dir(cfg, bench, full, force):
    d = model_dir(cfg, bench, full)
    meta = d / "meta.json"
    finished = meta.exists() and json.loads(meta.read_text()).get("finished")
    if force:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    if force or not finished:
        # predictions belong to a model version: a model that (re)trains must not leave old
        # prediction files or shards behind (they would be mistaken for its new results)
        shutil.rmtree(d / "pred_shards", ignore_errors=True)
        for p in _outputs(cfg, bench, full):
            p.unlink(missing_ok=True)
    return d


def run(cfg, args) -> None:
    iso = Isolated("step09")
    force = getattr(args, "force", False)
    only_variant = os.environ.get("RAINNOW_VARIANT")
    val_frac = float(cfg.deep.get("val_pred_day_frac", 1.0))
    for bench in args.bench:
        names = list(cfg.deep.models if bench == "A" else cfg.deep.models_B)
        if args.models:
            names = [n for n in names if n in args.models]
        suffixes = [""] if bench == "A" else ["_B_station", "_B_radar"]
        if only_variant and bench == "B":
            suffixes = [s for s in suffixes if s == only_variant]
        todo = {s: [n for n in names if force or not all(p.exists() for p in _outputs(cfg, bench, f"{n}{s}"))]
                for s in suffixes}
        for s in suffixes:
            for n in set(names) - set(todo[s]):
                log.info("%s/%s%s already has all predictions - skipping (use --force to retrain)", bench, n, s)
        if not any(todo.values()):
            continue
        parts = {s: split_parts(cfg, bench, s) for s in ("train", "val", "test")}
        if not all(parts.values()):
            log.warning("bench %s: missing split parts %s", bench, {k: len(v) for k, v in parts.items()})
            continue
        frames = {s: _frames(cfg, parts[s], bench) for s in parts}
        channel_sets = {"": deep_channels("A")} if bench == "A" else \
            {"_B_station": deep_channels("A"), "_B_radar": deep_channels("B", frames["train"])}
        for suffix, names_s in todo.items():
            if not names_s:
                continue
            channels = channel_sets[suffix]
            scaler = fit_scaler(frames["train"], channels)
            st_tr = WindowStore.build(cfg, parts["train"], bench, "train", channels, scaler, True, frames["train"])
            st_vs = WindowStore.build(cfg, parts["val"], bench, "val", channels, scaler, True, frames["val"])
            st_ve = WindowStore.build(cfg, parts["val"], bench, "val", channels, scaler, False, frames["val"],
                                      day_frac=val_frac)
            st_te = WindowStore.build(cfg, parts["test"], bench, "test", channels, scaler, False, frames["test"])
            transfer = _transfer_stores(cfg, channels, scaler, val_frac) if bench == "A" else {}
            n_pred = len(st_ve) + len(st_te) + sum(len(t) for t in transfer.values())
            for name in names_s:
                set_seed(int(cfg.seed))
                full = f"{name}{suffix}"
                with iso.model(f"{bench}/{full}"):
                    if len(st_tr) == 0 or len(st_vs) == 0:
                        raise RuntimeError(f"no training/validation windows ({len(st_tr)}/{len(st_vs)})")
                    out = _prepare_model_dir(cfg, bench, full, force)
                    meta = {"bench": bench, "channels": channels, "scaler": scaler, "window": int(cfg.deep.window)}
                    net = train_model(name, cfg, st_tr, st_vs, out, meta, pred_windows=n_pred, rate_store=st_te)
                    shards = out / "pred_shards"
                    jobs = [(bench, full, "val", st_ve), (bench, full, "test", st_te)] + \
                           [("B", f"A__{full}", sp, st) for sp, st in transfer.items()]
                    for b, m, sp, st in jobs:
                        if preds_path(cfg, b, m, sp).exists() and not force:
                            continue
                        df = predict_store(net, st, cfg, name, shard_dir=shards / f"{b}_{sp}")
                        write_preds(cfg, b, m, sp, [df])
                    del net
                gc.collect()
            del st_tr, st_vs, st_ve, st_te, transfer
            gc.collect()
        (cfg.path(f"bench{bench}", "models", results=True, mkdir=True) / f"deep_channels{only_variant or ''}.json") \
            .write_text(json.dumps(channel_sets, indent=1))
    iso.finish()


def _transfer_stores(cfg, channels, scaler, val_frac) -> dict:
    out = {}
    for split in ("val", "test"):
        parts = split_parts(cfg, "B", split)
        if parts:
            out[split] = WindowStore.build(cfg, parts, "B", split, channels, scaler, False,
                                           day_frac=val_frac if split == "val" else 1.0)
    return out


__all__ = ["run", "DeadlineReached"]
