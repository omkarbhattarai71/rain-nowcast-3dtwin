"""Training / inference loop shared by all deep sequence models.

Built for time-limited cluster jobs (Slurm, 12 h):
  * deadline-aware: the job's end time comes from RAINNOW_DEADLINE (unix seconds, set by the sbatch
    script). Training measures its own speed, reserves the time needed for the predictions, and plans
    the number of steps (and the cosine schedule) to fit;
  * resumable: the full training state (weights, optimiser, schedule, epoch, RNG) is saved to last.pt
    after every epoch, and predictions are written per station-year, so a job that is stopped
    continues where it stopped when it is submitted again;
  * warm start: if a run was killed before the first last.pt existed, its best weights (model.pt)
    are used as initialisation instead of starting from scratch.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .blocks import head_outputs, hurdle_loss
from .data import WindowStore
from .nets import build_net

log = logging.getLogger(__name__)

MARGIN_S = 15 * 60          # safety margin before the job deadline (writing files, Slurm teardown)
_CPU_READY = False


class DeadlineReached(RuntimeError):
    """Raised when the job must stop; everything done so far is saved and is resumed next time."""


def deadline() -> float | None:
    v = os.environ.get("RAINNOW_DEADLINE")
    return float(v) if v else None


def time_left() -> float:
    d = deadline()
    return math.inf if d is None else d - time.time()


def pick_device(cfg) -> torch.device:
    global _CPU_READY
    want = cfg.deep.device
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
        torch.backends.cuda.matmul.allow_tf32 = True      # TF32 tensor cores for matmuls (Ampere/Ada)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if not _CPU_READY:
        # configure the CPU thread pool exactly once per process: re-calling set_num_threads after
        # the OpenMP pool exists can deadlock (observed on Windows when training several models)
        torch.set_num_threads(int(cfg.deep.num_threads))
        # the SSM state decays through exp(dt*A); subnormal floats make CPU math up to ~100x slower
        torch.set_flush_denormal(True)
        _CPU_READY = True
    return torch.device("cpu")


def batch_size_for(cfg, name: str) -> int:
    by = cfg.deep.get("batch_size_by_model", {}) or {}
    return int(by.get(name, cfg.deep.batch_size))


def eval_batch_for(cfg, name: str) -> int:
    by = cfg.deep.get("eval_batch_size_by_model", {}) or {}
    return int(by.get(name, cfg.deep.eval_batch_size))


def _to(x, dev):
    return torch.from_numpy(x).to(dev, non_blocking=True)


def _amp(dev):
    use = dev.type == "cuda"
    return use, (torch.bfloat16 if use and torch.cuda.is_bf16_supported() else torch.float16)


@torch.no_grad()
def forward_rate(net, store: WindowStore, cfg, name: str, n_batches: int = 6) -> float:
    """Measured prediction throughput in windows per second (used to reserve prediction time)."""
    if len(store) == 0:
        return math.inf
    dev = next(net.parameters()).device
    net.eval()
    use_amp, amp_dtype = _amp(dev)
    bs = min(eval_batch_for(cfg, name), len(store))
    idx = np.arange(bs)
    times = []
    for i in range(n_batches + 1):
        X, _, _ = store.gather(idx)
        t0 = time.time()
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            out = net(_to(X, dev))
        out[0].float().cpu()
        if i:                                  # first batch includes warm-up / autotuning
            times.append(time.time() - t0)
    gather_t = time.time()
    store.gather(idx)
    gather_t = time.time() - gather_t
    return bs / (float(np.median(times)) + gather_t)


def train_model(name: str, cfg, train: WindowStore, val: WindowStore, out_dir: Path, meta: dict,
                pred_windows: int = 0, rate_store: WindowStore | None = None):
    """Train (or resume / skip if already trained). Returns the best network."""
    dev = pick_device(cfg)
    n_ch = train.X[0].shape[1]
    out_dir = Path(out_dir)
    ckpt_best, ckpt_last, meta_f = out_dir / "model.pt", out_dir / "last.pt", out_dir / "meta.json"

    net = build_net(name, n_ch, cfg).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    if meta_f.exists() and ckpt_best.exists() and json.loads(meta_f.read_text()).get("finished"):
        net.load_state_dict(torch.load(ckpt_best, map_location=dev))
        log.info("%s: training already finished - loading the best checkpoint", name)
        return net

    bs = batch_size_for(cfg, name)
    steps = min(int(cfg.deep.max_steps_per_epoch), max(1, len(train) // bs))
    planned_total = steps * int(cfg.deep.epochs)
    total = {"n": planned_total}                      # mutable: re-planned after measuring speed
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.deep.lr, weight_decay=cfg.deep.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total["n"]) / total["n"])))
    use_amp, amp_dtype = _amp(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    rng = np.random.default_rng(cfg.seed)
    val_idx = rng.choice(len(val), size=min(len(val), int(cfg.deep.val_steps) * bs), replace=False)
    lw = dict(cfg.deep.loss_weights)
    best, bad, history, ep0, gstep, planned = float("inf"), 0, [], 0, 0, False

    # ---------------------------------------------------------------- resume / warm start
    if ckpt_last.exists():
        st = torch.load(ckpt_last, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        total["n"], gstep, ep0 = st["total"], st["gstep"], st["epoch"] + 1
        best, bad, history, planned = st["best"], st["bad"], st["history"], True
        sched.last_epoch = gstep - 1
        sched.step()
        rng.bit_generator.state = st["rng"]
        log.info("%s: resumed at epoch %d (step %d of %d, best val %.4f)", name, ep0, gstep, total["n"], best)
    elif ckpt_best.exists():
        try:
            net.load_state_dict(torch.load(ckpt_best, map_location=dev))
            log.info("%s: warm start from existing best weights %s", name, ckpt_best)
        except Exception as exc:  # noqa: BLE001
            log.info("%s: existing model.pt not compatible (%s) - training from scratch", name, exc)
    log.info("%s: %d parameters on %s; %d steps/epoch x %d epochs planned, batch %d",
             name, n_params, dev, steps, int(cfg.deep.epochs), bs)

    # ---------------------------------------------------------------- reserve prediction time
    pred_s = 0.0
    if deadline() is not None and pred_windows:
        rate = forward_rate(net, rate_store if rate_store is not None else val, cfg, name)
        pred_s = 1.15 * pred_windows / rate
        log.info("%s: prediction throughput %.0f windows/s -> %.1f h reserved for %d prediction windows",
                 name, rate, pred_s / 3600, pred_windows)

    stopped_early = False
    for ep in range(ep0, 10**6):
        if gstep >= total["n"]:
            break
        net.train()
        t0, losses = time.time(), []
        order = rng.permutation(len(train))[: steps * bs]
        n_steps = min(steps, total["n"] - gstep)
        for s in range(n_steps):
            X, y, w = train.gather(order[s * bs:(s + 1) * bs])
            X, y, w = _to(X, dev), _to(y, dev), _to(w, dev)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                logit, amt, q = net(X)
            loss, _ = hurdle_loss(logit.float(), amt.float(), q.float(), y, w, lw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            gstep += 1
            losses.append(loss.item())
            if not planned and s == 49 and deadline() is not None:
                # plan the total number of steps from the measured speed (incl. ~10% validation)
                step_t = 1.1 * (time.time() - t0) / 50
                afford = int((time_left() - pred_s - MARGIN_S) / step_t)
                total["n"] = max(gstep + 1, min(planned_total, afford))
                planned = True
                log.info("%s: %.2f s/step -> %d training steps fit before the deadline (planned %d)",
                         name, step_t, total["n"], planned_total)
                n_steps = min(steps, total["n"] - (gstep - s - 1))
            if s + 1 >= n_steps:
                break
        vloss = evaluate_loss(net, val, val_idx, bs, dev, lw, use_amp, amp_dtype)
        dt = time.time() - t0
        history.append({"epoch": ep, "steps": gstep, "train_loss": float(np.mean(losses)), "val_loss": vloss,
                        "seconds": dt})
        log.info("%s ep %d step %d/%d train %.4f val %.4f (%.0fs)", name, ep, gstep, total["n"],
                 np.mean(losses), vloss, dt)
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            torch.save(net.state_dict(), ckpt_best)
        else:
            bad += 1
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "total": total["n"], "gstep": gstep, "epoch": ep, "best": best, "bad": bad,
                    "history": history, "rng": rng.bit_generator.state}, ckpt_last)
        if bad >= int(cfg.deep.patience):
            log.info("%s early stop at epoch %d", name, ep)
            break
        if time_left() - pred_s - MARGIN_S < dt:        # another epoch would eat the prediction time
            log.info("%s: stopping training to keep %.1f h for predictions", name, pred_s / 3600)
            stopped_early = True
            break

    if ckpt_best.exists():
        net.load_state_dict(torch.load(ckpt_best, map_location=dev))
    meta = {**meta, "name": name, "n_ch": n_ch, "n_params": n_params, "best_val_loss": best, "history": history,
            "deep_cfg": cfg.deep.to_dict(), "finished": True, "steps_done": gstep, "steps_planned": planned_total,
            "stopped_for_deadline": stopped_early}
    meta_f.write_text(json.dumps(meta, indent=2))
    return net


@torch.no_grad()
def evaluate_loss(net, store, idx, bs, dev, lw, use_amp, amp_dtype) -> float:
    net.eval()
    tot, n = 0.0, 0
    for s in range(0, len(idx), bs):
        X, y, w = store.gather(idx[s:s + bs])
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            logit, amt, q = net(_to(X, dev))
        loss, _ = hurdle_loss(logit.float(), amt.float(), q.float(), _to(y, dev), _to(w, dev), lw)
        tot += float(loss) * len(y)
        n += len(y)
    return tot / max(n, 1)


@torch.no_grad()
def _predict_rows(net, store: WindowStore, rows: np.ndarray, bs: int) -> pd.DataFrame:
    dev = next(net.parameters()).device
    use_amp, amp_dtype = _amp(dev)
    out = []
    for s in range(0, len(rows), bs):
        idx = rows[s:s + bs]
        X, _, _ = store.gather(idx)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            logit, amt, q = net(_to(X, dev))
        p, yhat, q90 = head_outputs(logit.float(), amt.float(), q.float())
        k = store.keys(idx)
        k["p_rain"] = p.cpu().numpy()
        k["y_hat"] = yhat.cpu().numpy()
        k["q90"] = q90.cpu().numpy()
        out.append(k)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


@torch.no_grad()
def predict_store(net, store: WindowStore, cfg, name: str | None = None, shard_dir: Path | None = None) -> pd.DataFrame:
    """Predict all windows of a store. With shard_dir, predictions are written per station-year and
    existing shards are reused, so an interrupted prediction resumes where it stopped."""
    net.eval()
    bs = eval_batch_for(cfg, name) if name else int(cfg.deep.eval_batch_size)
    if shard_dir is None:
        return _predict_rows(net, store, np.arange(len(store)), bs)
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for p, key in enumerate(store.part_keys):
        f = shard_dir / f"{key}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            continue
        if time_left() < MARGIN_S:
            raise DeadlineReached(f"deadline reached while predicting {shard_dir.name}; resubmit to resume")
        rows = store.part_index(p)
        if not len(rows):
            continue
        df = _predict_rows(net, store, rows, bs)
        df.to_parquet(f, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_model(model_dir: Path, device: str = "cpu"):
    """Rebuild a trained network from its folder (architecture comes from the saved meta.json)."""
    from ...config import Cfg

    meta = json.loads((Path(model_dir) / "meta.json").read_text())
    net = build_net(meta["name"], meta["n_ch"], Cfg.wrap({"deep": meta["deep_cfg"]}))
    net.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=device))
    net.eval()
    return net, meta
