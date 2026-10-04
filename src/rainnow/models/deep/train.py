"""Training / inference loop shared by all deep sequence models."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .blocks import head_outputs, hurdle_loss
from .data import WindowStore
from .nets import build_net

log = logging.getLogger(__name__)


_CPU_READY = False


def pick_device(cfg) -> torch.device:
    global _CPU_READY
    want = cfg.deep.device
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
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


def _to(x, dev):
    return torch.from_numpy(x).to(dev, non_blocking=True)


def train_model(name: str, cfg, train: WindowStore, val: WindowStore, out_dir: Path, meta: dict):
    dev = pick_device(cfg)
    n_ch = train.X[0].shape[1]
    net = build_net(name, n_ch, cfg).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    log.info("%s: %d parameters on %s", name, n_params, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.deep.lr, weight_decay=cfg.deep.weight_decay)
    bs = batch_size_for(cfg, name)
    steps = min(int(cfg.deep.max_steps_per_epoch), max(1, len(train) // bs))
    total = steps * int(cfg.deep.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    rng = np.random.default_rng(cfg.seed)
    val_idx = rng.choice(len(val), size=min(len(val), int(cfg.deep.val_steps) * bs), replace=False)
    lw = dict(cfg.deep.loss_weights)

    best, bad, history = float("inf"), 0, []
    ckpt = out_dir / "model.pt"
    for ep in range(int(cfg.deep.epochs)):
        net.train()
        t0, losses = time.time(), []
        order = rng.permutation(len(train))[: steps * bs]
        for s in range(steps):
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
            losses.append(loss.item())
        vloss = evaluate_loss(net, val, val_idx, bs, dev, lw, use_amp, amp_dtype)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": vloss,
                        "seconds": time.time() - t0})
        log.info("%s ep %d train %.4f val %.4f (%.0fs)", name, ep, np.mean(losses), vloss, time.time() - t0)
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            torch.save(net.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= int(cfg.deep.patience):
                log.info("%s early stop at epoch %d", name, ep)
                break
    net.load_state_dict(torch.load(ckpt, map_location=dev))
    meta = {**meta, "name": name, "n_ch": n_ch, "n_params": n_params, "best_val_loss": best, "history": history,
            "deep_cfg": cfg.deep.to_dict()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
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
def predict_store(net, store: WindowStore, cfg, name: str | None = None) -> pd.DataFrame:
    dev = next(net.parameters()).device
    net.eval()
    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    by = cfg.deep.get("eval_batch_size_by_model", {}) or {}
    bs = int(by.get(name, cfg.deep.eval_batch_size))
    out = []
    for s in range(0, len(store), bs):
        idx = np.arange(s, min(s + bs, len(store)))
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


def load_model(model_dir: Path, device: str = "cpu"):
    """Rebuild a trained network from its folder (architecture comes from the saved meta.json)."""
    from ...config import Cfg

    meta = json.loads((Path(model_dir) / "meta.json").read_text())
    net = build_net(meta["name"], meta["n_ch"], Cfg.wrap({"deep": meta["deep_cfg"]}))
    net.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=device))
    net.eval()
    return net, meta
