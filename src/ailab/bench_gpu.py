"""GPU benchmark of the deep models (run on AI-Lab before a long training run, ~10-15 min).

Measures, per model, training speed (s/step at the configured batch size) and prediction throughput
(windows/s), with the fused mamba-ssm kernel (if installed and verified) and with the PyTorch scan.
Prints how long one epoch and the predictions of a full benchmark-A run take, so the step budget can
be set from measurements instead of guesses.

  srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 \
       singularity exec --nv rainnow.sif python src/ailab/bench_gpu.py
  ... python src/ailab/bench_gpu.py --models samba mamba --n-ch 36
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rainnow.config import load_config  # noqa: E402
from rainnow.models.deep import blocks  # noqa: E402
from rainnow.models.deep.blocks import hurdle_loss  # noqa: E402
from rainnow.models.deep.nets import build_net  # noqa: E402
from rainnow.models.deep.train import batch_size_for, eval_batch_for, pick_device  # noqa: E402


def _sync():
    torch.cuda.synchronize()


def bench(name, cfg, n_ch, dev, n=10):
    net = build_net(name, n_ch, cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    L = int(cfg.deep.window)
    bs, ebs = batch_size_for(cfg, name), eval_batch_for(cfg, name)
    x = torch.randn(bs, L, n_ch, device=dev)
    y = torch.rand(bs, device=dev) * (torch.rand(bs, device=dev) < 0.1)
    w = torch.ones(bs, device=dev)
    lw = dict(cfg.deep.loss_weights)
    times = []
    for i in range(n + 2):
        _sync()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, amt, q = net(x)
        loss, _ = hurdle_loss(logit.float(), amt.float(), q.float(), y, w, lw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        _sync()
        if i >= 2:
            times.append(time.time() - t0)
    step = sorted(times)[len(times) // 2]
    xe = torch.randn(ebs, L, n_ch, device=dev)
    net.eval()
    times = []
    with torch.no_grad():
        for i in range(n + 2):
            _sync()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                net(xe)
            _sync()
            if i >= 2:
                times.append(time.time() - t0)
    rate = ebs / sorted(times)[len(times) // 2]
    mem = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return step, rate, mem, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-ch", type=int, default=36, help="number of input channels (benchmark A: ~36)")
    ap.add_argument("--pred-windows", type=float, default=41e6, help="windows to predict per model (A: ~41M)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = pick_device(cfg)
    if dev.type != "cuda":
        sys.exit("needs a GPU (run inside srun --gres=gpu:1 ... --nv)")
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    models = args.models or list(cfg.deep.models)
    steps_epoch, epochs = int(cfg.deep.max_steps_per_epoch), int(cfg.deep.epochs)
    print(f"{'model':22s} {'scan':8s} {'batch':>5s} {'s/step':>8s} {'epoch[h]':>9s} {'20ep[h]':>8s} "
          f"{'pred win/s':>11s} {'pred[h]':>8s} {'GPU GB':>7s}")
    for fast in ("1", "0"):
        os.environ["RAINNOW_FAST_SCAN"] = fast
        blocks._FAST.update(checked=False, fn=None)
        kernel = "fused" if (fast == "1" and blocks._fast_scan_fn() is not None) else "pytorch"
        if fast == "1" and kernel == "pytorch":
            print("(mamba-ssm kernel not available: fused rows skipped)")
            continue
        for m in models:
            try:
                step, rate, mem, bs = bench(m, cfg, args.n_ch, dev)
            except torch.cuda.OutOfMemoryError:
                print(f"{m:22s} {kernel:8s} out of memory at the configured batch size")
                torch.cuda.empty_cache()
                continue
            ep = step * steps_epoch / 3600
            print(f"{m:22s} {kernel:8s} {bs:5d} {step:8.3f} {ep:9.2f} {ep * epochs:8.1f} "
                  f"{rate:11.0f} {args.pred_windows / rate / 3600:8.2f} {mem:7.1f}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
