"""Single entry point for the pipeline.

Examples
  python src/run.py all --config smoke                 # full smoke run on a laptop
  python src/run.py step02_truth step03_channels       # selected steps
  python src/run.py step09_deep --models samba gru --bench A
  python src/run.py all --set deep.epochs=30           # override any config value
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rainnow.config import load_config, set_seed, setup_logging  # noqa: E402
from rainnow.pipeline import STEPS  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", help=f"'all' or any of: {', '.join(STEPS)} (prefix like step05 is enough)")
    ap.add_argument("--config", default=None, help="overlay YAML (e.g. smoke or configs/smoke.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    ap.add_argument("--bench", nargs="+", default=["A", "B"], help="benchmarks to run (A, B)")
    ap.add_argument("--models", nargs="+", default=None, help="restrict a model step to these models")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--from-step", default=None, help="with 'all': start from this step")
    ap.add_argument("--force", action="store_true", help="step09: retrain models that already have predictions")
    args = ap.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config, args.set)
    set_seed(int(cfg.seed))
    log = logging.getLogger("rainnow.run")
    log.info("data=%s results=%s", cfg.data_dir, cfg.results_dir)

    if args.steps == ["all"]:
        todo = STEPS[STEPS.index(_match(args.from_step)):] if args.from_step else STEPS
    else:
        todo = [_match(s) for s in args.steps]
    for step in todo:
        t0 = time.time()
        log.info("=" * 20 + f" {step} " + "=" * 20)
        importlib.import_module(f"rainnow.pipeline.{step}").run(cfg, args)
        log.info("%s done in %.1f min", step, (time.time() - t0) / 60)


def _match(name: str) -> str:
    hits = [s for s in STEPS if s == name or s.startswith(name)]
    if len(hits) != 1:
        raise SystemExit(f"unknown or ambiguous step '{name}'. Steps: {STEPS}")
    return hits[0]


if __name__ == "__main__":
    main()
