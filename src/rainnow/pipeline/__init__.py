"""Pipeline steps. Each module exposes run(cfg, args). Use src/run.py to execute them."""
from __future__ import annotations

import logging
from contextlib import contextmanager

STEPS = [
    "step01_inventory",
    "step02_truth",
    "step03_channels",
    "step04_radar",
    "step05_tabular",
    "step06_baselines",
    "step07_gbm",
    "step08_ssm",
    "step09_deep",
    "step10_hybrids",
    "step11_evaluate",
    "step12_attenuation",
    "step13_leakage_audit",
]

log = logging.getLogger(__name__)


class Isolated:
    """Run models one at a time: a failing model is logged and skipped, the others still run.

    finish() raises at the end if anything failed, so the job is still marked as failed
    (and the failure is visible in sacct), but without losing the models that worked.
    """

    def __init__(self, step: str):
        self.step = step
        self.failed: list[str] = []

    @contextmanager
    def model(self, name: str):
        try:
            yield
        except Exception:  # noqa: BLE001
            log.exception("%s: model %s FAILED - continuing with the next model", self.step, name)
            self.failed.append(name)

    def finish(self) -> None:
        if self.failed:
            raise RuntimeError(f"{self.step}: {len(self.failed)} model(s) failed: {self.failed} (see log above)")
