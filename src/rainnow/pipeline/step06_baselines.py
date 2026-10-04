"""Step 06 - reference baselines (zero, climatology, persistence, ...) and the linear hurdle model."""
from __future__ import annotations

import logging

from ..models.baselines import REGISTRY as BASE
from ..models.linear import LogisticHurdle
from ._tabular_common import fit_and_predict, load_train_val

log = logging.getLogger(__name__)


def run(cfg, args) -> None:
    for bench in args.bench:
        try:
            tr, va = load_train_val(cfg, bench)
        except FileNotFoundError as exc:
            log.warning("bench %s: %s", bench, exc)
            continue
        names = list(cfg.models.baselines) + list(cfg.models.linear)
        if args.models:
            names = [n for n in names if n in args.models]
        for name in names:
            model = LogisticHurdle() if name == "logistic_hurdle" else BASE[name]()
            fit_and_predict(cfg, bench, name, model, tr, va)
