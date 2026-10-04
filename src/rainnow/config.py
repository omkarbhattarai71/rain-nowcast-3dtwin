"""Configuration loading (YAML + command-line overrides) and project paths."""
from __future__ import annotations

import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
CONFIG_DIR = SRC_DIR / "configs"

log = logging.getLogger("rainnow")


def _load_dotenv() -> None:
    """Load credentials from .env (repo root or src/) without overriding the environment.

    If an AWS profile exists (~/.aws/credentials), it takes precedence over AWS keys in .env
    (set RAINNOW_DOTENV_AWS=1 to force the .env keys). Inside Docker there is no profile, so
    the .env keys are used.
    """
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    has_profile = (Path.home() / ".aws" / "credentials").exists() and os.environ.get("RAINNOW_DOTENV_AWS") != "1"
    for candidate in (ROOT_DIR / ".env", SRC_DIR / ".env"):
        if not candidate.exists():
            continue
        for key, value in dotenv_values(candidate).items():
            if value is None or key in os.environ:
                continue
            if has_profile and key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
                continue
            os.environ[key] = value


class Cfg(dict):
    """Dict with attribute access (cfg.deep.epochs) that stays JSON/YAML serialisable."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))

    # ------------------------------------------------------------------ paths
    @property
    def data_dir(self) -> Path:
        return _resolve(os.environ.get("RAINNOW_DATA") or self["paths"]["data"])

    @property
    def results_dir(self) -> Path:
        return _resolve(os.environ.get("RAINNOW_RESULTS") or self["paths"]["results"])

    def path(self, *parts: str, results: bool = False, mkdir: bool = False) -> Path:
        base = self.results_dir if results else self.data_dir
        p = base.joinpath(*parts)
        if mkdir:
            (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
        return p


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT_DIR / p


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _set_by_path(d: dict, dotted: str, raw_value: str) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Cfg:
    """Load default.yaml, then an optional overlay file, then `key.sub=value` overrides."""
    _load_dotenv()
    with open(CONFIG_DIR / "default.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path:
        p = Path(path)
        for cand in (CONFIG_DIR / p, CONFIG_DIR / f"{path}.yaml"):
            if not p.exists() and cand.exists():
                p = cand
        with open(p, encoding="utf-8") as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        key, _, value = item.partition("=")
        _set_by_path(cfg, key.strip(), value)
    return Cfg.wrap(cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
