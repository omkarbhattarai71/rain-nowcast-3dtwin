"""Common interface for tabular models and the probability calibrator."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class TabularModel:
    """fit(X, y, w, Xv, yv, wv) / predict(X) -> DataFrame[p_rain, y_hat, q90]."""

    name = "base"
    features: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
            Xv: pd.DataFrame, yv: np.ndarray, wv: np.ndarray) -> "TabularModel":
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def save(self, d: Path) -> None:
        Path(d).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(d) / "model.joblib")

    @staticmethod
    def load(d: Path) -> "TabularModel":
        return joblib.load(Path(d) / "model.joblib")


class ProbCalibrator:
    """Monotone map from a model score (e.g. expected amount) to P(rain > 0)."""

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, score: np.ndarray, obs: np.ndarray, w: np.ndarray | None = None) -> "ProbCalibrator":
        score = np.nan_to_num(np.asarray(score, float))
        self.iso.fit(score, np.asarray(obs, float), sample_weight=w)
        return self

    def __call__(self, score: np.ndarray) -> np.ndarray:
        return self.iso.predict(np.nan_to_num(np.asarray(score, float)))


def output(p_rain=None, y_hat=None, q90=None, n: int | None = None) -> pd.DataFrame:
    n = n if n is not None else len(next(v for v in (p_rain, y_hat, q90) if v is not None))
    nan = np.full(n, np.nan, dtype="float32")
    return pd.DataFrame({
        "p_rain": nan if p_rain is None else np.asarray(p_rain, "float32"),
        "y_hat": nan if y_hat is None else np.clip(np.asarray(y_hat, "float32"), 0, None),
        "q90": nan if q90 is None else np.clip(np.asarray(q90, "float32"), 0, None),
    })
