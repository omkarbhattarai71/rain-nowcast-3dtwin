"""Reference forecasts: zero, climatology, persistence, decayed persistence, moving average."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import TabularModel, output


class Zero(TabularModel):
    name = "zero"
    features = ["P_lag0"]

    def predict(self, X):
        return output(np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X)))


class Persistence(TabularModel):
    """y(t+1) = y(t). The reference model: all skill scores are relative to it."""

    name = "persistence"
    features = ["P_lag0"]

    def predict(self, X):
        r = X["P_lag0"].to_numpy()
        return output((r > 0).astype(float), r, r)


class PersistenceDecay(TabularModel):
    """y(t+1) = a*y(t) + b*mean10; P(rain) from a logistic fit on (lag0>0, wet fraction)."""

    name = "persistence_decay"
    features = ["P_lag0", "P_sum10", "P_wetfrac10", "P_lag1"]

    def fit(self, X, y, w, Xv, yv, wv):
        from sklearn.linear_model import LogisticRegression

        A = np.c_[X["P_lag0"], X["P_sum10"] / 10.0]
        sw = np.sqrt(w)
        self.coef_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        Z = np.c_[(X["P_lag0"] > 0), X["P_wetfrac10"], (X["P_lag1"] > 0)].astype(float)
        self.clf = LogisticRegression(max_iter=500).fit(Z, y > 0, sample_weight=w)
        return self

    def predict(self, X):
        yhat = self.coef_[0] * X["P_lag0"].to_numpy() + self.coef_[1] * X["P_sum10"].to_numpy() / 10.0
        Z = np.c_[(X["P_lag0"] > 0), X["P_wetfrac10"], (X["P_lag1"] > 0)].astype(float)
        return output(self.clf.predict_proba(Z)[:, 1], yhat)


class MovingAverage(TabularModel):
    name = "moving_average"
    features = ["P_sum10", "P_wetfrac10"]

    def predict(self, X):
        return output(X["P_wetfrac10"].to_numpy(), X["P_sum10"].to_numpy() / 10.0)


class Climatology(TabularModel):
    """Mean rain and P(rain) by (month, hour) from the training period."""

    name = "climatology"
    features = ["P_lag0"]

    def fit(self, X, y, w, Xv, yv, wv):
        idx = X.index
        df = pd.DataFrame({"m": idx.month, "h": idx.hour, "y": y * w, "wet": (y > 0) * w, "w": w})
        g = df.groupby(["m", "h"]).sum()
        self.table = pd.DataFrame({"mean": g["y"] / g["w"], "p": g["wet"] / g["w"]})
        self.fallback = (float((y * w).sum() / w.sum()), float(((y > 0) * w).sum() / w.sum()))
        return self

    def predict(self, X):
        key = pd.MultiIndex.from_arrays([X.index.month, X.index.hour])
        t = self.table.reindex(key)
        return output(t["p"].fillna(self.fallback[1]).to_numpy(), t["mean"].fillna(self.fallback[0]).to_numpy())


REGISTRY = {
    "zero": Zero,
    "climatology": Climatology,
    "persistence": Persistence,
    "persistence_decay": PersistenceDecay,
    "moving_average": MovingAverage,
}
