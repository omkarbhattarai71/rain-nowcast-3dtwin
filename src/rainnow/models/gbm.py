"""LightGBM models.

* lgbm_hurdle   : binary classifier P(rain) x regressor E[y | rain] (+ q90 quantile model)
* lgbm_tweedie  : one Tweedie regressor for the zero-inflated amount; P(rain) via isotonic calibration
* lgbm_supervisor_features : hurdle model restricted to the supervisors' step-3 feature set
* gbm_abl_*     : hurdle models restricted to feature groups (ablation)

Early stopping always uses the *validation* split, never the test split.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np

from ..features import SUPERVISOR_FEATURES, select_groups
from .base import ProbCalibrator, TabularModel, output

log = logging.getLogger(__name__)


def _train(params, X, y, w, Xv, yv, wv, rounds, es):
    dtr = lgb.Dataset(X, label=y, weight=w, free_raw_data=True)
    dva = lgb.Dataset(Xv, label=yv, weight=wv, reference=dtr, free_raw_data=True)
    booster = lgb.train(
        params, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["val"],
        callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(200)],
    )
    return booster


class LGBMHurdle(TabularModel):
    name = "lgbm_hurdle"

    def __init__(self, cfg, groups=None, features=None, quantile=True):
        self.params = dict(cfg.models.gbm.params)
        self.params["seed"] = cfg.seed
        self.rounds = int(cfg.models.gbm.num_boost_round)
        self.es = int(cfg.models.gbm.early_stopping)
        self.groups = groups
        self.fixed_features = features
        self.quantile = quantile

    def _cols(self, X):
        if self.fixed_features:
            return [c for c in self.fixed_features if c in X.columns]
        if self.groups:
            return select_groups(list(X.columns), self.groups)
        return list(X.columns)

    def fit(self, X, y, w, Xv, yv, wv):
        self.features = self._cols(X)
        X, Xv = X[self.features], Xv[self.features]
        self.clf = _train({**self.params, "objective": "binary", "metric": "binary_logloss"},
                          X, (y > 0).astype(float), w, Xv, (yv > 0).astype(float), wv, self.rounds, self.es)
        wet, wetv = y > 0, yv > 0
        self.reg = _train({**self.params, "objective": "regression", "metric": "l2"},
                          X[wet], y[wet], w[wet], Xv[wetv], yv[wetv], wv[wetv], self.rounds, self.es)
        self.q = None
        if self.quantile:
            self.q = _train({**self.params, "objective": "quantile", "alpha": 0.9, "metric": "quantile"},
                            X, y, w, Xv, yv, wv, self.rounds, self.es)
        log.info("%s: clf %d it, reg %d it", self.name, self.clf.best_iteration, self.reg.best_iteration)
        return self

    def predict(self, X):
        X = X[self.features]
        p = self.clf.predict(X, num_iteration=self.clf.best_iteration)
        amt = np.clip(self.reg.predict(X, num_iteration=self.reg.best_iteration), 0, None)
        q = self.q.predict(X, num_iteration=self.q.best_iteration) if self.q is not None else None
        return output(p, p * amt, q)

    def contributions(self, X):
        """SHAP values of the classifier (log-odds space); last column is the bias."""
        return self.clf.predict(X[self.features], num_iteration=self.clf.best_iteration, pred_contrib=True)


class LGBMTweedie(TabularModel):
    name = "lgbm_tweedie"

    def __init__(self, cfg):
        self.params = {**cfg.models.gbm.params, "seed": cfg.seed, "objective": "tweedie",
                       "tweedie_variance_power": cfg.models.gbm.tweedie_power, "metric": "tweedie"}
        self.rounds = int(cfg.models.gbm.num_boost_round)
        self.es = int(cfg.models.gbm.early_stopping)

    def fit(self, X, y, w, Xv, yv, wv):
        self.features = list(X.columns)
        self.reg = _train(self.params, X, y, w, Xv, yv, wv, self.rounds, self.es)
        self.cal = ProbCalibrator().fit(self.reg.predict(Xv, num_iteration=self.reg.best_iteration), yv > 0, wv)
        return self

    def predict(self, X):
        mu = self.reg.predict(X[self.features], num_iteration=self.reg.best_iteration)
        return output(self.cal(mu), mu)


def build(name: str, cfg) -> TabularModel:
    if name == "lgbm_hurdle":
        return LGBMHurdle(cfg)
    if name == "lgbm_tweedie":
        return LGBMTweedie(cfg)
    if name == "lgbm_supervisor_features":
        m = LGBMHurdle(cfg, features=SUPERVISOR_FEATURES, quantile=False)
        m.name = name
        return m
    if name in cfg.models.gbm.ablation_groups:
        m = LGBMHurdle(cfg, groups=list(cfg.models.gbm.ablation_groups[name]), quantile=False)
        m.name = name
        return m
    if name.startswith("lgbm_groups_"):
        m = LGBMHurdle(cfg, groups=name.removeprefix("lgbm_groups_").split("+"), quantile=False)
        m.name = name
        return m
    raise KeyError(name)
