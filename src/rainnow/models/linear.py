"""Linear hurdle model: logistic regression for P(rain) x ridge regression for the wet amount."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features import select_groups
from .base import TabularModel, output


class LogisticHurdle(TabularModel):
    name = "logistic_hurdle"

    def __init__(self, groups=("P", "N", "T")):
        self.groups = list(groups)

    def _x(self, X):
        return np.nan_to_num(X[self.features].to_numpy(dtype="float64"))

    def fit(self, X, y, w, Xv, yv, wv):
        self.features = select_groups(list(X.columns), self.groups)
        Z = self._x(X)
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0))
        self.clf.fit(Z, y > 0, logisticregression__sample_weight=w)
        wet = y > 0
        self.reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        self.reg.fit(Z[wet], y[wet], ridge__sample_weight=w[wet])
        return self

    def predict(self, X):
        Z = self._x(X)
        p = self.clf.predict_proba(Z)[:, 1]
        amt = np.clip(self.reg.predict(Z), 0, None)
        return output(p, p * amt)


REGISTRY = {"logistic_hurdle": LogisticHurdle}
