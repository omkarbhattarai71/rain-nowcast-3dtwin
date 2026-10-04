"""Classical state-space models (priority track A).

All Kalman models use the *steady-state* Kalman predictor of a linear time-invariant system.
After the gain converges (a few minutes), the one-step predictor is a fixed linear filter of the
observations and inputs, so it can be evaluated with scipy.signal.lfilter over millions of
minutes in milliseconds. That makes exact maximum-likelihood (MLE) fitting cheap, unlike the
hand-set Q/R of the supervisors' step 13.

Observation transform: z = log1p(100 * y)  (y in mm/min, gauge resolution 0.01 mm).

* kalman_ll   local level            x_{t+1} = x_t + w,                    z_t = x_t + v
* kalman_llt  damped local trend     [l, b]_{t+1} = [[1, phi],[0, phi]] [l, b]_t + w
* kalman_x    SSM with inputs        x_{t+1} = a x_t + B u_t + w  (neighbours, radar-free channels)
* hmm_switch  3-regime hidden Markov model (dry / light / heavy) with zero-inflated log-normal
              emissions, fitted by batched Baum-Welch; forecasts via the filtered regime probabilities.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, signal
from scipy.linalg import solve_discrete_are

from .base import ProbCalibrator

log = logging.getLogger(__name__)

SCALE = 100.0


def to_z(y):
    return np.log1p(SCALE * np.clip(np.nan_to_num(np.asarray(y, float)), 0, None))


# =========================================================================== LTI Kalman
@dataclass
class LTIKalman:
    A: np.ndarray
    H: np.ndarray            # (1, n)
    Q: np.ndarray            # (n, n)
    R: float
    B: np.ndarray | None = None   # (n, m)
    _tf: list = field(default_factory=list, repr=False)

    def gain(self):
        n = self.A.shape[0]
        P = solve_discrete_are(self.A.T, self.H.T, self.Q + 1e-12 * np.eye(n), np.array([[self.R]]))
        S = (self.H @ P @ self.H.T).item() + self.R
        K = self.A @ P @ self.H.T / S          # predictor gain (n, 1)
        return K, S

    def prepare(self):
        K, S = self.gain()
        F = self.A - K @ self.H
        G = K if self.B is None else np.hstack([K, self.B])
        tfs = []
        for i in range(G.shape[1]):
            num, den = signal.ss2tf(F, G[:, [i]], self.H @ F, self.H @ G[:, [i]])
            tfs.append((np.atleast_1d(num.squeeze()), den))
        self._tf, self.S = tfs, S
        return self

    def predict_next(self, z: np.ndarray, U: np.ndarray | None = None) -> np.ndarray:
        """out[t] = E[z_{t+1} | z_<=t, u_<=t] (steady state, zero initial state)."""
        if not self._tf:
            self.prepare()
        inputs = [z] + ([] if U is None else [U[:, j] for j in range(U.shape[1])])
        out = np.zeros(len(z))
        for (num, den), x in zip(self._tf, inputs):
            out += signal.lfilter(num, den, x)
        return out

    def nll(self, blocks) -> float:
        total, n = 0.0, 0
        for z, U in blocks:
            pred = self.predict_next(z, U)
            e = z[1:] - pred[:-1]
            total += 0.5 * (len(e) * np.log(2 * np.pi * self.S) + np.dot(e, e) / self.S)
            n += len(e)
        return total / max(n, 1)


def _sig(x):
    return 1 / (1 + np.exp(-x))


class KalmanModel:
    """Wrapper: parameterisation, MLE fit, monotone output maps to (P(rain), E[y])."""

    def __init__(self, kind: str, n_inputs: int = 0):
        self.kind = kind
        self.m = n_inputs

    # ---------------------------------------------------------------- parameterisation
    def x0(self):
        if self.kind == "kalman_ll":
            return np.array([np.log(0.05), np.log(0.5)])
        if self.kind == "kalman_llt":
            return np.array([np.log(0.05), np.log(0.005), np.log(0.5), 1.0])
        return np.concatenate([[np.log(0.05), np.log(0.5), 2.0], np.zeros(self.m)])

    def build(self, th) -> LTIKalman:
        if self.kind == "kalman_ll":
            return LTIKalman(np.eye(1), np.eye(1), np.eye(1) * np.exp(th[0]), float(np.exp(th[1])))
        if self.kind == "kalman_llt":
            phi = _sig(th[3])
            A = np.array([[1.0, phi], [0.0, phi]])
            return LTIKalman(A, np.array([[1.0, 0.0]]), np.diag([np.exp(th[0]), np.exp(th[1])]), float(np.exp(th[2])))
        a = _sig(th[2])
        B = np.asarray(th[3:], float).reshape(1, -1)
        return LTIKalman(np.array([[a]]), np.eye(1), np.eye(1) * np.exp(th[0]), float(np.exp(th[1])), B)

    def fit(self, blocks, max_iter: int = 200):
        def obj(th):
            try:
                v = self.build(th).nll(blocks)
            except (np.linalg.LinAlgError, ValueError):
                return 1e6
            return v if np.isfinite(v) else 1e6

        res = optimize.minimize(obj, self.x0(), method="L-BFGS-B", options={"maxiter": max_iter})
        self.theta = res.x
        self.model = self.build(res.x).prepare()
        log.info("%s MLE: nll=%.4f params=%s", self.kind, res.fun, np.round(res.x, 3))
        return self

    def fit_output_maps(self, scores, y, w=None):
        """Isotonic maps from the predicted z to P(rain) and to E[y] (fit on training data)."""
        from sklearn.isotonic import IsotonicRegression

        self.p_map = ProbCalibrator().fit(scores, y > 0, w)
        self.y_map = IsotonicRegression(out_of_bounds="clip", y_min=0).fit(np.nan_to_num(scores), y, sample_weight=w)
        return self

    def score(self, z, U=None):
        return self.model.predict_next(z, U)

    def outputs(self, scores):
        s = np.nan_to_num(scores)
        return self.p_map(s), self.y_map.predict(s)


# =========================================================================== HMM
class RegimeHMM:
    """K-state HMM with zero-inflated log-normal emissions on 1-minute rain (mm)."""

    def __init__(self, k: int = 3):
        self.k = k
        self.G = np.full((k, k), 0.02 / (k - 1))
        np.fill_diagonal(self.G, 0.98)
        self.pi0 = np.full(k, 1.0 / k)
        self.p_zero = np.linspace(0.995, 0.05, k)
        self.mu = np.log(np.geomspace(0.02, 0.4, k))
        self.sig = np.linspace(0.5, 0.9, k)

    def _emis(self, o):
        """Emission likelihoods, o: (..., ) -> (..., k)."""
        o = np.nan_to_num(o)[..., None]
        wet = o > 0
        lo = np.log(np.where(wet, o, 1.0))
        dens = np.exp(-0.5 * ((lo - self.mu) / self.sig) ** 2) / (self.sig * np.sqrt(2 * np.pi))
        return np.where(wet, (1 - self.p_zero) * dens, self.p_zero) + 1e-300

    def fit(self, windows: np.ndarray, iters: int = 30):
        """Batched Baum-Welch on windows (N, L)."""
        N, L = windows.shape
        for it in range(iters):
            E = self._emis(windows)                               # (N, L, k)
            alpha = np.zeros((N, L, self.k))
            c = np.zeros((N, L))
            a = self.pi0 * E[:, 0]
            c[:, 0] = a.sum(1)
            alpha[:, 0] = a / c[:, :1]
            for t in range(1, L):
                a = (alpha[:, t - 1] @ self.G) * E[:, t]
                c[:, t] = a.sum(1)
                alpha[:, t] = a / c[:, t:t + 1]
            beta = np.ones((N, L, self.k))
            for t in range(L - 2, -1, -1):
                beta[:, t] = ((E[:, t + 1] * beta[:, t + 1]) @ self.G.T) / c[:, t + 1:t + 2]
            gam = alpha * beta
            gam /= gam.sum(2, keepdims=True)
            xi = np.einsum("nti,ij,ntj->ij", alpha[:, :-1], self.G, E[:, 1:] * beta[:, 1:] / c[:, 1:, None])
            self.G = xi / xi.sum(1, keepdims=True)
            self.pi0 = gam[:, 0].mean(0)
            o = windows[..., None]
            wet = (o > 0).astype(float)
            gw = gam.sum((0, 1))
            self.p_zero = np.clip(((1 - wet) * gam).sum((0, 1)) / gw, 1e-4, 1 - 1e-4)
            lw = (wet * gam).sum((0, 1))
            lo = np.log(np.where(o > 0, o, 1.0))
            self.mu = (wet * gam * lo).sum((0, 1)) / np.maximum(lw, 1e-9)
            self.sig = np.sqrt((wet * gam * (lo - self.mu) ** 2).sum((0, 1)) / np.maximum(lw, 1e-9)) + 1e-3
            order = np.argsort(self.p_zero)[::-1]                   # keep states sorted dry -> heavy
            self.G = self.G[np.ix_(order, order)]
            self.pi0, self.p_zero, self.mu, self.sig = self.pi0[order], self.p_zero[order], self.mu[order], self.sig[order]
            if it % 5 == 0 or it == iters - 1:
                log.info("HMM EM it %d loglik/step=%.4f", it, np.log(c).mean())
        return self

    def filter_forecast(self, series: list[np.ndarray]):
        """Filtered next-minute forecasts for several series at once (loop over time, vectorised over series)."""
        P = len(series)
        T = max(len(s) for s in series)
        O = np.zeros((P, T))
        for i, s in enumerate(series):
            O[i, : len(s)] = np.nan_to_num(s)
        E = self._emis(O)
        mean_wet = np.exp(self.mu + 0.5 * self.sig**2)
        p_rain = np.zeros((P, T))
        y_hat = np.zeros((P, T))
        probs = np.zeros((P, T, self.k), dtype="float32")
        a = self.pi0 * E[:, 0]
        a /= a.sum(1, keepdims=True)
        for t in range(T):
            if t:
                a = (a @ self.G) * E[:, t]
                a /= a.sum(1, keepdims=True)
            nxt = a @ self.G
            probs[:, t] = a
            p_rain[:, t] = nxt @ (1 - self.p_zero)
            y_hat[:, t] = nxt @ ((1 - self.p_zero) * mean_wet)
        return [(p_rain[i, : len(s)], y_hat[i, : len(s)], probs[i, : len(s)]) for i, s in enumerate(series)]


# =========================================================================== residual hybrid
def fit_residual_ar(resid_blocks: list[np.ndarray]) -> KalmanModel:
    """AR(1)-plus-noise Kalman model for ML residuals (used by the GBM+Kalman hybrid)."""
    m = KalmanModel("kalman_x", n_inputs=0)
    m.fit([(r, None) for r in resid_blocks])
    return m
