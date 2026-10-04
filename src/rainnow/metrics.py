"""Verification metrics: continuous, categorical, probabilistic, event-based, bootstrap and DM test."""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- categorical
def contingency(obs: np.ndarray, fc: np.ndarray, w: np.ndarray | None = None) -> dict:
    obs = np.asarray(obs, bool)
    fc = np.asarray(fc, bool)
    w = np.ones(len(obs)) if w is None else np.asarray(w, float)
    return {
        "hits": float(w[obs & fc].sum()),
        "misses": float(w[obs & ~fc].sum()),
        "false_alarms": float(w[~obs & fc].sum()),
        "correct_neg": float(w[~obs & ~fc].sum()),
    }


def categorical_scores(c: dict) -> dict:
    h, m, f, cn = c["hits"], c["misses"], c["false_alarms"], c["correct_neg"]
    n = h + m + f + cn
    nan = float("nan")
    pod = h / (h + m) if h + m else nan
    far = f / (h + f) if h + f else nan
    csi = h / (h + m + f) if h + m + f else nan
    fbias = (h + f) / (h + m) if h + m else nan
    h_rand = (h + m) * (h + f) / n if n else nan
    ets = (h - h_rand) / (h + m + f - h_rand) if n and (h + m + f - h_rand) else nan
    exp_correct = ((h + m) * (h + f) + (cn + m) * (cn + f)) / n if n else nan
    hss = (h + cn - exp_correct) / (n - exp_correct) if n and (n - exp_correct) else nan
    return {"pod": pod, "far": far, "csi": csi, "fbias": fbias, "ets": ets, "hss": hss}


def tune_threshold(score: np.ndarray, obs: np.ndarray, n_grid: int = 200) -> float:
    """Threshold on `score` maximising CSI for event `obs` (chosen on validation data only)."""
    score = np.asarray(score, float)
    obs = np.asarray(obs, bool)
    ok = np.isfinite(score)
    score, obs = score[ok], obs[ok]
    if obs.sum() == 0 or len(np.unique(score)) < 2:
        return 0.5
    cand = np.unique(np.quantile(score, np.linspace(0.5, 0.9999, n_grid)))
    cand = cand[cand > 0] if (cand > 0).any() else cand
    order = np.argsort(-score)
    s_sorted, o_sorted = score[order], obs[order]
    cum_hits = np.cumsum(o_sorted)
    best, best_thr = -1.0, 0.5
    total_pos = obs.sum()
    for thr in cand:
        k = np.searchsorted(-s_sorted, -thr, side="right")  # number of rows with score >= thr
        hits = cum_hits[k - 1] if k else 0
        fa = k - hits
        csi = hits / (total_pos + fa) if total_pos + fa else 0
        if csi > best:
            best, best_thr = csi, float(thr)
    return best_thr


# --------------------------------------------------------------------------- continuous / prob
def continuous_scores(y: np.ndarray, yhat: np.ndarray) -> dict:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    e = yhat - y
    wet = y > 0
    out = {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "corr": float(np.corrcoef(y, yhat)[0, 1]) if np.std(yhat) > 0 and np.std(y) > 0 else float("nan"),
        "rmse_wet": float(np.sqrt(np.mean(e[wet] ** 2))) if wet.any() else float("nan"),
        "mae_wet": float(np.mean(np.abs(e[wet]))) if wet.any() else float("nan"),
    }
    return out


def brier(p: np.ndarray, obs: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 0, 1)
    return float(np.mean((p - np.asarray(obs, float)) ** 2))


def pr_auc(p: np.ndarray, obs: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    obs = np.asarray(obs, bool)
    if obs.all() or (~obs).all():
        return float("nan")
    return float(average_precision_score(obs, np.nan_to_num(p)))


def pinball(q: np.ndarray, y: np.ndarray, tau: float = 0.9) -> float:
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def reliability_table(p: np.ndarray, obs: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p = np.clip(np.asarray(p, float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "o": np.asarray(obs, float)})
    g = df.groupby("bin").agg(p_mean=("p", "mean"), obs_freq=("o", "mean"), n=("o", "size")).reset_index()
    return g


# --------------------------------------------------------------------------- bootstrap
def day_aggregates(df: pd.DataFrame, models: list[str], thr: dict[str, float]) -> dict[str, pd.DataFrame]:
    """Per-day sufficient statistics (SSE, n, contingency for rain/no-rain) per model."""
    obs = df["y_true"].to_numpy() > 0
    day = df["day"].to_numpy()
    out = {}
    for m in models:
        yhat = df[f"{m}__y_hat"].to_numpy()
        p = df[f"{m}__p_rain"].to_numpy()
        fc = p >= thr[m]
        se = (yhat - df["y_true"].to_numpy()) ** 2
        agg = pd.DataFrame({
            "day": day, "se": se, "n": 1.0,
            "hits": (obs & fc).astype(float), "misses": (obs & ~fc).astype(float), "fa": (~obs & fc).astype(float),
        }).groupby("day").sum()
        out[m] = agg
    return out


def bootstrap_ci(aggs: dict[str, pd.DataFrame], ref: str, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """Day-block bootstrap of RMSE, CSI and their differences to the reference model."""
    days = next(iter(aggs.values())).index.to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(n_boot, len(days)))
    res = {}
    for m, a in aggs.items():
        a = a.loc[days]
        se, n = a["se"].to_numpy(), a["n"].to_numpy()
        h, mi, fa = a["hits"].to_numpy(), a["misses"].to_numpy(), a["fa"].to_numpy()
        rmse = np.sqrt(se[draws].sum(1) / n[draws].sum(1))
        den = h[draws].sum(1) + mi[draws].sum(1) + fa[draws].sum(1)
        csi = np.where(den > 0, h[draws].sum(1) / np.where(den > 0, den, 1), np.nan)
        res[m] = (rmse, csi)
    rows = []
    r_rmse, r_csi = res[ref]
    for m, (rmse, csi) in res.items():
        skill = 1 - rmse / r_rmse
        rows.append({
            "model": m,
            "rmse_lo": np.nanpercentile(rmse, 2.5), "rmse_hi": np.nanpercentile(rmse, 97.5),
            "csi_lo": np.nanpercentile(csi, 2.5), "csi_hi": np.nanpercentile(csi, 97.5),
            "skill_lo": np.nanpercentile(skill, 2.5), "skill_hi": np.nanpercentile(skill, 97.5),
            "dcsi_vs_ref_lo": np.nanpercentile(csi - r_csi, 2.5), "dcsi_vs_ref_hi": np.nanpercentile(csi - r_csi, 97.5),
        })
    return pd.DataFrame(rows)


def paired_bootstrap(aggs: dict[str, pd.DataFrame], a: str, b: str, n_boot: int, seed: int = 1) -> dict:
    """CI of CSI(a)-CSI(b) and RMSE(a)-RMSE(b) on the same resampled days."""
    days = aggs[a].index.to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(n_boot, len(days)))

    def stats(m):
        x = aggs[m].loc[days]
        rm = np.sqrt(x["se"].to_numpy()[draws].sum(1) / x["n"].to_numpy()[draws].sum(1))
        h = x["hits"].to_numpy()[draws].sum(1)
        den = h + x["misses"].to_numpy()[draws].sum(1) + x["fa"].to_numpy()[draws].sum(1)
        return rm, h / np.where(den > 0, den, 1)

    ra, ca = stats(a)
    rb, cb = stats(b)
    return {
        "model_a": a, "model_b": b,
        "d_csi_mean": float(np.mean(ca - cb)), "d_csi_lo": float(np.percentile(ca - cb, 2.5)),
        "d_csi_hi": float(np.percentile(ca - cb, 97.5)),
        "d_rmse_mean": float(np.mean(ra - rb)), "d_rmse_lo": float(np.percentile(ra - rb, 2.5)),
        "d_rmse_hi": float(np.percentile(ra - rb, 97.5)),
    }


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int = 7) -> dict:
    """DM test on daily mean loss differentials with Newey-West variance (two-sided)."""
    from scipy import stats

    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    dc = d - d.mean()
    gamma0 = np.dot(dc, dc) / n
    var = gamma0
    for k in range(1, min(max_lag, n - 1) + 1):
        g = np.dot(dc[k:], dc[:-k]) / n
        var += 2 * (1 - k / (max_lag + 1)) * g
    stat = d.mean() / np.sqrt(var / n) if var > 0 else float("nan")
    p = 2 * (1 - stats.norm.cdf(abs(stat))) if np.isfinite(stat) else float("nan")
    return {"dm_stat": float(stat), "p_value": float(p)}
