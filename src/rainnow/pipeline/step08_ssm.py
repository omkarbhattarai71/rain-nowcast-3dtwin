"""Step 08 - classical state-space models (priority track A).

Fitted on benchmark-A training data (MLE / Baum-Welch), predicted on A val/test and, as
transfer models, on the B radar period.
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np

from ..dataset import eval_mask, load_part, model_dir, pred_frame, split_parts, write_preds
from ..models.ssm_classic import KalmanModel, RegimeHMM, to_z

log = logging.getLogger(__name__)

BLOCK = 7 * 1440


def exog(df) -> np.ndarray:
    """Exogenous inputs for kalman_x (all causal channels, log-scaled like the observation)."""
    def lz(c, div=10.0):
        return np.log1p(100.0 * np.clip(np.nan_to_num(df[c].to_numpy(dtype="float64")), 0, None) / div)

    p10_recent = np.where(df["p10_age"].to_numpy() <= 9, df["p10_last"].to_numpy(), 0.0)
    rh = np.nan_to_num((df["rh"].to_numpy(dtype="float64") - 80.0) / 20.0)
    cols = [np.ones(len(df)), lz("nb_upwind_r0"), lz("nb_upwind_r1"), lz("nb_max"), lz("nb_mean"),
            np.log1p(100.0 * np.nan_to_num(p10_recent) / 10.0), lz("rain_signal"), rh]
    return np.column_stack(cols)


def _blocks(cfg, parts, rng):
    """Rain-weighted random 7-day blocks from the training parts (total <= ssm_fit_rows)."""
    cands = []
    for part in parts:
        df = load_part(cfg, part, "A", pad=0)
        n = len(df) // BLOCK
        for b in range(n):
            sl = df.iloc[b * BLOCK:(b + 1) * BLOCK]
            cands.append((float((sl["rain_rt"] > 0).sum()) + 1.0, sl))
    if not cands:
        return []
    k = max(1, min(len(cands), int(cfg.models.ssm_fit_rows) // BLOCK))
    w = np.array([c[0] for c in cands])
    pick = rng.choice(len(cands), size=k, replace=False, p=w / w.sum())
    return [cands[i][1] for i in pick]


def run(cfg, args) -> None:
    if "A" not in args.bench:
        log.info("SSM models are trained on benchmark A only")
        return
    names = [m for m in cfg.models.ssm if not args.models or m in args.models]
    rng = np.random.default_rng(int(cfg.seed))
    train_parts = split_parts(cfg, "A", "train")
    blocks = _blocks(cfg, train_parts, rng)
    log.info("SSM fit data: %d blocks (%d minutes)", len(blocks), sum(len(b) for b in blocks))
    targets = {"A": ("val", "test"), "B": ("val", "test")}

    fitted = {}
    for name in names:
        if name == "hmm_switch":
            fitted[name] = _fit_hmm(cfg, blocks, rng)
        else:
            fitted[name] = _fit_kalman(name, blocks)
        joblib.dump(fitted[name], model_dir(cfg, "A", name) / "model.joblib")

    for bench, splits in targets.items():
        for split in splits:
            parts = split_parts(cfg, bench, split)
            if not parts:
                continue
            for name, model in fitted.items():
                frames = _predict(cfg, name, model, parts, bench)
                write_preds(cfg, bench, name if bench == "A" else f"A__{name}", split, frames)


def _fit_kalman(name, blocks):
    use_u = name == "kalman_x"
    data = [(to_z(b["rain_rt"]), exog(b) if use_u else None) for b in blocks]
    m = KalmanModel(name, n_inputs=data[0][1].shape[1] if use_u else 0).fit(data)
    scores, ys = [], []
    for (z, U), b in zip(data, blocks):
        s = m.score(z, U)
        ok = b["target_valid"].to_numpy() > 0
        scores.append(s[ok])
        ys.append(np.nan_to_num(b["target"].to_numpy()[ok]))
    m.fit_output_maps(np.concatenate(scores), np.concatenate(ys))
    return m


def _fit_hmm(cfg, blocks, rng):
    L = 360
    wins = []
    for b in blocks:
        r = np.nan_to_num(b["rain_rt"].to_numpy())
        for s in range(0, len(r) - L, L):
            seg = r[s:s + L]
            if (seg > 0).any() or rng.random() < 0.1:
                wins.append(seg)
    wins = np.array(wins[:4000]) if wins else np.zeros((1, L))
    hmm = RegimeHMM(int(cfg.models.hmm_states)).fit(wins, int(cfg.models.hmm_em_iters))
    log.info("HMM states: p_zero=%s mean_wet=%s\nG=\n%s", np.round(hmm.p_zero, 3),
             np.round(np.exp(hmm.mu + hmm.sig**2 / 2), 3), np.round(hmm.G, 4))
    return hmm


def _predict(cfg, name, model, parts, bench):
    frames = []
    if name == "hmm_switch":
        for i in range(0, len(parts), 6):
            group = parts[i:i + 6]
            dfs = [load_part(cfg, p, bench) for p in group]
            res = model.filter_forecast([d["rain_rt"].to_numpy() for d in dfs])
            for p, d, (pr, yh, _) in zip(group, dfs, res):
                m = eval_mask(d)
                frames.append(pred_frame(d, p.sid, m, p_rain=pr[m], y_hat=yh[m]))
        return frames
    for p in parts:
        d = load_part(cfg, p, bench)
        U = exog(d) if name == "kalman_x" else None
        s = model.score(to_z(d["rain_rt"]), U)
        pr, yh = model.outputs(s)
        m = eval_mask(d)
        frames.append(pred_frame(d, p.sid, m, p_rain=pr[m], y_hat=yh[m]))
    return frames


def describe(model) -> str:
    return json.dumps({"theta": getattr(model, "theta", np.array([])).tolist()})
