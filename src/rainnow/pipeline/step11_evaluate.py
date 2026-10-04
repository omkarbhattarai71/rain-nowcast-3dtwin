"""Step 11 - evaluation of every model on identical test rows; thresholds tuned on validation only.

Outputs (results/bench<X>/eval/): metrics_test.csv, thresholds.json, bootstrap_ci.csv, pairwise.csv,
per_station.csv, per_month.csv, reliability.csv, figures/*.png, best_model.json, REPORT.md
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..dataset import index_path, model_dir
from ..metrics import (bootstrap_ci, brier, categorical_scores, contingency, continuous_scores, day_aggregates,
                       diebold_mariano, paired_bootstrap, pinball, pr_auc, reliability_table, tune_threshold)

log = logging.getLogger(__name__)

PRIORITY = ("samba", "kalman", "hmm")       # supervisor-priority families (state-space / SAMBA)


def family(name: str) -> str:
    n = name.removeprefix("A__")
    if n.startswith("samba"):
        return "SAMBA (deep SSM)"
    if n.startswith(("mamba", "s4d")):
        return "deep SSM"
    if n.startswith(("kalman", "hmm")):
        return "classical SSM"
    if n.startswith(("gru", "tcn", "patchtst")):
        return "deep (non-SSM)"
    if n.startswith(("lgbm", "gbm")):
        return "gradient boosting"
    if n.startswith("hyb"):
        return "hybrid"
    if n.startswith("logistic"):
        return "linear"
    return "baseline"


def _preds_models(cfg, bench) -> list[str]:
    d = cfg.path(f"bench{bench}", "preds", results=True)
    return sorted(p.name for p in d.iterdir() if (p / "test.parquet").exists() and (p / "val.parquet").exists()) \
        if d.exists() else []


def run(cfg, args) -> None:
    for bench in args.bench:
        try:
            evaluate_bench(cfg, bench)
        except FileNotFoundError as exc:
            log.warning("bench %s: %s", bench, exc)


def evaluate_bench(cfg, bench: str) -> None:
    idx_v = pd.read_parquet(index_path(cfg, bench, "val"))
    idx_t = pd.read_parquet(index_path(cfg, bench, "test"))
    idx_t["row"] = np.arange(len(idx_t))
    models = _preds_models(cfg, bench)
    if not models:
        raise FileNotFoundError(f"no predictions for bench {bench}")
    out = cfg.path(f"bench{bench}", "eval", results=True, mkdir=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    keys = ["station_id", "time"]
    pdir = cfg.path(f"bench{bench}", "preds", results=True)

    # ---------------------------------------------------------------- thresholds on validation
    thr = {}
    for m in models:
        v = idx_v.merge(pd.read_parquet(pdir / m / "val.parquet"), on=keys)
        obs = v["y_true"].to_numpy() > 0
        t = {"rain": tune_threshold(v["p_rain"].to_numpy(), obs)}
        for th in cfg.evaluation.intensity_thresholds_mmh:
            t[f"ge{th}"] = tune_threshold(v["y_hat"].to_numpy(), v["y_true"].to_numpy() * 60 >= th)
        thr[m] = t
    (out / "thresholds.json").write_text(json.dumps(thr, indent=1))

    # ---------------------------------------------------------------- common test rows
    common = np.ones(len(idx_t), bool)
    for m in models:
        k = pd.read_parquet(pdir / m / "test.parquet", columns=keys)
        rows = idx_t.merge(k, on=keys)["row"].to_numpy()
        present = np.zeros(len(idx_t), bool)
        present[rows] = True
        common &= present
    base = idx_t[common].reset_index(drop=True)
    log.info("bench %s: %d models, %d common test rows (of %d)", bench, len(models), len(base), len(idx_t))
    y = base["y_true"].to_numpy()
    obs = y > 0
    dry_now = base["rain_now"].to_numpy() == 0
    wet_now = ~dry_now
    ref = cfg.evaluation.reference_model if cfg.evaluation.reference_model in models else models[0]

    rows, rel_rows, st_rows, mo_rows = [], [], [], []
    wide = base[keys + ["y_true", "day"]].copy()
    for m in models:
        p = base[keys].merge(pd.read_parquet(pdir / m / "test.parquet"), on=keys, how="left")
        pr, yh, q = p["p_rain"].to_numpy(), p["y_hat"].to_numpy(), p["q90"].to_numpy()
        pr = np.nan_to_num(pr)
        wide[f"{m}__p_rain"], wide[f"{m}__y_hat"] = pr.astype("float32"), yh.astype("float32")
        fc = pr >= thr[m]["rain"]
        r = {"model": m, "family": family(m), "rows": len(y)}
        r |= continuous_scores(y, yh)
        r |= {f"{k}_rain": v for k, v in categorical_scores(contingency(obs, fc)).items()}
        r["brier"] = brier(pr, obs)
        r["pr_auc"] = pr_auc(pr, obs)
        r["pinball_q90"] = pinball(q, y) if np.isfinite(q).all() else np.nan
        for th in cfg.evaluation.intensity_thresholds_mmh:
            ev = y * 60 >= th
            sc = categorical_scores(contingency(ev, yh >= thr[m][f"ge{th}"]))
            r[f"csi_ge{th}"], r[f"pod_ge{th}"], r[f"far_ge{th}"] = sc["csi"], sc["pod"], sc["far"]
        on = categorical_scores(contingency(obs[dry_now], fc[dry_now]))
        r |= {"onset_pod": on["pod"], "onset_far": on["far"], "onset_csi": on["csi"]}
        ce = categorical_scores(contingency(~obs[wet_now], ~fc[wet_now]))
        r |= {"cessation_pod": ce["pod"], "cessation_csi": ce["csi"]}
        r["rmse_wet_now"] = float(np.sqrt(np.mean((yh[wet_now] - y[wet_now]) ** 2))) if wet_now.any() else np.nan
        r["threshold_rain"] = thr[m]["rain"]
        rows.append(r)
        rel = reliability_table(pr, obs)
        rel["model"] = m
        rel_rows.append(rel)
        for (sid,), g in pd.DataFrame({"sid": base["station_id"], "o": obs, "f": fc, "e": (yh - y) ** 2}).groupby(["sid"]):
            st_rows.append({"model": m, "station_id": sid, "rmse": float(np.sqrt(g.e.mean())),
                            **{f"{k}": v for k, v in categorical_scores(contingency(g.o, g.f)).items() if k in ("csi", "pod", "far")}})
        month = base["time"].dt.month.to_numpy()
        for mo in np.unique(month):
            mm = month == mo
            mo_rows.append({"model": m, "month": int(mo), "rmse": float(np.sqrt(np.mean((yh[mm] - y[mm]) ** 2))),
                            "csi": categorical_scores(contingency(obs[mm], fc[mm]))["csi"]})
    met = pd.DataFrame(rows)
    ref_rmse = float(met.loc[met.model == ref, "rmse"].iloc[0])
    met["skill_rmse_vs_persistence"] = 1 - met["rmse"] / ref_rmse
    met = met.sort_values("csi_rain", ascending=False)
    met.to_csv(out / "metrics_test.csv", index=False)
    pd.concat(rel_rows).to_csv(out / "reliability.csv", index=False)
    per_station = pd.DataFrame(st_rows)
    per_station.to_csv(out / "per_station.csv", index=False)
    per_station.groupby("model")[["rmse", "csi", "pod", "far"]].agg(["median", lambda s: s.quantile(.75) - s.quantile(.25)]) \
        .to_csv(out / "per_station_summary.csv")
    pd.DataFrame(mo_rows).to_csv(out / "per_month.csv", index=False)

    # ---------------------------------------------------------------- uncertainty & tests
    aggs = day_aggregates(wide, models, {m: thr[m]["rain"] for m in models})
    ci = bootstrap_ci(aggs, ref, int(cfg.evaluation.bootstrap), int(cfg.seed))
    ci.to_csv(out / "bootstrap_ci.csv", index=False)
    met = met.merge(ci, on="model", how="left")
    met.to_csv(out / "metrics_test.csv", index=False)
    best = met.iloc[0]["model"]
    pairs = []
    focus = [m for m in models if m.removeprefix("A__").startswith(PRIORITY)] + [best]
    for a in dict.fromkeys(focus):
        for b in models:
            if a == b:
                continue
            pb = paired_bootstrap(aggs, a, b, int(cfg.evaluation.bootstrap))
            da = aggs[a]["se"] / aggs[a]["n"]
            db = aggs[b]["se"] / aggs[b]["n"]
            pairs.append(pb | diebold_mariano(da.to_numpy(), db.reindex(da.index).to_numpy()))
    pd.DataFrame(pairs).to_csv(out / "pairwise.csv", index=False)

    _best_model_json(cfg, bench, met, out)
    _figures(cfg, met, pd.concat(rel_rows), wide, out, models, thr)
    _report(cfg, bench, met, pd.DataFrame(pairs), len(base), out)
    log.info("bench %s top 10 by CSI:\n%s", bench, met[["model", "csi_rain", "pod_rain", "far_rain", "rmse",
                                                         "skill_rmse_vs_persistence", "onset_csi"]].head(10).to_string(index=False))


def _deployable(cfg, bench, m) -> bool:
    if m.startswith(("A__", "hyb")):
        return False
    d = model_dir(cfg, bench, m)
    return (d / "model.pt").exists() or ((d / "model.joblib").exists() and not m.startswith(("kalman", "hmm")))


def _best_model_json(cfg, bench, met, out):
    ranked = met.sort_values(["csi_rain", "skill_rmse_vs_persistence"], ascending=False)
    dep = [m for m in ranked.model if _deployable(cfg, bench, m)]
    info = {"bench": bench, "best_overall": ranked.iloc[0]["model"],
            "best_deployable": dep[0] if dep else None,
            "ranking_metric": "csi_rain (threshold tuned on validation), tie-break RMSE skill"}
    (out / "best_model.json").write_text(json.dumps(info, indent=1))


def _figures(cfg, met, rel, wide, out, models, thr):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    m = met.sort_values("csi_rain")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(m))))
    err = np.vstack([m["csi_rain"] - m["csi_lo"], m["csi_hi"] - m["csi_rain"]]).clip(0)
    ax.barh(m["model"], m["csi_rain"], xerr=err, color=["#d62728" if f.startswith("SAMBA") else
                                                      "#9467bd" if "SSM" in f else "#1f77b4" for f in m["family"]])
    ax.set_xlabel("CSI (rain / no rain), 95% day-block bootstrap CI")
    ax.set_title("Rain detection skill, test period")
    fig.tight_layout()
    fig.savefig(fig_dir / "csi_ci.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(m))))
    ms = met.sort_values("skill_rmse_vs_persistence")
    ax.barh(ms["model"], ms["skill_rmse_vs_persistence"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("RMSE skill vs persistence (1 - RMSE/RMSE_pers)")
    fig.tight_layout()
    fig.savefig(fig_dir / "rmse_skill.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name in met["model"].head(6):
        r = rel[rel.model == name]
        ax.plot(r["p_mean"], r["obs_freq"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.legend(fontsize=7)
    ax.set_title("Reliability (top 6 by CSI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(met["onset_far"], met["onset_pod"])
    for _, r in met.iterrows():
        ax.annotate(r["model"], (r["onset_far"], r["onset_pod"]), fontsize=6)
    ax.set_xlabel("FAR on dry-now minutes")
    ax.set_ylabel("POD of rain onsets")
    ax.set_title("Onset detection (persistence scores 0 here)")
    fig.tight_layout()
    fig.savefig(fig_dir / "onset.png", dpi=150)
    plt.close(fig)

    # case study: wettest test day at one station
    day_tot = wide.groupby(["station_id", "day"])["y_true"].sum()
    if len(day_tot) and day_tot.max() > 0:
        sid, day = day_tot.idxmax()
        g = wide[(wide.station_id == sid) & (wide.day == day)].sort_values("time")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(g["time"] + pd.Timedelta(minutes=1), g["y_true"], "k", lw=1.5, label="observed (t+1)")
        for name in list(dict.fromkeys(["persistence", met.iloc[0]["model"]] +
                                       [x for x in models if x.startswith("samba")][:1])):
            if f"{name}__y_hat" in g:
                ax.plot(g["time"] + pd.Timedelta(minutes=1), g[f"{name}__y_hat"], lw=1, label=name)
        ax.set_ylabel("mm / min")
        ax.set_title(f"Case study: station {sid}, {day}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "case_study.png", dpi=150)
        plt.close(fig)


def _published(cfg) -> str:
    pub = cfg.path("inventory", "published", results=True)
    if not pub.exists():
        return "_not downloaded_"
    lines = []
    for f in sorted(pub.glob("*.csv")):
        df = pd.read_csv(f)
        cols = [c for c in ("precision", "recall", "f1", "pr_auc", "csi", "pod", "far") if c in df.columns]
        if cols:
            lines.append(f"* `{f.name}`: " + ", ".join(f"{c}={df[c].mean():.3f}" for c in cols)
                         + f" (mean over {len(df)} rows)")
    return "\n".join(lines) or "_no metric columns found_"


def _report(cfg, bench, met, pairs, n_rows, out):
    cols = ["model", "family", "csi_rain", "csi_lo", "csi_hi", "pod_rain", "far_rain", "ets_rain", "rmse", "mae",
            "skill_rmse_vs_persistence", "onset_pod", "onset_far", "onset_csi", "csi_ge2.5", "csi_ge10.0",
            "brier", "pr_auc"]
    cols = [c for c in cols if c in met.columns]
    tab = met[cols].copy()
    for c in cols[2:]:
        tab[c] = tab[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
    best = met.iloc[0]
    sam = met[met.model.str.removeprefix("A__").str.startswith("samba")]
    lines = [
        f"# Benchmark {bench} - evaluation report", "",
        f"* test rows (identical for all models): **{n_rows:,}**",
        f"* thresholds tuned on validation only; CIs: day-block bootstrap ({cfg.evaluation.bootstrap} draws)",
        f"* best by CSI: **{best.model}** ({best.family}), CSI={best.csi_rain:.4f}, RMSE skill vs persistence="
        f"{best.skill_rmse_vs_persistence:.4f}", "",
    ]
    if len(sam):
        s = sam.iloc[0]
        lines.append(f"* best SAMBA variant: **{s.model}**, CSI={s.csi_rain:.4f} [{s.csi_lo:.4f}, {s.csi_hi:.4f}]")
    lines += ["", "## All models (sorted by CSI)", "", _md(tab), "",
              "## Pairwise tests (priority models vs. all)", "",
              _md(pairs.round(5)) if len(pairs) else "_none_", "",
              "## Supervisors' published metrics (as published, NOT comparable: different truth & protocol)", "",
              _published(cfg), "",
              "Figures: `figures/csi_ci.png`, `rmse_skill.png`, `reliability.png`, `onset.png`, `case_study.png`."]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)
