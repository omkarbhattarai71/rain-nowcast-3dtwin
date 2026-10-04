"""Step 12 - rain forecasts -> link attenuation (ITU-R P.838-3 / P.530 / P.618).

For each link scenario and each selected model: attenuation error in dB and fade-margin
exceedance skill (the decision a network controller acts on). The "observed" attenuation is
computed from the gauge's next-minute rain with the same ITU-R model.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..attenuation import p838_coefficients, scenario_attenuation
from ..dataset import index_path
from ..metrics import categorical_scores, contingency
from ..stations import load_station_table

log = logging.getLogger(__name__)


def _models_to_analyse(cfg, bench) -> list[str]:
    ev = cfg.path(f"bench{bench}", "eval", results=True)
    met = pd.read_csv(ev / "metrics_test.csv")
    pick = ["persistence", met.iloc[0]["model"]]
    for fam in ("SAMBA (deep SSM)", "classical SSM", "gradient boosting", "deep SSM", "deep (non-SSM)"):
        sub = met[met.family == fam]
        if len(sub):
            pick.append(sub.iloc[0]["model"])
    return [m for m in dict.fromkeys(pick) if m in set(met.model)]


def run(cfg, args) -> None:
    out_root = cfg.path("attenuation", results=True, mkdir=True)
    out_root.mkdir(parents=True, exist_ok=True)
    scen = list(cfg.attenuation.terrestrial) + list(cfg.attenuation.earth_space)
    coef = []
    for s in scen:
        k, a = p838_coefficients(s["f_ghz"], s.get("pol", "V"), s.get("elevation_deg", 0.0))
        coef.append({"scenario": s["name"], "f_ghz": s["f_ghz"], "pol": s.get("pol"), "k": k, "alpha": a,
                     "A_dB_at_10mmh": float(scenario_attenuation(10 / 60, s, rain_height_km=cfg.attenuation.rain_height_km)),
                     "A_dB_at_50mmh": float(scenario_attenuation(50 / 60, s, rain_height_km=cfg.attenuation.rain_height_km))})
    pd.DataFrame(coef).to_csv(out_root / "itu_coefficients.csv", index=False)
    stab = load_station_table(cfg)
    for bench in args.bench:
        try:
            idx = pd.read_parquet(index_path(cfg, bench, "test"))
            models = _models_to_analyse(cfg, bench)
        except FileNotFoundError:
            continue
        rows, series = [], {}
        for m in models:
            p = idx.merge(pd.read_parquet(cfg.path(f"bench{bench}", "preds", m, "test.parquet", results=True)),
                          on=["station_id", "time"])
            la = p["station_id"].map(stab["lat"]).fillna(56.0).to_numpy()
            for s in scen:
                lat_s = float(np.median(la))
                a_obs = scenario_attenuation(p["y_true"].to_numpy(), s, lat_s, cfg.attenuation.rain_height_km)
                a_hat = scenario_attenuation(p["y_hat"].to_numpy(), s, lat_s, cfg.attenuation.rain_height_km)
                q = p["q90"].to_numpy()
                a_q = scenario_attenuation(q, s, lat_s, cfg.attenuation.rain_height_km) if np.isfinite(q).all() else None
                e = a_hat - a_obs
                wet = a_obs > 0.1
                r = {"bench": bench, "model": m, "scenario": s["name"],
                     "rmse_db": float(np.sqrt(np.mean(e**2))), "mae_db": float(np.mean(np.abs(e))),
                     "bias_db": float(e.mean()),
                     "rmse_db_when_fading": float(np.sqrt(np.mean(e[wet] ** 2))) if wet.any() else np.nan,
                     "corr": float(np.corrcoef(a_obs, a_hat)[0, 1]) if a_hat.std() > 0 and a_obs.std() > 0 else np.nan}
                for mg in cfg.attenuation.margins_db:
                    ev = a_obs >= mg
                    sc = categorical_scores(contingency(ev, a_hat >= mg))
                    r |= {f"pod_{mg}dB": sc["pod"], f"far_{mg}dB": sc["far"], f"csi_{mg}dB": sc["csi"],
                          f"events_{mg}dB": int(ev.sum())}
                    if a_q is not None:
                        sq = categorical_scores(contingency(ev, a_q >= mg))
                        r |= {f"pod_{mg}dB_q90": sq["pod"], f"csi_{mg}dB_q90": sq["csi"]}
                rows.append(r)
                if s is scen[1]:
                    series[m] = (p, a_obs, a_hat)
        res = pd.DataFrame(rows)
        res.to_csv(out_root / f"attenuation_bench{bench}.csv", index=False)
        _plot(cfg, bench, res, series, out_root)
        log.info("bench %s attenuation (%s):\n%s", bench, scen[1]["name"],
                 res[res.scenario == scen[1]["name"]].drop(columns=["bench", "scenario"]).round(3).to_string(index=False))
    (out_root / "scenarios.json").write_text(json.dumps(scen, indent=1))


def _plot(cfg, bench, res, series, out_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    margins = list(cfg.attenuation.margins_db)
    for m, g in res.groupby("model"):
        g = g.set_index("scenario")
        ax.plot([f"{s}\n{mg}dB" for s in g.index for mg in margins],
                [g.loc[s, f"csi_{mg}dB"] for s in g.index for mg in margins], marker="o", label=m)
    ax.set_ylabel("CSI of fade-margin exceedance")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_root / f"exceedance_csi_bench{bench}.png", dpi=150)
    plt.close(fig)
    if not series:
        return
    p, a_obs, _ = next(iter(series.values()))
    day_tot = p.assign(a=a_obs).groupby(["station_id", "day"])["a"].sum()
    if day_tot.max() <= 0:
        return
    sid, day = day_tot.idxmax()
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (m, (pp, ao, ah)) in enumerate(series.items()):
        sel = ((pp["station_id"] == sid) & (pp["day"] == day)).to_numpy()
        t = pp.loc[sel, "time"] + pd.Timedelta(minutes=1)
        if i == 0:
            ax.plot(t, ao[sel], "k", lw=1.5, label="from observed rain")
        ax.plot(t, ah[sel], lw=1, label=m)
    ax.set_ylabel("attenuation [dB]")
    ax.set_title(f"38 GHz, 1 km link at station {sid}, {day}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_root / f"case_study_bench{bench}.png", dpi=150)
    plt.close(fig)
