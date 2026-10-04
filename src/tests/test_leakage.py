"""Future-perturbation tests: changing anything after time t must not change inputs at time <= t."""
import numpy as np
import pandas as pd

from conftest import synthetic_obs
from rainnow.channels import build_all_channels
from rainnow.features import build_features
from rainnow.radar import minute_channels


def _channels(obs, stations, cfg):
    return build_all_channels(obs, list(obs), stations, cfg)


def test_channels_and_features_are_causal(cfg, stations):
    obs = {s: synthetic_obs(i) for i, s in enumerate(stations.index)}
    cut = obs["00001"].index[400]
    perturbed = {}
    for i, (s, df) in enumerate(obs.items()):
        d = df.copy()
        future = d.index > cut
        d.loc[future] = (np.random.default_rng(100 + i).random((future.sum(), d.shape[1])) * 3).astype("float32")
        perturbed[s] = d
    a = _channels(obs, stations, cfg)
    b = _channels(perturbed, stations, cfg)
    for s in stations.index:
        pd.testing.assert_frame_equal(a[s].loc[:cut], b[s].loc[:cut])
        fa = build_features(a[s], cfg).loc[:cut]
        fb = build_features(b[s], cfg).loc[:cut]
        pd.testing.assert_frame_equal(fa, fb)


def test_target_not_in_features(cfg, stations):
    obs = {s: synthetic_obs(i) for i, s in enumerate(stations.index)}
    ch = _channels(obs, stations, cfg)["00001"]
    feats = build_features(ch, cfg)
    target = obs["00001"]["p1_obs"].fillna(0).shift(-1)
    corr = feats.apply(lambda c: np.corrcoef(np.nan_to_num(c), np.nan_to_num(target))[0, 1] if c.std() > 0 else 0)
    assert corr.abs().max() < 0.95, corr.abs().sort_values().tail()


def test_radar_minute_merge_respects_latency(cfg):
    scans = pd.DataFrame({
        "scan_time": pd.date_range("2025-07-01 00:00", periods=6, freq="10min", tz="UTC"),
        "radar_r_mean11": [0, 1, 2, 3, 4, 5.0],
        "radar_r_pix": [0, 1, 2, 3, 4, 5.0],
    })
    for L in range(1, int(cfg.radar.max_lead_min) + 1):
        scans[f"adv_{L}"] = scans["radar_r_pix"] + L / 100
    minutes = pd.date_range("2025-07-01 00:00", periods=60, freq="1min", tz="UTC")
    mc = minute_channels(scans, minutes, cfg).set_index("time")
    lat = int(cfg.radar.latency_min)
    for t, r in mc.dropna(subset=["radar_r_pix"]).iterrows():
        scan_used = t - pd.Timedelta(minutes=float(r["radar_age_min"]))
        assert scan_used + pd.Timedelta(minutes=lat) <= t
    # a scan must not be visible before scan_time + latency
    assert np.isnan(mc.loc[minutes[lat - 1], "radar_r_pix"])
    assert mc.loc[minutes[lat], "radar_r_pix"] == 0.0
