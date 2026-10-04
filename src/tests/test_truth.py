import numpy as np
import pandas as pd

from rainnow.truth import build_truth


def _raw(p1: dict, p10: dict):
    times = sorted(set(p1) | set(p10))
    df = pd.DataFrame(index=pd.DatetimeIndex(times, name="time"))
    df["precip_past1min"] = pd.Series(p1, dtype=float)
    df["precip_past10min"] = pd.Series(p10, dtype=float)
    return df


def t(m):
    return pd.Timestamp("2025-01-01 00:00", tz="UTC") + pd.Timedelta(minutes=m)


def test_window_rules(cfg):
    p1 = {t(12): 0.2, t(13): 0.3,          # window ending 20: sum 0.5 == p10 0.5 -> valid, gaps = 0
          t(22): 0.1,                      # window ending 30: p10 = 1.0, 1-min sum 0.1 -> invalid
          t(41): 0.1}                      # window ending 50: p10 = 0, one tip -> valid (dry tolerance)
    p10 = {t(0): 0.0, t(10): 0.0, t(20): 0.5, t(30): 1.0, t(40): 0.0, t(50): 0.0, t(60): 0.0}
    tr = build_truth(_raw(p1, p10), cfg)
    assert tr.loc[t(1):t(10), "valid"].all() and (tr.loc[t(1):t(10), "precip"] == 0).all()
    w2 = tr.loc[t(11):t(20)]
    assert w2["valid"].all()
    assert np.isclose(w2["precip"].sum(), 0.5)
    assert tr.loc[t(14), "precip"] == 0.0           # unreported minute in a mass-consistent window
    assert not tr.loc[t(21):t(30), "valid"].any()     # missing rain must not become "no rain"
    assert tr.loc[t(21):t(30), "precip"].isna().all()
    assert tr.loc[t(41):t(50), "valid"].all()


def test_dense_without_10min(cfg):
    p1 = {t(m): 0.0 for m in range(1, 11)}
    p1[t(5)] = 0.4
    tr = build_truth(_raw(p1, {t(0): 0.0}), cfg)
    assert tr.loc[t(1):t(10), "valid"].all()
    assert np.isclose(tr.loc[t(1):t(10), "precip"].sum(), 0.4)


def test_sentinels_removed(cfg):
    tr = build_truth(_raw({t(3): -999999.0}, {t(0): 0.0, t(10): 0.0}), cfg)
    assert np.isnan(tr.loc[t(3), "p1_obs"])


def test_tip_harmonisation():
    from rainnow.truth import harmonise_tips

    idx = pd.date_range("2025-01-01", periods=6, freq="1min", tz="UTC")
    fine = pd.Series([0.03, 0.04, np.nan, 0.05, 0.0, 0.2], index=idx)
    out = harmonise_tips(fine, 0.1)
    assert np.isclose(np.nansum(out), 0.3)                    # 0.32 mm -> 3 whole tips
    assert np.isnan(out.iloc[2])                              # missing stays missing
    coarse = pd.Series([0.1, 0.0, 0.3, np.nan, 0.2, 0.0], index=idx)
    pd.testing.assert_series_equal(harmonise_tips(coarse, 0.1).round(6), coarse.round(6))
