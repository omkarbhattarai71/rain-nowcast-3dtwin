import numpy as np
import pytest

from rainnow.metrics import categorical_scores, contingency, diebold_mariano, tune_threshold
from rainnow.models.ssm_classic import LTIKalman, RegimeHMM
from rainnow.radar import motion_field


def test_categorical_scores():
    obs = np.array([1, 1, 0, 0, 1], bool)
    fc = np.array([1, 0, 1, 0, 1], bool)
    s = categorical_scores(contingency(obs, fc))
    assert s["pod"] == pytest.approx(2 / 3) and s["far"] == pytest.approx(1 / 3) and s["csi"] == pytest.approx(0.5)


def test_tune_threshold_finds_separator():
    rng = np.random.default_rng(0)
    obs = rng.random(5000) < 0.1
    score = np.where(obs, 0.7, 0.2) + rng.normal(0, 0.05, 5000)
    thr = tune_threshold(score, obs)
    assert 0.3 < thr < 0.65


def test_steady_state_kalman_matches_explicit_filter():
    rng = np.random.default_rng(1)
    z = np.cumsum(rng.normal(0, 0.3, 3000)) + rng.normal(0, 0.5, 3000)
    A, H, Q, R = np.eye(1) * 0.95, np.eye(1), np.eye(1) * 0.09, 0.25
    kf = LTIKalman(A, H, Q, R).prepare()
    fast = kf.predict_next(z)
    K, _ = kf.gain()
    x, slow = 0.0, []
    for v in z:                        # explicit steady-state predictor recursion
        x = A[0, 0] * x + K[0, 0] * (v - x)
        slow.append(x)
    np.testing.assert_allclose(fast, slow, atol=1e-8)


def test_hmm_filter_probabilities():
    hmm = RegimeHMM(3)
    y = np.r_[np.zeros(50), np.full(20, 0.3), np.zeros(30)]
    p, yh, probs = hmm.filter_forecast([y])[0]
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-5)
    assert p[60] > p[10]               # rain probability rises during the wet spell


def test_motion_sign():
    f = np.zeros((256, 256))
    f[100:120, 100:120] = 5.0
    g = np.roll(np.roll(f, 6, axis=0), -4, axis=1)   # moved 6 rows south, 4 cols west in 10 min
    gv, _, _ = motion_field(f, g, 10.0)
    assert gv[0] == pytest.approx(0.6, abs=0.15) and gv[1] == pytest.approx(-0.4, abs=0.15)


def test_dm_test_detects_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 0.1, 200)
    b = rng.normal(1.3, 0.1, 200)
    assert diebold_mariano(a, b)["p_value"] < 0.01


def test_deep_nets_forward(cfg):
    torch = pytest.importorskip("torch")
    from rainnow.config import Cfg
    from rainnow.models.deep.nets import build_net

    c = Cfg.wrap({"deep": {**cfg.deep, "d_model": 16, "d_state": 4, "n_layers": 1, "window": 60}})
    x = torch.randn(3, 60, 7)
    for name in ["samba", "samba_nonlinear", "samba_temporal_only", "mamba", "s4d", "gru", "tcn", "patchtst"]:
        logit, amt, q = build_net(name, 7, c)(x)
        assert logit.shape == (3,) and (amt >= 0).all() and (q >= 0).all(), name
