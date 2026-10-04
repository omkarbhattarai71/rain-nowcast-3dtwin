import numpy as np
import pytest

from rainnow.attenuation import earth_space_attenuation, p838_coefficients, specific_attenuation, terrestrial_attenuation

# reference values from the ITU-R P.838-3 coefficient table
REF = {10: (0.01217, 1.2571, 0.01129, 1.2156), 20: (0.09164, 1.0568, 0.09611, 0.9847),
       30: (0.2403, 0.9485, 0.2291, 0.9129)}


@pytest.mark.parametrize("f", sorted(REF))
def test_p838_table(f):
    kh, ah, kv, av = REF[f]
    k, a = p838_coefficients(f, "H")
    assert k == pytest.approx(kh, rel=2e-3) and a == pytest.approx(ah, rel=2e-3)
    k, a = p838_coefficients(f, "V")
    assert k == pytest.approx(kv, rel=2e-3) and a == pytest.approx(av, rel=2e-3)


def test_monotone_and_zero():
    r = np.array([0, 1, 5, 20, 80.0])
    for f in (23, 38, 80):
        g = specific_attenuation(r, f)
        assert g[0] == 0 and np.all(np.diff(g) > 0)
        a = terrestrial_attenuation(r, f, 2.0)
        assert a[0] == 0 and np.all(np.diff(a) > 0)
    s = earth_space_attenuation(r, 20, 25, 56.0)
    assert s[0] == 0 and np.all(np.diff(s) > 0)


def test_effective_path_shorter_than_link_for_heavy_rain():
    d = 5.0
    a = terrestrial_attenuation(100.0, 38, d)
    assert a < specific_attenuation(100.0, 38) * d
