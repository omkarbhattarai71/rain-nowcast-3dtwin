"""Rain attenuation following ITU-R recommendations.

* ITU-R P.838-3  specific attenuation  gamma = k * R^alpha  [dB/km]
* ITU-R P.530    terrestrial line-of-sight links: effective path length via distance factor r
* ITU-R P.618    earth-space (satellite) slant path, step-by-step method

Rain rate R is in mm/h. A 1-minute accumulation y (mm) corresponds to R = 60 * y.
The P.530 / P.618 path-reduction factors are defined for the 0.01 %-exceeded rain rate; applying
them to instantaneous rates is the usual engineering approximation and is documented as such.
"""
from __future__ import annotations

import numpy as np

# P.838-3 Tables 1-4: (a_j, b_j, c_j) and (m, c)
_KH = ([(-5.33980, -0.10008, 1.13098), (-0.35351, 1.26970, 0.45400),
        (-0.23789, 0.86036, 0.15354), (-0.94158, 0.64552, 0.16817)], -0.18961, 0.71147)
_KV = ([(-3.80595, 0.56934, 0.81061), (-3.44965, -0.22911, 0.51059),
        (-0.39902, 0.73042, 0.11899), (0.50167, 1.07319, 0.27195)], -0.16398, 0.63297)
_AH = ([(-0.14318, 1.82442, -0.55187), (0.29591, 0.77564, 0.19822), (0.32177, 0.63773, 0.13164),
        (-5.37610, -0.96230, 1.47828), (16.1721, -3.29980, 3.43990)], 0.67849, -1.95537)
_AV = ([(-0.07771, 2.33840, -0.76284), (0.56727, 0.95545, 0.54039), (-0.20238, 1.14520, 0.26809),
        (-48.2991, 0.791669, 0.116226), (48.5833, 0.791459, 0.116479)], -0.053739, 0.83433)


def _curve(f_ghz: float, table) -> float:
    terms, m, c = table
    lf = np.log10(f_ghz)
    return sum(a * np.exp(-(((lf - b) / cc) ** 2)) for a, b, cc in terms) + m * lf + c


def p838_coefficients(f_ghz: float, pol: str = "V", elevation_deg: float = 0.0) -> tuple[float, float]:
    """(k, alpha) for frequency 1-1000 GHz. pol: 'H', 'V', 'circular' or a tilt angle in degrees."""
    kh = 10 ** _curve(f_ghz, _KH)
    kv = 10 ** _curve(f_ghz, _KV)
    ah = _curve(f_ghz, _AH)
    av = _curve(f_ghz, _AV)
    tau = {"H": 0.0, "V": 90.0, "circular": 45.0}.get(str(pol), None)
    tau = float(pol) if tau is None else tau
    th = np.radians(elevation_deg)
    c2 = np.cos(th) ** 2 * np.cos(2 * np.radians(tau))
    k = (kh + kv + (kh - kv) * c2) / 2
    alpha = (kh * ah + kv * av + (kh * ah - kv * av) * c2) / (2 * k)
    return float(k), float(alpha)


def specific_attenuation(rain_mmh, f_ghz: float, pol: str = "V", elevation_deg: float = 0.0):
    k, a = p838_coefficients(f_ghz, pol, elevation_deg)
    r = np.clip(np.asarray(rain_mmh, float), 0, None)
    return k * r**a


def terrestrial_attenuation(rain_mmh, f_ghz: float, d_km: float, pol: str = "V"):
    """Path attenuation [dB] of a terrestrial link (ITU-R P.530-17/18 distance factor)."""
    r_mmh = np.clip(np.asarray(rain_mmh, float), 0, None)
    k, a = p838_coefficients(f_ghz, pol)
    gamma = k * r_mmh**a
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 0.477 * d_km**0.633 * np.where(r_mmh > 0, r_mmh, 1.0) ** (0.073 * a) * f_ghz**0.123 \
            - 10.579 * (1 - np.exp(-0.024 * d_km))
        r = np.where(denom > 0, 1.0 / denom, 2.5)
    r = np.clip(r, None, 2.5)
    return gamma * d_km * r


def earth_space_attenuation(rain_mmh, f_ghz: float, elevation_deg: float, lat_deg: float,
                            rain_height_km: float = 2.6, station_height_km: float = 0.05,
                            pol: str = "circular"):
    """Slant-path rain attenuation [dB], ITU-R P.618 steps 2-9 with R in place of R0.01."""
    r_mmh = np.clip(np.asarray(rain_mmh, float), 0, None)
    th = np.radians(elevation_deg)
    hr, hs = rain_height_km, station_height_km
    if hr <= hs:
        return np.zeros_like(r_mmh)
    ls = (hr - hs) / np.sin(th)                       # slant path below rain height (theta >= 5 deg)
    lg = ls * np.cos(th)                              # horizontal projection
    k, a = p838_coefficients(f_ghz, pol, elevation_deg)
    gamma = k * r_mmh**a
    with np.errstate(divide="ignore", invalid="ignore"):
        r001 = 1.0 / (1 + 0.78 * np.sqrt(np.where(gamma > 0, lg * gamma / f_ghz, 0.0)) - 0.38 * (1 - np.exp(-2 * lg)))
        zeta = np.degrees(np.arctan2(hr - hs, lg * r001))
        lr = np.where(zeta > elevation_deg, lg * r001 / np.cos(th), (hr - hs) / np.sin(th))
        chi = 36 - abs(lat_deg) if abs(lat_deg) < 36 else 0.0
        v = 1.0 / (1 + np.sqrt(np.sin(th)) * (31 * (1 - np.exp(-(elevation_deg / (1 + chi))))
                                              * np.sqrt(np.where(gamma > 0, lr * gamma, 0.0)) / f_ghz**2 - 0.45))
    le = lr * v
    return np.where(r_mmh > 0, gamma * le, 0.0)


def scenario_attenuation(rain_mm_per_min, scenario: dict, lat_deg: float = 56.0, rain_height_km: float = 2.6):
    """Attenuation [dB] for one configured link scenario from 1-minute rain amounts (mm)."""
    r = 60.0 * np.clip(np.asarray(rain_mm_per_min, float), 0, None)
    if "d_km" in scenario:
        return terrestrial_attenuation(r, scenario["f_ghz"], scenario["d_km"], scenario.get("pol", "V"))
    return earth_space_attenuation(r, scenario["f_ghz"], scenario["elevation_deg"], lat_deg,
                                   rain_height_km, scenario.get("station_height_km", 0.05),
                                   scenario.get("pol", "circular"))
