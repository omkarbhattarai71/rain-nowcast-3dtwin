"""Radar pipeline rebuilt from the raw ODIM HDF5 pseudo-CAPPI files.

Fixes relative to the supervisors' step 8/9:
  * each of the 5 radars has its own gnomonic grid -> the station/pixel mapping is computed per site
    (the old lookup belonged to the `ekxr` grid only and was applied to all five radars);
  * raw value == `undetect` means "no echo" (0 mm/h), raw == `nodata` means "unknown" (NaN);
    the old code turned undetect into -32 dBZ (a constant ~0.0004 mm/h);
  * the five radars are merged into a nearest-radar composite on a common 1 km grid;
  * features are joined to the minute grid *causally* (scan time + latency <= t), with the age kept.

On top of the composite we compute motion vectors (FFT phase correlation per tile) and a
semi-Lagrangian extrapolation nowcast for 1..max_lead minutes, which is the core of the
PySTEPS extrapolation nowcast. If `pysteps` is importable, its Lucas-Kanade optical flow and
semi-Lagrangian scheme are used as an additional nowcast (nowcast_pysteps_*).
"""
from __future__ import annotations

import io
import logging
import re
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FNAME_RE = re.compile(r"^(ek[a-z]{2})(\d{8})_(\d{4})\.")
WIN_MEAN = (3, 11, 31)


# --------------------------------------------------------------------------- ODIM reading
def read_odim(raw: bytes):
    """Return (rain_mmh float32 array, where-attrs dict)."""
    import h5py

    with h5py.File(io.BytesIO(raw), "r") as f:
        data = f["dataset1/data1/data"][()]
        what = f["dataset1/what"].attrs
        gain, offset = float(what.get("gain", 0.5)), float(what.get("offset", -32.0))
        nodata, undetect = float(what.get("nodata", 255)), float(what.get("undetect", 0))
        how = f["how"].attrs if "how" in f else {}
        a, b = float(how.get("zr-a", 200.0)), float(how.get("zr-b", 1.6))
        where = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["where"].attrs.items()}
    dbz = gain * data.astype("float32") + offset
    rain = np.power(np.power(10.0, dbz / 10.0) / a, 1.0 / b).astype("float32")
    rain[rain < 0.05] = 0.0
    rain[data == undetect] = 0.0
    rain[data == nodata] = np.nan
    return rain, where


# --------------------------------------------------------------------------- geometry
class Composite:
    """Common gnomonic 1 km grid and per-site index maps."""

    def __init__(self, cfg):
        from pyproj import Transformer

        lon0, lat0 = cfg.radar.composite_center
        self.res = float(cfg.radar.composite_res_km) * 1000.0
        half = float(cfg.radar.composite_half_size_km) * 1000.0
        self.n = int(2 * half / self.res)
        self.proj = f"+proj=gnom +ellps=WGS84 +lon_0={lon0} +lat_0={lat0}"
        self.fwd = Transformer.from_crs("EPSG:4326", self.proj, always_xy=True)
        self.inv = Transformer.from_crs(self.proj, "EPSG:4326", always_xy=True)
        c = (np.arange(self.n) + 0.5) * self.res - half
        self.xc, self.yc = np.meshgrid(c, c[::-1])           # row 0 = north
        self.lon, self.lat = self.inv.transform(self.xc, self.yc)
        self.half = half
        self.site_maps: dict[tuple, np.ndarray] = {}
        self.site_dist: dict[tuple, np.ndarray] = {}
        self.site_shape: dict[tuple, tuple[int, int]] = {}

    def add_site(self, site: str, where: dict) -> tuple:
        """Index map for one radar grid. A site can change grid (e.g. 960x960 at 500 m and
        480x480 at 1 km), so maps are cached per (site, grid geometry), not per site."""
        from pyproj import Transformer

        nx, ny = int(where.get("xsize", 960)), int(where.get("ysize", 960))
        key = (site, nx, ny, float(where["xscale"]), float(where["yscale"]),
               round(float(where["LL_lon"]), 6), round(float(where["LL_lat"]), 6), str(where["projdef"]))
        if key in self.site_maps:
            return key
        tr = Transformer.from_crs("EPSG:4326", where["projdef"], always_xy=True)
        x0, y0 = tr.transform(float(where["LL_lon"]), float(where["LL_lat"]))
        xs, ys = float(where["xscale"]), float(where["yscale"])
        x, y = tr.transform(self.lon, self.lat)
        ix = np.floor((x - x0) / xs).astype(int)
        iy = (ny - 1) - np.floor((y - y0) / ys).astype(int)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        self.site_maps[key] = np.where(inside, iy * nx + ix, -1)
        self.site_shape[key] = (ny, nx)
        m = re.search(r"lon_0=([-\d.]+).*lat_0=([-\d.]+)", where["projdef"])
        slon, slat = float(m.group(1)), float(m.group(2))
        from .stations import haversine_km

        self.site_dist[key] = np.where(inside, haversine_km(self.lon, self.lat, slon, slat), np.inf)
        return key

    def merge(self, site_fields: dict[tuple, np.ndarray]) -> np.ndarray:
        """Nearest-radar composite: per cell, the closest radar that has data.

        site_fields is keyed by the grid key returned from add_site; a field whose shape does not
        match its grid (corrupt or mislabelled file) is skipped instead of crashing the day."""
        sites = [s for s in site_fields
                 if s in self.site_maps and site_fields[s].shape == self.site_shape[s]]
        for s in site_fields:
            if s not in sites:
                log.warning("radar field %s has shape %s, grid expects %s: skipped",
                            s[0], site_fields[s].shape, self.site_shape.get(s))
        if not sites:
            return np.full((self.n, self.n), np.nan, "float32")
        V = np.full((len(sites), self.n, self.n), np.nan, "float32")
        D = np.full((len(sites), self.n, self.n), np.inf)
        for i, s in enumerate(sites):
            mp = self.site_maps[s]
            flat = site_fields[s].ravel()
            V[i] = np.where(mp >= 0, flat[np.clip(mp, 0, None)], np.nan)
            D[i] = np.where(np.isfinite(V[i]), self.site_dist[s], np.inf)
        best = np.argmin(D, axis=0)
        out = np.take_along_axis(V, best[None], 0)[0]
        out[~np.isfinite(D.min(0))] = np.nan
        return out

    def station_pixels(self, stab: pd.DataFrame) -> pd.DataFrame:
        x, y = self.fwd.transform(stab["lon"].to_numpy(), stab["lat"].to_numpy())
        col = np.floor((x + self.half) / self.res).astype(int)
        row = (self.n - 1) - np.floor((y + self.half) / self.res).astype(int)
        ok = (col >= 2) & (col < self.n - 2) & (row >= 2) & (row < self.n - 2)
        return pd.DataFrame({"row": row, "col": col, "inside": ok}, index=stab.index)


# --------------------------------------------------------------------------- motion
def _phase_corr(a: np.ndarray, b: np.ndarray):
    """Shift (dy, dx) such that b(x) ~ a(x - shift); returns (dy, dx, peak strength)."""
    wy = np.hanning(a.shape[0])[:, None]
    wx = np.hanning(a.shape[1])[None, :]
    A = np.fft.fft2(a * wy * wx)
    B = np.fft.fft2(b * wy * wx)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    py, px = np.unravel_index(np.argmax(r), r.shape)
    peak = r[py, px]

    def sub(v, n, axis_vals):
        vm, v0, vp = axis_vals
        den = vm - 2 * v0 + vp
        off = 0.5 * (vm - vp) / den if den != 0 else 0.0
        s = v + off
        return s - n if s > n / 2 else s

    ny, nx = r.shape
    dy = sub(py, ny, (r[(py - 1) % ny, px], r[py, px], r[(py + 1) % ny, px]))
    dx = sub(px, nx, (r[py, (px - 1) % nx], r[py, px], r[py, (px + 1) % nx]))
    return dy, dx, peak


def motion_field(prev: np.ndarray, curr: np.ndarray, dt_min: float, tile: int = 128, step: int = 64,
                 min_wet: float = 0.02, max_speed: float = 2.5):
    """Tile-wise advection (rows/min, cols/min) on the composite grid; global fallback.

    Vectors faster than max_speed pixels/min (2.5 km/min = 150 km/h at 1 km) are rejected as
    spurious correlation peaks (e.g. clutter or appearing/decaying cells).
    """
    a = np.log1p(np.nan_to_num(prev))
    b = np.log1p(np.nan_to_num(curr))
    n = a.shape[0]
    g = _phase_corr(a, b)
    gv = (g[0] / dt_min, g[1] / dt_min) if (b > 0).mean() > 0.005 else (0.0, 0.0)
    if np.hypot(*gv) > max_speed:
        gv = (0.0, 0.0)
    centers, vecs = [], []
    for r0 in range(0, n - tile + 1, step):
        for c0 in range(0, n - tile + 1, step):
            ta, tb = a[r0:r0 + tile, c0:c0 + tile], b[r0:r0 + tile, c0:c0 + tile]
            if (tb > 0).mean() < min_wet or (ta > 0).mean() < min_wet:
                continue
            dy, dx, pk = _phase_corr(ta, tb)
            if pk < 0.05 or np.hypot(dy, dx) / dt_min > max_speed:
                continue
            centers.append((r0 + tile / 2, c0 + tile / 2))
            vecs.append((dy / dt_min, dx / dt_min))
    return np.array(gv), np.array(centers), np.array(vecs)


def local_motion(row, col, gv, centers, vecs, radius: float = 150.0):
    """Inverse-distance weighted tile motion at a pixel (global vector as weak prior)."""
    if len(centers) == 0:
        return gv
    d = np.hypot(centers[:, 0] - row, centers[:, 1] - col)
    w = np.exp(-(d / radius) ** 2)
    wsum = w.sum() + 0.05
    return (w @ vecs + 0.05 * gv) / wsum


def _sample(field: np.ndarray, r: float, c: float, k: int = 1) -> float:
    n = field.shape[0]
    ri, ci = int(round(r)), int(round(c))
    if ri - k < 0 or ci - k < 0 or ri + k >= n or ci + k >= n:
        return np.nan
    win = field[ri - k:ri + k + 1, ci - k:ci + k + 1]
    return float(np.nanmean(win)) if np.isfinite(win).any() else np.nan


# --------------------------------------------------------------------------- per-scan features
def scan_features(comp: np.ndarray, pix: pd.DataFrame, motion, cfg, pysteps_fields=None) -> list[dict]:
    gv, centers, vecs = motion
    wet = float(cfg.radar.wet_mmh)
    max_lead = int(cfg.radar.max_lead_min)
    rows = []
    for sid, p in pix[pix.inside].iterrows():
        r, c = int(p.row), int(p.col)
        rec = {"station_id": sid, "radar_r_pix": float(comp[r, c])}
        for k in WIN_MEAN:
            h = k // 2
            win = comp[max(r - h, 0):r + h + 1, max(c - h, 0):c + h + 1]
            with np.errstate(all="ignore"):
                rec[f"radar_r_mean{k}"] = float(np.nanmean(win)) if np.isfinite(win).any() else np.nan
                rec[f"radar_r_max{k}"] = float(np.nanmax(win)) if np.isfinite(win).any() else np.nan
                rec[f"radar_wetfrac{k}"] = float(np.nanmean(win > wet)) if np.isfinite(win).any() else np.nan
        rec["radar_nodata31"] = float(np.isnan(comp[max(r - 15, 0):r + 16, max(c - 15, 0):c + 16]).mean())
        vy, vx = local_motion(r, c, gv, centers, vecs)
        rec["radar_motion_v"], rec["radar_motion_u"] = float(-vy), float(vx)   # km/min north / east
        for lead in range(1, max_lead + 1):
            rec[f"adv_{lead}"] = _sample(comp, r - vy * lead, c - vx * lead, 1)
            if pysteps_fields is not None:
                rec[f"pys_{lead}"] = _sample(pysteps_fields[lead - 1], r, c, 1)
        rows.append(rec)
    return rows


@lru_cache(maxsize=1)
def pysteps_available() -> bool:
    try:
        import pysteps  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def pysteps_extrapolate(history: list[np.ndarray], max_lead: int, dt_min: float = 10.0):
    """pysteps LK optical flow + semi-Lagrangian extrapolation at 1-minute steps."""
    from pysteps import motion
    from pysteps.extrapolation import semilagrangian

    stack = np.stack([np.nan_to_num(h) for h in history])
    v = motion.get_method("LK")(np.log1p(stack))          # pixels per 10-min step
    v = v / dt_min                                        # pixels per minute
    return semilagrangian.extrapolate(np.nan_to_num(history[-1]), v, max_lead)


# --------------------------------------------------------------------------- day processing
def _group_scans(keys: list[str]) -> dict[pd.Timestamp, dict[str, str]]:
    scans: dict[pd.Timestamp, dict[str, str]] = {}
    for k in keys:
        m = FNAME_RE.match(k.rsplit("/", 1)[-1])
        if not m:
            continue
        t = pd.Timestamp(f"{m.group(2)} {m.group(3)}", tz="UTC")
        scans.setdefault(t, {})[m.group(1)] = k
    return dict(sorted(scans.items()))


def process_day(day: date, cfg, stab: pd.DataFrame) -> pd.DataFrame:
    """All scans of one day -> per-station feature rows (one per scan time)."""
    from . import io_s3

    bucket, region = cfg.s3.radar_bucket, cfg.s3.region
    sites = set(cfg.radar.sites)
    keys = []
    for d in (day - timedelta(days=1), day):
        pre = f"{cfg.s3.radar_prefix}/{d:%Y/%m/%d}/"
        keys += [o["Key"] for o in io_s3.list_keys(bucket, pre, region) if o["Key"].endswith(".h5")]
    scans = _group_scans(keys)
    times = [t for t in scans if t.date() == day]
    if not times:
        return pd.DataFrame()
    prev_times = [t for t in scans if t.date() < day][-2:]
    comp_grid = Composite(cfg)
    pix = None
    use_pys = cfg.radar.use_pysteps in (True, "true") or (cfg.radar.use_pysteps == "auto" and pysteps_available())
    history: list[tuple[pd.Timestamp, np.ndarray]] = []
    rows = []
    for t in prev_times + times:
        fields = {}
        for site, key in scans[t].items():
            if site not in sites:
                continue
            try:
                rain, where = read_odim(io_s3.read_bytes(bucket, key, region))
            except Exception as exc:  # noqa: BLE001
                log.warning("bad radar file %s: %s", key, exc)
                continue
            try:
                fields[comp_grid.add_site(site, where)] = rain
            except (KeyError, ValueError, AttributeError) as exc:
                log.warning("radar file %s has unusable geometry: %s", key, exc)
        if not fields:
            continue
        comp = comp_grid.merge(fields)
        if pix is None:
            pix = comp_grid.station_pixels(stab)
        if history and (t - history[-1][0]) <= pd.Timedelta(minutes=30):
            dt = (t - history[-1][0]).total_seconds() / 60.0
            mot = motion_field(history[-1][1], comp, dt)
        else:
            mot = (np.zeros(2), np.zeros((0, 2)), np.zeros((0, 2)))
        history = (history + [(t, comp)])[-3:]
        if t.date() != day:
            continue
        pys = None
        if use_pys and len(history) >= 2:
            try:
                pys = pysteps_extrapolate([h[1] for h in history], int(cfg.radar.max_lead_min))
            except Exception as exc:  # noqa: BLE001
                log.debug("pysteps failed at %s: %s", t, exc)
        for rec in scan_features(comp, pix, mot, cfg, pys):
            rec["scan_time"] = t
            rec["n_sites"] = len(fields)
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- minute merge
def minute_channels(scans: pd.DataFrame, minutes: pd.DatetimeIndex, cfg) -> pd.DataFrame:
    """Causal as-of join of per-scan features to a minute grid for one station."""
    lat = pd.Timedelta(minutes=int(cfg.radar.latency_min))
    max_lead = int(cfg.radar.max_lead_min)
    s = scans.sort_values("scan_time").copy()
    s["radar_trend11"] = s["radar_r_mean11"].diff().fillna(0.0)
    s["avail_time"] = s["scan_time"] + lat
    left = pd.DataFrame({"time": minutes})
    m = pd.merge_asof(left, s, left_on="time", right_on="avail_time", direction="backward",
                      tolerance=pd.Timedelta(minutes=40))
    age = (m["time"] - m["scan_time"]).dt.total_seconds() / 60.0
    out = pd.DataFrame({"time": m["time"]})
    for c in [c for c in s.columns if c.startswith("radar_")]:
        out[c] = m[c].to_numpy(dtype="float32")
    out["radar_age_min"] = age.to_numpy(dtype="float32")
    lead = np.clip(np.nan_to_num(age.to_numpy(), nan=max_lead - 1) + 1, 1, max_lead).astype(int)
    for prefix, name in (("adv", "nowcast_adv"), ("pys", "nowcast_pysteps")):
        cols = [f"{prefix}_{L}" for L in range(1, max_lead + 1)]
        if cols[0] not in m:
            continue
        A = m[cols].to_numpy(dtype="float32")
        out[f"{name}_t1"] = A[np.arange(len(A)), lead - 1]
        lead5 = np.clip(lead + 4, 1, max_lead)
        out[f"{name}_t5"] = A[np.arange(len(A)), lead5 - 1]
    return out
