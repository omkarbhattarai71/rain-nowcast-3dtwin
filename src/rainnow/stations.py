"""Station metadata: coordinates, Denmark filter, distances, bearings and neighbour sets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0


def norm_id(s: pd.Series | list | str) -> pd.Series | str:
    if isinstance(s, str):
        return "".join(ch for ch in s if ch.isdigit()).zfill(5)
    s = pd.Series(s).astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.str.replace(r"\D+", "", regex=True).str.zfill(5)


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def bearing_deg(lon1, lat1, lon2, lat2):
    """Initial bearing from point 1 to point 2 (0 = north, 90 = east)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def load_station_table(cfg) -> pd.DataFrame:
    """Merge the DMI metadata file and the radar lookup into one coordinate table."""
    meta_dir = cfg.path("raw", "meta")
    frames = []
    meta_file = meta_dir / "stations_metObs_v1.csv"
    if meta_file.exists():
        m = pd.read_csv(meta_file, encoding_errors="replace")
        m["station_id"] = norm_id(m["station_id"])
        frames.append(m[["station_id", "lon", "lat"] + (["name"] if "name" in m else [])])
    lookup_file = meta_dir / "stations_xy_lookup.csv"
    if lookup_file.exists():
        lk = pd.read_csv(lookup_file)
        lk["station_id"] = norm_id(lk["station_id"])
        frames.append(lk[["station_id", "lon", "lat"]])
    if not frames:
        raise FileNotFoundError(f"No station metadata in {meta_dir}; run step01 first.")
    tab = pd.concat(frames, ignore_index=True)
    tab = tab.dropna(subset=["lon", "lat"]).drop_duplicates("station_id", keep="first")
    return tab.set_index("station_id").sort_index()


def select_stations(cfg, available: list[str]) -> list[str]:
    """Apply include / exclude / bbox / has-coordinates rules."""
    tab = load_station_table(cfg)
    lon0, lat0, lon1, lat1 = cfg.stations.bbox
    out = []
    for sid in sorted(set(available)):
        if cfg.stations.include and sid not in cfg.stations.include:
            continue
        if any(sid.startswith(p) for p in cfg.stations.exclude_prefixes):
            continue
        if sid not in tab.index:
            continue
        r = tab.loc[sid]
        if lon0 <= r.lon <= lon1 and lat0 <= r.lat <= lat1:
            out.append(sid)
    return out


def neighbour_table(tab: pd.DataFrame, stations: list[str], max_km: float) -> pd.DataFrame:
    """Long table (station, neighbour, dist_km, bearing_deg) for all pairs within max_km."""
    t = tab.loc[stations]
    rows = []
    for sid, r in t.iterrows():
        d = haversine_km(r.lon, r.lat, t.lon.values, t.lat.values)
        b = bearing_deg(r.lon, r.lat, t.lon.values, t.lat.values)
        for nid, dd, bb in zip(t.index, d, b):
            if nid != sid and dd <= max_km:
                rows.append((sid, nid, float(dd), float(bb)))
    return pd.DataFrame(rows, columns=["station_id", "neighbour_id", "dist_km", "bearing_deg"])


def raw_station_ids(raw_dir: Path, suffix: str) -> list[str]:
    return sorted(p.name[:5] for p in Path(raw_dir).glob(f"*{suffix}"))
