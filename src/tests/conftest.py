import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rainnow.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def stations():
    return pd.DataFrame({"lon": [9.0, 9.2, 9.1, 9.4], "lat": [56.0, 56.1, 55.9, 56.05]},
                        index=pd.Index(["00001", "00002", "00003", "00004"], name="station_id"))


def synthetic_obs(seed: int, n: int = 600, start="2024-07-01 00:00") -> pd.DataFrame:
    """Raw observation frame (minute grid) as produced by truth.build_truth."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC", name="time")
    p1 = np.where(rng.random(n) < 0.2, rng.choice([0.1, 0.2, 0.3], n), 0.0)
    p1[rng.random(n) < 0.3] = np.nan                         # unreported minutes
    df = pd.DataFrame(index=idx)
    df["p1_obs"] = p1
    p10 = pd.Series(np.nan, index=idx)
    ends = idx[idx.minute % 10 == 0]
    for t in ends:
        p10[t] = np.nansum(p1[max(0, idx.get_loc(t) - 9): idx.get_loc(t) + 1])
    df["p10_obs"] = p10
    for c, base in (("temp_dry", 15.0), ("humidity", 80.0), ("wind_speed", 5.0), ("wind_dir", 250.0), ("cloud_cover", 90.0)):
        v = pd.Series(np.nan, index=idx)
        v[ends] = base + rng.normal(0, 1, len(ends))
        df[c] = v
    return df.astype("float32")
