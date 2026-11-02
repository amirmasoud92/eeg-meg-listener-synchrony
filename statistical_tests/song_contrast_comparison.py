"""Comparison of song contrasts: item-level effect sizes of the word-level contrast and of the synchrony excess.

The difference D is tested with 10^4 paired bootstrap resamples over the 18 songs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.settings import load_paths
from statistical_tests.inference import jzs_bf10, sesoi_resource_based, summarise, tost_one_sample

def paired(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = summarise(x, "two-sided")
    sesoi = sesoi_resource_based(x.size, 0.8, 0.05, "larger")
    return {"n": int(x.size), "mean": float(x.mean()), "dz": s["dz"], "dz_ci95": s["dz_ci95"], "p_two_sided": s["p_signflip"],
            "tost_0.5_p": tost_one_sample(x, 0.5).p, "sesoi_80pct_power": sesoi, "tost_sesoi_p": tost_one_sample(x, sesoi).p,
            "bf01_two_sided": float(1 / jzs_bf10(x, alternative="two-sided"))}


def listener_level_equivalence(L: pd.DataFrame, S: pd.DataFrame | None) -> dict:
    wide = L.pivot(index="listener", columns="version", values="pooled")
    out = {"by_listeners": paired((wide["song"] - wide["speech"]).dropna())}
    if S is not None:
        item = S.groupby(["song", "version"]).dr.mean().unstack()
        out["by_songs"] = paired((item["song"] - item["speech"]).dropna())
    return out


def item_level_difference(song_means: pd.DataFrame, seed: int) -> dict:
    """D = dz(sung - hummed delta r over songs) - dz(L over songs), paired bootstrap over songs."""
    paths = load_paths()
    tc = pd.read_csv(paths.results_root / "song_excess_primary.csv", dtype={"song": str})
    Lsong = tc.groupby("song").L.mean()
    Lsong.index = Lsong.index.str.zfill(2)
    dr = (song_means["song"] - song_means["melody"]).dropna()
    dr.index = dr.index.astype(str).str.zfill(2)
    both = pd.concat([dr.rename("dr"), Lsong.rename("L")], axis=1).dropna()
    dz = lambda x: float(np.mean(x) / np.std(x, ddof=1))
    obs = dz(both.dr) - dz(both.L)
    rng = np.random.default_rng([seed, 6, 72])
    boot = []
    for _ in range(10000):
        ix = rng.integers(0, len(both), len(both))
        b = both.iloc[ix]
        boot.append(dz(b.dr) - dz(b.L))
    boot = np.array(boot)
    return {"n_songs": int(len(both)), "dz_dr_sung_minus_hummed": dz(both.dr), "dz_L": dz(both.L), "D": obs,
            "ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
            "p_one_sided": float((np.sum(boot <= 0) + 1) / (len(boot) + 1))}
