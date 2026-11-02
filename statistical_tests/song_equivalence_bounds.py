"""Evidence bounds for the speech-specific excess L: effect size, Bayes factor and equivalence tests."""
from __future__ import annotations

import json

import pandas as pd

from common.settings import load_analysis, load_paths
from statistical_tests.inference import dz_ci, jzs_bf10, sesoi_resource_based, tost_one_sample

def excess_bounds():
    paths, cfg = load_paths(), load_analysis()
    out = paths.results_root
    res = {}
    for name, f in (("primary_items_L", "song_excess_primary.csv"), ("envelope_removed_items_L", "song_excess_envelope_removed.csv")):
        L = pd.read_csv(out / f).groupby("song").L.mean().to_numpy()
        t = tost_one_sample(L, 0.5)
        res[name] = {"n": int(L.size), "dz": float(L.mean() / L.std(ddof=1)),
                     "bf_plus0": float(jzs_bf10(L, alternative="greater")), "bf_01": float(1 / jzs_bf10(L, alternative="two-sided")),
                     "dz_ci90": [float(v) for v in dz_ci(L, 0.90)],
                     "tost_bound_0.5": {"p_lower": t.p_lower, "p_upper": t.p_upper, "p": t.p},
                     "dz_80pct_power_one_sided": float(sesoi_resource_based(L.size, 0.8, 0.05, "larger"))}
    (out / "song_excess_bounds.json").write_text(json.dumps(res, indent=2, default=str))
