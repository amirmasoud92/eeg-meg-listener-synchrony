"""Pooled difference between correlations with the Alice word-level gain from up to 500 permutations."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from common.settings import PROJECT_ROOT
from statistical_tests.meg_attention_tests import delta_rho_boot

T = PROJECT_ROOT / "results"


def pooled_difference():
    step0 = json.loads((T / "alice_listener_tests.json").read_text())
    r7 = json.loads((T / "meg_attention_results.json").read_text())["B"]
    H = pd.read_csv(T / "alice_listeners.csv", index_col=0)
    core = pd.read_csv(T / "word_level_listeners.csv")
    stab = core[core.dataset == "alice"].set_index("listener").pooled
    d = H.assign(dr_stab=stab).dropna(subset=["comprehension", "isc", "dr_stab"])
    a_obs, a_boot = delta_rho_boot(d.comprehension, d.isc, d.dr_stab, "p7|alice|boot")
    st = step0["delta_rho"]
    a_se = float(a_boot.std(ddof=1))
    m_obs, m_se = r7["delta_rho"]["observed"], r7["delta_rho"]["se_boot"]
    w = 1 / np.array([a_se, m_se]) ** 2
    dp = float((w * np.array([a_obs, m_obs])).sum() / w.sum())
    dse = float(1 / np.sqrt(w.sum()))
    out = {
        "alice_delta_rho": {"delta_rho": a_obs, "ci95": st["ci95"], "se_boot": a_se},
        "meg_delta_rho": {"delta_rho": m_obs, "ci95": r7["delta_rho"]["ci95"], "se_boot": m_se},
        "pooled_delta_rho": {"delta_rho": dp, "ci95": [dp - 1.96 * dse, dp + 1.96 * dse], "se": dse, "p_one_sided": float(stats.norm.sf(dp / dse))},
    }
    f = T / "pooled_word_level_contrast.json"
    f.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
