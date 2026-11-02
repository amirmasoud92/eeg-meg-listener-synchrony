"""Alice: the listener-level statistics with the word-level gain from up to 500 permutations, reliability correction and the difference between correlations."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

from common.settings import load_analysis, load_paths
from synchrony import alice_comprehension as h

RSCRIPT = "Rscript"


def fisher_ci(r: float, n: int) -> list[float]:
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def alice_listener_tests():
    paths, cfg = load_paths(), load_analysis()
    seed = cfg.project.seed
    t = out = paths.results_root
    cache = paths.derived_root / "word_level"
    core = json.loads((out / "word_level.json").read_text())
    n_a = core["monte_carlo"]["alice|speech|pooled"]["n_used"]
    # ---- per-listener and per-segment delta r
    lis, seg_rows = {}, []
    for f in sorted(cache.glob("alice_speech_*.npz")):
        with np.load(f) as z:
            m = json.loads(str(z["meta"]))
            sub = m["key"].split("|")[2]
            lis[sub] = float(z["S_true_ch"].mean() - z["S_ctrl_ch"][:n_a].mean(1).mean())
            per = np.nanmean(z["S_true_per"] - z["S_ctrl_per"][:n_a].mean(0), axis=1)
        seg_rows += [{"listener": sub, "segment": int(s), "dr": float(d)} for s, d in zip(m["units"], per)]
    H = pd.read_csv(t / "alice_listeners.csv", index_col=0)
    d = H.assign(dr_stab=pd.Series(lis)).dropna(subset=["comprehension", "dr_stab"])
    res = {"n": int(len(d)), "alice_n_permutations": n_a}
    # ---- comprehension and the word-level gain
    rho, p = h.spearman_perm(d.comprehension, d.dr_stab, seed=seed)
    res["rho_comprehension_delta_r"] = {"rho": rho, "ci95_fisher": fisher_ci(rho, len(d)), "p_one_sided": p}
    # ---- difference between the correlations with synchrony and with the word-level gain
    gb = np.random.default_rng(seed)
    diffs = []
    for _ in range(10000):
        ix = gb.integers(0, len(d), len(d))
        b = d.iloc[ix]
        diffs.append(stats.spearmanr(b.comprehension, b.dr_stab).correlation - stats.spearmanr(b.comprehension, b.isc).correlation)
    diffs = np.array(diffs)
    rho_isc = float(stats.spearmanr(d.comprehension, d.isc).correlation)
    obs = rho - rho_isc
    res["delta_rho"] = {"isc_minus_dr": -obs, "ci95": sorted(float(-x) for x in (np.nanpercentile(diffs, 2.5), np.nanpercentile(diffs, 97.5))),
                        "p_one_sided": float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1)), "rho_isc": rho_isc}
    # ---- split-half reliability and corrected rho
    S = pd.DataFrame(seg_rows)
    S.to_csv(out / "alice_listener_tests_alice_delta_r_per_segment.csv", index=False)
    wide = S.pivot(index="listener", columns="segment", values="dr")
    odd = wide[[c for c in wide.columns if c % 2 == 1]].mean(1)
    even = wide[[c for c in wide.columns if c % 2 == 0]].mean(1)
    r_half = float(stats.spearmanr(odd, even, nan_policy="omit").correlation)
    rel = 2 * r_half / (1 + r_half)
    res["reliability_delta_r"] = {"r_half": r_half, "spearman_brown": rel}
    res["rho_corrected_for_reliability"] = rho / np.sqrt(rel) if rel > 0 else None
    # ---- listener share of the word-level gain
    f_in = out / "alice_delta_r_icc_input.csv"
    S.assign(dr=S.dr * 1000).to_csv(f_in, index=False)
    code = f'''suppressMessages(library(lme4)); set.seed(20260929)
icc <- function(m) {{ v <- as.data.frame(VarCorr(m)); lv <- v$vcov[v$grp %in% c("listener", "subject")]; lv / sum(v$vcov) }}
fitb <- function(m) {{ b <- bootMer(m, icc, nsim = 1000, type = "parametric"); c(icc(m), quantile(b$t, c(.025, .975), na.rm = TRUE)) }}
d <- read.csv("{f_in.as_posix()}"); d$segment <- factor(d$segment)
r <- fitb(lmer(dr ~ 1 + (1 | listener) + (1 | segment), data = d))
write.csv(data.frame(icc = r[1], lo = r[2], hi = r[3]), "{(out / 'alice_delta_r_icc.csv').as_posix()}", row.names = FALSE)
'''
    (out / "alice_delta_r_icc.R").write_text(code)
    subprocess.run([RSCRIPT, str(out / "alice_delta_r_icc.R")], check=True, capture_output=True)
    ic = pd.read_csv(out / "alice_delta_r_icc.csv").iloc[0]
    res["icc_delta_r"] = {"icc_listener": float(ic.icc), "ci95": [float(ic.lo), float(ic.hi)]}
    (out / "alice_listener_tests.json").write_text(json.dumps(res, indent=2, default=float))
