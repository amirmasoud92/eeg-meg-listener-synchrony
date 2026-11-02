"""Evidence for the null results and the variance components of synchrony.

Bayes factors and equivalence tests from dz and n (both depend only on t and n, so they are evaluated on a sample with
exactly that dz and n), a fixed-effect combination of the within-listener tests, and listener intraclass correlations
(lme4, parametric bootstrap).
"""
from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
from scipy import stats

from statistical_tests.inference import jzs_bf10, sesoi_resource_based, tost_one_sample

RSCRIPT = "Rscript"


def synthetic(dz: float, n: int) -> np.ndarray:
    z = np.random.default_rng(0).normal(size=n)
    return (z - z.mean()) / z.std(ddof=1) + dz


def absence(tests: dict[str, dict]) -> dict:
    """`tests`: for each within-listener test its n and dz (keys D, E1, E2, I1, I3: seeing the talker, attention,
    run-wise comprehension, engagement, alpha)."""
    out = {}
    for k, s in tests.items():
        n, dz = int(s["n"]), float(s["dz"])
        x = synthetic(dz, n)
        sesoi = sesoi_resource_based(n, 0.8, 0.05, "larger")
        out[k] = {"n": n, "dz": dz, "bf01_two_sided": float(1 / jzs_bf10(x, alternative="two-sided")),
                  "bf0plus": float(1 / jzs_bf10(x, alternative="greater")), "tost_0.5_p": tost_one_sample(x, 0.5).p,
                  "sesoi_80pct_power": sesoi, "tost_sesoi_p": tost_one_sample(x, sesoi).p}
    ks = ["D", "E2", "I1", "I3"]
    dz = np.array([out[k]["dz"] for k in ks])
    n = np.array([out[k]["n"] for k in ks], float)
    w = 1 / (1 / n + dz ** 2 / (2 * n))
    pooled, se = float((w * dz).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
    out["pooled_D_E2_I1_I3"] = {"dz": pooled, "se": se, "ci95": [pooled - 1.96 * se, pooled + 1.96 * se],
                                "p_one_sided": float(stats.norm.sf(pooled / se))}
    return out


def icc(t, out) -> dict:
    ex = t
    dr = pd.read_csv(ex / "alice_delta_r_per_segment.csv").melt(id_vars="listener", var_name="segment", value_name="dr").dropna()
    dr.to_csv(out / "alice_delta_r_long.csv", index=False)
    code = f'''suppressMessages(library(lme4)); set.seed(20260929)
icc <- function(m) {{ v <- as.data.frame(VarCorr(m)); lv <- v$vcov[v$grp %in% c("listener", "subject")]; lv / sum(v$vcov) }}
fitb <- function(m) {{ b <- bootMer(m, icc, nsim = 1000, type = "parametric"); c(icc(m), quantile(b$t, c(.025, .975), na.rm = TRUE)) }}
a <- read.csv("{(ex / "alice_isc_per_segment.csv").as_posix()}"); a$segment <- factor(a$segment)
r1 <- fitb(lmer(isc ~ 1 + (1 | listener) + (1 | segment), data = a))
d <- read.csv("{(out / "alice_delta_r_long.csv").as_posix()}"); d$dr <- d$dr * 1000
r2 <- fitb(lmer(dr ~ 1 + (1 | listener) + (1 | segment), data = d))
b <- read.csv("{(t / "aalborg_isc_speech_space.csv").as_posix()}"); b$subject <- factor(b$subject)
r3 <- fitb(lmer(isc ~ condition + (1 | subject) + (1 | stimulus), data = b))
write.csv(rbind(alice_isc = r1, alice_delta_r = r2, aalborg_isc = r3), "{(out / "listener_icc.csv").as_posix()}")
'''
    (out / "listener_icc.R").write_text(code)
    subprocess.run([RSCRIPT, str(out / "listener_icc.R")], check=True, capture_output=True)
    r = pd.read_csv(out / "listener_icc.csv", index_col=0)
    return {k: {"icc_listener": float(v.iloc[0]), "ci95": [float(v.iloc[1]), float(v.iloc[2])]} for k, v in r.iterrows()}
