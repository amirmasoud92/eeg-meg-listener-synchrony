"""MEG-MASC tests: envelope tracking in each session, synchrony across days with its partial correlation, sentences against lists, and variance components."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

from readers import meg_masc as C
from statistical_tests.inference import dz_ci, jzs_bf10, sign_flip_test, tost_one_sample

N_FLIP, N_PERM = 100_000, 100_000
RSCRIPT = "Rscript"


def one_sample(x, alternative: str, key: str, tost: bool = False) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    p, _ = sign_flip_test(x, alternative, n_perm=N_FLIP, rng=C.rng(f"9|signflip|{key}"))
    lo, hi = dz_ci(x)
    out = {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "dz": float(x.mean() / x.std(ddof=1)),
           "dz_ci95": [float(lo), float(hi)], "p": float(p), "alternative": alternative, "n_positive": int((x > 0).sum()),
           "bf01_two_sided": float(1.0 / jzs_bf10(x, alternative="two-sided"))}
    if tost:
        out["tost_0.5_p"] = float(tost_one_sample(x, 0.5).p)
    return out


def fisher_ci(r: float, n: int) -> list[float]:
    z, se = np.arctanh(r), np.sqrt(1.06 / (n - 3))
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def spearman_perm(x, y, key: str) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(stats.spearmanr(x, y).correlation)
    g = C.rng(f"9|rho|{key}")
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    null = np.array([np.corrcoef(rx, g.permutation(ry))[0, 1] for _ in range(N_PERM)])
    return {"rho": r, "p_one_sided": float((1 + (null >= r).sum()) / (1 + N_PERM)), "n": int(x.size), "ci95_fisher": fisher_ci(r, x.size)}


def partial_perm(x, y, Z, key: str) -> dict:
    """Partial Spearman rho by Freedman-Lane permutation of rank residuals, one-sided."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Zr = np.column_stack([np.ones(len(rx))] + [stats.rankdata(z) for z in Z])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    g = C.rng(f"9|partial|{key}")
    null = np.array([np.corrcoef(ex, g.permutation(ey))[0, 1] for _ in range(N_PERM)])
    return {"partial_rho": r, "p_one_sided": float((1 + (null >= r).sum()) / (1 + N_PERM)), "n": int(len(rx)), "n_covariates": len(Z)}


def masc_tests():
    inc = C.inclusion()
    res = {"n_listener_sessions_usable": int(inc.usable.sum()), "two_session_listeners": C.two_session_listeners(),
           "one_session_listeners": C.one_session_listeners()}
    ck = pd.read_csv(C.TAB / "meg_masc_checks_listeners.csv", dtype={"listener": str})
    ck = ck.merge(inc[inc.usable][["listener", "session"]], on=["listener", "session"])
    missing = len(inc[inc.usable]) - len(ck)
    if missing:
        raise SystemExit(f"checks missing for {missing} usable sessions")
    chk = {f"envelope_tracking_session{s + 1}": one_sample(ck[ck.session == s].c1_r, "greater", f"C1|{s}") for s in (0, 1)}
    chk["pass"] = bool(all(chk[k]["p"] < 0.05 for k in ("envelope_tracking_session1", "envelope_tracking_session2")))
    res["checks"] = chk
    # sentences against lists
    ls = pd.read_csv(C.TAB / "meg_masc_sync_lists.csv", dtype={"listener": str})
    ls["diff"] = ls.isc_sentence - ls.isc_list
    for src in ("meg", "resid"):
        d = ls[ls.source == src].groupby("listener")["diff"].mean()
        res[f"sync_lists_{'all' if src == 'meg' else 'acoustics_removed'}"] = one_sample(d, "two-sided", f"T2b|{src}")
    # synchrony across days
    sy = pd.read_csv(C.TAB / "meg_masc_sync_listeners.csv", dtype={"listener": str}).pivot(index="listener", columns="session", values="sync")
    L1 = C.two_session_listeners()
    sy = sy.loc[L1]
    t1 = {"primary": spearman_perm(sy[0], sy[1], "T1")}
    cov = inc[inc.two_session_set].pivot(index="listener", columns="session", values=["rejected_fraction", "head_to_sensor_mm"]).loc[L1]
    disp = inc[inc.two_session_set].groupby("listener").head_displacement_mm.first().loc[L1]
    c1 = ck.pivot(index="listener", columns="session", values="c1_r").loc[L1]
    Z = [cov[("rejected_fraction", 0)], cov[("rejected_fraction", 1)], disp, c1[0], c1[1], cov[("head_to_sensor_mm", 0)], cov[("head_to_sensor_mm", 1)]]
    t1["partial"] = partial_perm(sy[0].to_numpy(), sy[1].to_numpy(), [np.asarray(z, float) for z in Z], "T1")
    t1["share_surviving"] = t1["partial"]["partial_rho"] / t1["primary"]["rho"] if t1["primary"]["rho"] != 0 else None
    t1["icc"] = icc()
    res["across_days"] = t1
    (C.TAB / "meg_masc_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


def icc() -> dict:
    """Variance components of one-minute synchrony segments over both sessions (lme4; parametric bootstrap, 1000)."""
    seg = C.TAB / "meg_masc_sync_segments.csv"
    out = C.TAB / "_p9_icc.csv"
    code = f'''suppressMessages(library(lme4)); set.seed(20261006)
d <- read.csv("{seg.as_posix()}", colClasses = c(listener = "character")); d$ls <- interaction(d$listener, d$session)
sh <- function(m) {{ v <- as.data.frame(VarCorr(m)); t <- sum(v$vcov); l <- v$vcov[v$grp == "listener"]; s <- v$vcov[v$grp == "ls"]
  c(across_days = l / t, within_day_only = s / t, within_day_total = (l + s) / t) }}
m <- lmer(isc ~ 1 + (1 | listener) + (1 | ls) + (1 | segment), data = d)
b <- bootMer(m, sh, nsim = 1000, type = "parametric")
r <- rbind(estimate = sh(m), lo = apply(b$t, 2, quantile, .025, na.rm = TRUE), hi = apply(b$t, 2, quantile, .975, na.rm = TRUE))
write.csv(r, "{out.as_posix()}")
'''
    rf = C.TAB / "_p9_icc.R"
    rf.write_text(code, encoding="utf-8")
    subprocess.run([RSCRIPT, str(rf)], check=True, capture_output=True)
    r = pd.read_csv(out, index_col=0)
    return {c: {"share": float(r.loc["estimate", c]), "ci95": [float(r.loc["lo", c]), float(r.loc["hi", c])]} for c in r.columns}
