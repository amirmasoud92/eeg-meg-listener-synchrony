"""Engagement ratings and posterior alpha power against synchrony.

Within-listener association of trial-wise engagement with ISC in the speech-defined space (double-centred), relative
posterior alpha against ISC and comprehension across listeners, and segment-wise alpha against ISC within listeners.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
from joblib import Parallel, delayed
from scipy import stats

from statistical_tests.inference import summarise
from synchrony import alice_comprehension as h, speech_in_noise as d01

RSCRIPT = "Rscript"
pick = lambda s: {k: s[k] for k in ("n", "mean", "dz", "dz_ci95", "p_signflip", "n_positive")}


def within_corr(A: np.ndarray, B: np.ndarray, min_n: int = 4) -> np.ndarray:
    Ac, Bc = d01.double_centre(A), d01.double_centre(B)
    z = []
    for i in range(A.shape[0]):
        m = ~np.isnan(Ac[i]) & ~np.isnan(Bc[i])
        if m.sum() >= min_n and Ac[i][m].std() > 0 and Bc[i][m].std() > 0:
            z.append(np.arctanh(np.clip(np.corrcoef(Ac[i][m], Bc[i][m])[0, 1], -0.999, 0.999)))
    return np.array(z)


def engagement_within_listeners(paths, out: Path) -> dict:
    isc = pd.read_csv(paths.results_root / "aalborg_isc_speech_space.csv")
    tt = pd.read_csv(paths.processed_root / "trial_table.csv")[["subject", "label", "engagement"]].rename(columns={"label": "stimulus"})
    d = isc.merge(tt, on=["subject", "stimulus"])
    res = {"n_trials": int(len(d))}
    for name, conds in (("all", None), ("speech", ["SD", "SF"]), ("song", ["MD", "MF"]), ("instrumental", ["MC"])):
        dd = d if conds is None else d[d.condition.isin(conds)]
        A = dd.pivot(index="subject", columns="stimulus", values="isc").to_numpy()
        B = dd.pivot(index="subject", columns="stimulus", values="engagement").to_numpy()
        z = within_corr(A, B, 4 if name != "instrumental" else 3)
        res[f"{name}"] = pick(summarise(z, "greater"))
    c = d.engagement - d.groupby("subject").engagement.transform("mean")
    d["eng_c"] = c / c.std()
    path = out / "engagement_aalborg_trials.csv"
    d.to_csv(path, index=False)
    code = f"""
suppressMessages({{library(lmerTest); library(jsonlite)}})
d <- read.csv("{path.as_posix()}")
m <- lmer(isc ~ condition + eng_c + (1 | subject) + (1 | stimulus), data = d)
s <- as.data.frame(summary(m)$coefficients); s$term <- rownames(s)
cat(toJSON(s, digits = 6))
"""
    script = path.with_suffix(".R")
    script.write_text(code, encoding="utf-8")
    r = subprocess.run([RSCRIPT, str(script)], capture_output=True, text=True)
    res["mixed_model"] = [row for row in json.loads(r.stdout) if row["term"] == "eng_c"] if r.returncode == 0 else r.stderr[-1500:]
    return res


def posterior_channels(labels: list[str], axis: dict, n: int = 12) -> list[str]:
    rows = [ln.split() for ln in (h.ROOT / "easycapM10-acti61_elec.sfp").read_text().splitlines() if ln.strip()]
    pos = {r[0]: np.array([float(v) for v in r[1:4]]) for r in rows}
    ant = axis["anterior_axis"]
    front = np.array([axis["anterior_sign"] * pos[c][ant] for c in labels])
    return [labels[i] for i in np.argsort(front)[:n]]


def alpha_one(sub: str, axis: dict) -> dict:
    X, labels, _, onsets = h.load_raw(sub)
    X = X - X.mean(0, keepdims=True)
    post = [labels.index(c) for c in posterior_channels(labels, axis)]
    dur = h.seg_durations()
    out = {"listener": sub}
    for s in h.SEGMENTS:
        if s not in onsets:
            continue
        a = onsets[s] + int(h.START_S * h.FS)
        e = onsets[s] + int((dur[s] - h.END_TRIM_S) * h.FS)
        if e > X.shape[1]:
            continue
        seg = X[post, a:e]
        n_win = seg.shape[1] // 256
        wins = seg[:, : n_win * 256].reshape(len(post), n_win, 256)
        f, P = ss.welch(wins, fs=h.FS, nperseg=256, axis=-1)
        alpha = P[..., (f >= 8) & (f <= 12)].sum(-1)
        total = P[..., (f >= 1) & (f <= 30)].sum(-1)
        rel = np.log(np.median(alpha / total, axis=1))           # median over windows, per channel
        out[f"seg{s}"] = float(rel.mean())
    return out


def spearman_perm(x, y, alt="less", n=100000, seed=0):
    r = stats.spearmanr(x, y).correlation
    g_ = np.random.default_rng(seed)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    null = np.array([np.corrcoef(rx, g_.permutation(ry))[0, 1] for _ in range(n)])
    p = (1 + (null <= r).sum()) / (1 + n) if alt == "less" else (1 + (null >= r).sum()) / (1 + n)
    return float(r), float(p)


def alpha_across_and_within_listeners(paths, out: Path, seed: int) -> dict:
    ti = paths.results_root
    axis = h.anterior_axis(["S01", "S10", "S20", "S35"])
    lis = pd.read_csv(ti / "alice_listeners.csv", index_col=0)
    rows = Parallel(n_jobs=6)(delayed(alpha_one)(s, axis) for s in lis.index)
    al = pd.DataFrame(rows).set_index("listener")
    al.to_csv(out / "engagement_alice_alpha_per_segment.csv")
    lis["alpha"] = al.mean(1)
    res = {"posterior_channels": posterior_channels([str(c) for c in range(1, 62) if c != 29], axis)}
    res["rho_alpha_isc"], res["p_alpha_isc_one_sided"] = spearman_perm(lis.alpha, lis.isc, "less", seed=seed)
    d = lis.dropna(subset=["comprehension"])
    res["rho_alpha_comprehension"], res["p_alpha_comprehension_one_sided"] = spearman_perm(d.alpha, d.comprehension, "less", seed=seed)
    rk = {c: stats.rankdata(d[c]) for c in ("comprehension", "isc", "alpha")}
    resid = lambda y: y - np.polyval(np.polyfit(rk["alpha"], y, 1), rk["alpha"])
    res["partial_comprehension_isc_given_alpha"] = float(np.corrcoef(resid(rk["comprehension"]), resid(rk["isc"]))[0, 1])
    res["reference_rho_comprehension_isc"] = float(stats.spearmanr(d.comprehension, d.isc).correlation)
    seg_isc = pd.read_csv(paths.results_root / "alice_isc_per_segment.csv")
    I = seg_isc.pivot(index="listener", columns="segment", values="isc")
    I.columns = [f"seg{c}" for c in I.columns]
    common = I.index.intersection(al.index)
    cols = [c for c in I.columns if c in al.columns]
    z = within_corr(I.loc[common, cols].to_numpy(), al.loc[common, cols].to_numpy(), 6)
    res["within_alpha_isc_negative"] = pick(summarise(-z, "greater"))
    res["mean_within_r"] = float(np.tanh(z.mean()))
    return res
