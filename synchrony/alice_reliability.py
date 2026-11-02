"""Alice: per-segment synchrony and encoding gain, and their split-half reliability."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from common.settings import load_analysis, load_paths
from encoding_models import song_regressors as g
from synchrony import alice_comprehension as h
from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

def per_segment_delta_r(sub: str, data: dict, stims: dict, seed: int) -> dict:
    segs = [s for s in h.SEGMENTS if s in data and data[s][1].any()]
    if len(segs) < h.MIN_SEGMENTS:
        return {}
    a = int(h.START_S * 64)
    rng = np.random.default_rng([seed, int(sub[1:]), 17])
    blocks, masks = [], []
    for s in segs:
        st = stims[s]
        Y, v = data[s]
        T = min(Y.shape[1], st["acoustic"].shape[1] - a)
        X = {k: g.lagged(st[k][:, a:a + T], T) for k in ("acoustic", "word", "surprisal")}
        m = np.zeros(T, bool)
        for i in st["word_idx"]:
            lo = i - a
            if lo + g.WORD_WIN > 0 and lo < T:
                m[max(lo, 0): min(lo + g.WORD_WIN, T)] = True
        blocks.append({"X": X, "Y": Y[:, :T], "v": v[:T], "st": st, "a": a, "T": T})
        masks.append(m & v[:T])
    _, lam = g.fit_predict(blocks, ["acoustic", "word"], None)
    full = g.fit_predict(blocks, ["acoustic", "word", "surprisal"], lam)[0]
    shuf = []
    for _ in range(g.N_SHUF):
        for bl in blocks:
            st = bl["st"]
            raw = np.zeros_like(st["raw_surprisal"])
            raw[st["word_idx"]] = rng.permutation(st["raw_surprisal"][st["word_idx"]])
            bl["X"]["surprisal_shuf"] = g.lagged(g.bz(raw[None])[:, bl["a"]:bl["a"] + bl["T"]], bl["T"])
        shuf.append(g.fit_predict(blocks, ["acoustic", "word", "surprisal_shuf"], lam)[0])
    out = {"listener": sub}
    for j, s in enumerate(segs):
        one = [blocks[j]]
        rf = g.r_mean(one, [full[j]], [masks[j]])
        rs = np.mean([g.r_mean(one, [sh[j]], [masks[j]]) for sh in shuf])
        out[f"seg{s}"] = rf - rs
    return out


def per_segment_isc(data: dict) -> pd.DataFrame:
    subs = sorted(data)
    per_seg = {}
    for seg in h.SEGMENTS:
        have = [s for s in subs if seg in data[s]]
        n = min(data[s][seg][0].shape[1] for s in have)
        per_seg[seg] = (have, np.stack([data[s][seg][0][:, :n] for s in have]), np.stack([data[s][seg][1][:n] for s in have]))
    terms = {seg: _cov_terms(x, v) for seg, (_, x, v) in per_seg.items()}
    rows = []
    for seg, (have, x, v) in per_seg.items():
        W = corrca_from_terms(sum(t[0] for q, t in terms.items() if q != seg), sum(t[1] for q, t in terms.items() if q != seg),
                              len(subs), 0.5, 3)[0]
        isc = subject_isc(x, W, v)
        for s, row in zip(have, isc):
            rows.append({"listener": s, "segment": seg, "isc": float(np.nansum(row)), **{f"c{k + 1}": float(row[k]) for k in range(3)}})
    return pd.DataFrame(rows)


def split_half(wide: pd.DataFrame) -> dict:
    odd = wide[[c for c in wide.columns if int(str(c).replace("seg", "")) % 2 == 1]].mean(1)
    even = wide[[c for c in wide.columns if int(str(c).replace("seg", "")) % 2 == 0]].mean(1)
    r = stats.spearmanr(odd, even, nan_policy="omit").correlation
    return {"r_half": float(r), "spearman_brown": float(2 * r / (1 + r)) if r > -1 else np.nan}


def split_half_reliability():
    paths, cfg = load_paths(), load_analysis()
    out = paths.results_root
    cache = paths.derived_root / "alice"
    words = pd.read_csv(paths.derived_root / "alice" / "words_gpt2.csv")
    stims = {s: h.seg_stim(s, words, "gpt2_surprisal_bits") for s in h.SEGMENTS}
    data8, _ = h.load_all(cache, "1-8")
    rows = Parallel(n_jobs=6)(delayed(per_segment_delta_r)(sub, data8[sub], stims, cfg.project.seed) for sub in sorted(data8))
    dr = pd.DataFrame([r for r in rows if r]).set_index("listener")
    data4, _ = h.load_all(cache, "1-4")
    isc = per_segment_isc(data4)
    isc.to_csv(out / "alice_isc_per_segment.csv", index=False)
    dr.to_csv(out / "x17_delta_r_per_segment.csv")
    isc_w = isc.pivot(index="listener", columns="segment", values="isc")
    isc_w.columns = [f"seg{c}" for c in isc_w.columns]
    lis = pd.read_csv(paths.results_root / "alice_listeners.csv", index_col=0)
    d = lis.dropna(subset=["comprehension"])
    gb = np.random.default_rng(cfg.project.seed)
    boots = [stats.spearmanr(*d.iloc[ix][["comprehension", "isc"]].to_numpy().T).correlation
             for ix in (gb.integers(0, len(d), len(d)) for _ in range(10000))]
    comp = isc.groupby("listener")[["c1", "c2", "c3"]].mean().reindex(d.index)
    hi = d[d.comprehension >= 0.5]
    res = {"reliability_isc": split_half(isc_w), "reliability_delta_r": split_half(dr),
           "comprehension_isc": {"spearman": float(stats.spearmanr(d.comprehension, d.isc).correlation),
                                 "pearson": float(stats.pearsonr(d.comprehension, d.isc)[0]),
                                 "spearman_ci95_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                 "spearman_scores_ge_50pct": {"n": int(len(hi)), "rho": float(stats.spearmanr(hi.comprehension, hi.isc).correlation)},
                                 "per_component": {c: float(stats.spearmanr(d.comprehension, comp[c]).correlation) for c in comp}},
           "n": int(len(d))}
    (out / "x17_alice_reliability.json").write_text(json.dumps(res, indent=2, default=float))
