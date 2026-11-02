"""Word-level gain per listener in the MEG attention dataset.

Ridge chosen on the base model (both streams' acoustic, word-onset and control regressors); for each stream, the true
surprisal regressor and permuted versions (values shuffled over the stream's words within each chapter), with the other
stream's true surprisal in the model.
"""
from __future__ import annotations

import os

import numpy as np

from encoding_models import gpu_encoding as WG, sufficient_statistics as PE
from pathlib import Path
from readers import meg_attention as C

OUT = C.DER / "A"
VARIANTS = {"primary": ("surprisal", None, 500)}
BATCH = 16


def regressor(tr: dict, stream: str, valkey: str, keepkey: str | None, values=None) -> np.ndarray:
    st = tr["st"]
    v = st[f"{stream}_{valkey}"] if values is None else values
    keep = st[f"{stream}_{keepkey}"] if keepkey else None
    return C.bz(C.surprisal_impulse(st, stream, v, keep, tr["T"]))


def permuted_versions(trials, stream, valkey, keepkey, n, key):
    g = C.rng(key)
    for _ in range(n):
        ver = []
        for tr in trials:
            vals = tr["st"][f"{stream}_{valkey}"]
            if keepkey:
                keep = tr["st"][f"{stream}_{keepkey}"]
                shuffled = np.zeros_like(vals)
                shuffled[keep] = g.permutation(vals[keep])
            else:
                shuffled = g.permutation(vals)
            ver.append(regressor(tr, stream, valkey, keepkey, shuffled))
        yield ver


def listener_gain(p: int) -> Path:
    target = OUT / f"P{p:02d}.npz"
    if target.exists():
        return target
    if not C.included(p):
        np.savez(target, excluded=np.array(True))
        return target
    trials = C.listener_trials(p)
    base = C.add_base(trials)
    files = np.array([tr["file"] for tr in trials])
    subsets = [np.arange(len(trials)), np.flatnonzero(files == "LP"), np.flatnonzero(files == "HP")]
    ff = WG.GpuFolds(trials, base, C.N_LAGS)
    out = {"lam": np.array(ff.lam), "chapters": np.array([f"{tr['file']}{tr['chapter']}" for tr in trials])}
    for variant, (valkey, keepkey, n_perm) in VARIANTS.items():
        for stream in C.STREAMS:
            other = "ignored" if stream == "attended" else "attended"
            F_other = [regressor(tr, other, valkey, keepkey) for tr in trials]
            ms = PE.MaskStats(ff, C.word_masks(trials, stream, keepkey))
            st = PE.SuffTest(ff, F_other, ms, C.N_LAGS)
            S_true = st.stats([[regressor(tr, stream, valkey, keepkey) for tr in trials]])
            S_perm, batch = [], []
            for ver in permuted_versions(trials, stream, valkey, keepkey, n_perm, f"p7|perm|{p}|{stream}|{variant}"):
                batch.append(ver)
                if len(batch) == BATCH:
                    S_perm.append(st.stats(batch))
                    batch = []
            if batch:
                S_perm.append(st.stats(batch))
            S_perm = np.concatenate(S_perm, 0)
            out[f"{variant}_{stream}_rtrue"] = np.stack([PE.r_subset(S_true[0], s) for s in subsets])
            out[f"{variant}_{stream}_rperm"] = np.stack([PE.r_subset(S_perm, s) for s in subsets], 1)
            out[f"{variant}_{stream}_nmask"] = np.array([float(S_true[0][s, 0, 0].sum()) for s in subsets])
    tmp = OUT / f"P{p:02d}.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, target)
    return target
