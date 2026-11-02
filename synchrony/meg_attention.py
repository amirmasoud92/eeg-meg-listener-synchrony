"""Synchrony in the MEG attention dataset.

Three correlated components on the 1-4 Hz data with leave-one-chapter-out spaces; per listener, leave-one-listener-out
ISC per component, averaged over chapters and summed over components.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from readers import meg_attention as C
from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

N_COMP, SHRINK = 3, 0.5


def chapter_data(listeners: list[int], f: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(C.features(f, k)["n"])
    X, V = [], []
    for p in listeners:
        with np.load(C.DER / "meg" / f"P{p:02d}.npz") as z:
            y = z[f"{f}_{k}_d14"].astype(np.float64)
            v = z[f"{f}_{k}_valid"]
        T = min(n, y.shape[1])
        yy = np.zeros((y.shape[0], n))
        vv = np.zeros(n, bool)
        yy[:, :T], vv[:T] = y[:, :T], v[:T]
        X.append(yy)
        V.append(vv)
    return np.stack(X), np.stack(V)


def listener_synchrony():
    listeners = [p for p in range(1, 59) if C.included(p)]
    N = len(listeners)
    terms = {}
    for f, k in C.CHAPTERS:
        X, V = chapter_data(listeners, f, k)
        terms[(f, k)] = _cov_terms(X, V)
    isc = {}
    for f, k in C.CHAPTERS:
        Rw = sum(t[0] for key, t in terms.items() if key != (f, k))
        Rb = sum(t[1] for key, t in terms.items() if key != (f, k))
        W, _ = corrca_from_terms(Rw, Rb, N, SHRINK, N_COMP)
        X, V = chapter_data(listeners, f, k)
        isc[(f, k)] = subject_isc(X, W, V)
    stack = np.stack([isc[key] for key in C.CHAPTERS])                         # chapters x listeners x components
    per = np.nanmean(stack, axis=0)
    n_ch = np.isfinite(stack[:, :, 0]).sum(0)
    rows = [{"listener": p, "sync": float(per[i].sum()), **{f"sync_c{c + 1}": float(per[i, c]) for c in range(N_COMP)},
             "n_chapters_sync": int(n_ch[i])} for i, p in enumerate(listeners)]
    pd.DataFrame(rows).to_csv(C.TAB / "meg_attention_sync_listeners.csv", index=False)
    Rw = sum(t[0] for t in terms.values())
    Rb = sum(t[1] for t in terms.values())
    W, isc_all = corrca_from_terms(Rw, Rb, N, SHRINK, N_COMP)
    A = Rw @ W @ np.linalg.inv(W.T @ Rw @ W)
    share = [float(W[:, c] @ Rb @ W[:, c] / (W[:, c] @ W[:, c] * np.trace(Rb))) for c in range(N_COMP)]
    (C.TAB / "meg_attention_sync_components.json").write_text(json.dumps(
        {"n_listeners": N, "isc_all_chapters": [float(x) for x in isc_all], "share_between_cov": share,
         "shrinkage": SHRINK}, indent=2), encoding="utf-8")
    pd.DataFrame(A, columns=[f"A_c{c + 1}" for c in range(N_COMP)]).assign(channel=np.arange(A.shape[0])).to_csv(
        C.TAB / "meg_attention_sync_maps.csv", index=False)
