"""Synchrony in MEG-MASC across days, and for sentences against word lists.

Components fitted within each session, leave-one-sound-file-out. The list comparison uses list segments and an equal
duration of the neighbouring sentences, for all activity and after removal of the acoustic response.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.linalg as sl

from preprocessing.cnd_recordings import band_64
from readers import meg_masc as C
from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

N_COMP, SHRINK = 3, 0.5
SEG = 60 * C.FS
RIDGE_GRID = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
RES = C.DER / "resid"


def file_data(listeners: list[str], ses: int, key: str, n: int, source: str = "meg") -> tuple[np.ndarray, np.ndarray]:
    X, V = np.zeros((len(listeners), 208, n)), np.zeros((len(listeners), n), bool)
    for i, sub in enumerate(listeners):
        if key not in C.kept_keys(sub, ses):
            continue
        folder = C.DER / "meg" if source == "meg" else RES
        with np.load(folder / f"sub-{sub}_ses-{ses}.npz") as z:
            y, v = z[f"{key}_d14"].astype(np.float64), z[f"{key}_valid"]
        T = min(n, y.shape[1])
        X[i, :, :T], V[i, :T] = y[:, :T], v[:T]
    return X, V


def spaces(listeners: list[str], ses: int, keys: list[str], ns: dict, source: str = "meg") -> dict:
    terms = {k: _cov_terms(*file_data(listeners, ses, k, ns[k], source)) for k in keys}
    W = {}
    for k in keys:
        Rw = sum(t[0] for kk, t in terms.items() if kk != k)
        Rb = sum(t[1] for kk, t in terms.items() if kk != k)
        W[k], _ = corrca_from_terms(Rw, Rb, len(listeners), SHRINK, N_COMP)
    return W


def list_masks(key: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """List-segment samples and the matched sentence samples of one sound file."""
    _, words, _ = C.stimulus_tables()
    w = words[(words.key == key) & words.condition.isin(["sentence", "word_list", "pseudo_words"])].sort_values("t").reset_index(drop=True)
    w["run"] = (w.condition.eq("word_list") != w.condition.eq("word_list").shift()).cumsum()
    lm = np.zeros(n, bool)
    segs = []
    for _, g in w[w.condition == "word_list"].groupby("run"):
        a, b = int(np.floor(g.t.min() * C.FS)), int(np.ceil((g.t + g.duration).max() * C.FS))
        a, b = max(a, 0), min(b, n)
        lm[a:b] = True
        segs.append((a, b))
    sm = np.zeros(n, bool)
    for a, b in segs:
        need = b - a
        lo = a
        while lo > 0 and need > 0 and not lm[lo - 1] and not sm[lo - 1]:
            lo -= 1
            need -= 1
        sm[lo:a] = True
        hi = b
        while hi < n and need > 0 and not lm[hi] and not sm[hi]:
            sm[hi] = True
            hi += 1
            need -= 1
    return lm, sm


def pooled_isc(Ys: list[np.ndarray], oks: list[np.ndarray]) -> np.ndarray:
    """Leave-one-listener-out ISC per component over samples pooled across sound files (each file with its own space):
    as subject_isc, but the listener's and the others' mean series are concatenated over files before correlating."""
    N, K = Ys[0].shape[:2]
    A, B = [[] for _ in range(N)], [[] for _ in range(N)]
    for Y, ok in zip(Ys, oks):
        okf = ok.astype(float)
        total = np.einsum("nkt,nt->kt", Y, okf)
        count = okf.sum(0)
        for i in range(N):
            wsum = count - okf[i]
            sel = ok[i] & (wsum >= max(1, (N - 1) // 2))
            if sel.any():
                A[i].append(Y[i][:, sel])
                B[i].append(((total - Y[i] * okf[i]) / np.maximum(wsum, 1))[:, sel])
    out = np.full((N, K), np.nan)
    for i in range(N):
        if not A[i]:
            continue
        a, b = np.concatenate(A[i], 1), np.concatenate(B[i], 1)
        a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        out[i] = np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1.0), np.nan)
    return out


def lagged(F: np.ndarray, n_lags: int, step: int = 2) -> np.ndarray:
    r, T = F.shape
    X = np.zeros((T, r * n_lags))
    for k in range(n_lags):
        d = k * step
        X[d:, k::n_lags] = F[:, : T - d].T
    return X


def residualise(sub: str, ses: int) -> None:
    """Subtract the held-out acoustic prediction (34 regressors at 1-4 Hz, lags 0-594 ms, leave-one-file-out ridge)."""
    target = RES / f"sub-{sub}_ses-{ses}.npz"
    if target.exists():
        return
    keys = C.kept_keys(sub, ses)
    with np.load(C.DER / "meg" / f"sub-{sub}_ses-{ses}.npz") as z:
        Y = {k: z[f"{k}_d14"].astype(np.float64) for k in keys}
        V = {k: z[f"{k}_valid"] for k in keys}
    Xs, Ys = {}, {}
    for k in keys:
        st = C.features(k)
        T = min(Y[k].shape[1], int(st["n"]))
        A = band_64(st["acoustic_raw"][:, :T], C.FS, 1.0, 4.0)
        A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-12)
        Xs[k], Ys[k] = lagged(A, 20), Y[k][:, :T].T
    G = {k: Xs[k][V[k][: len(Xs[k])]].T @ Xs[k][V[k][: len(Xs[k])]] for k in keys}
    Cy = {k: Xs[k][V[k][: len(Xs[k])]].T @ Ys[k][V[k][: len(Xs[k])]] for k in keys}
    GT, CT = sum(G.values()), sum(Cy.values())
    best, preds = -np.inf, None
    eig = {k: sl.eigh(GT - G[k]) for k in keys}
    for g in RIDGE_GRID:
        P, num, den = {}, 0.0, 0.0
        rs = []
        for k in keys:
            e, U = eig[k]
            beta = U @ ((U.T @ (CT - Cy[k])) / (e + g * e.mean())[:, None])
            P[k] = Xs[k] @ beta
            v = V[k][: len(Xs[k])]
            a, b = Ys[k][v], P[k][v]
            a, b = a - a.mean(0), b - b.mean(0)
            rs.append((a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-300))
        r = float(np.mean(np.nanmean(np.stack(rs), 0)))
        if r > best:
            best, preds = r, P
    arrays = {}
    with np.load(C.DER / "meg" / f"sub-{sub}_ses-{ses}.npz") as z:
        for k in z.files:
            if k.endswith("_valid"):
                arrays[k] = z[k]
    for k in keys:
        R = Y[k].copy()
        R[:, : len(preds[k])] -= preds[k].T
        arrays[f"{k}_d14"] = R.astype(np.float32)
    tmp = RES / f"sub-{sub}_ses-{ses}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, target)


def session_synchrony():
    t = C.inclusion()
    ns = {k: int(C.features(k)["n"]) for k in C.sound_keys()}
    keys = C.sound_keys()
    rows, segs, lists = [], [], []
    # the two-session listeners, both sessions
    L1 = C.two_session_listeners()
    for ses in (0, 1):
        W = spaces(L1, ses, keys, ns)
        per = []
        for k in keys:
            X, V = file_data(L1, ses, k, ns[k])
            Y = np.matmul(W[k].T, X)
            per.append(subject_isc(X, W[k], V))
            for a in range(0, ns[k] - SEG + 1, SEG):
                Vw = np.zeros_like(V)
                Vw[:, a: a + SEG] = V[:, a: a + SEG]
                if Vw.sum(1).max() < SEG // 2:
                    continue
                iw = subject_isc(X, W[k], Vw)
                for i, sub in enumerate(L1):
                    if Vw[i].sum() >= SEG // 2 and np.isfinite(iw[i]).all():
                        segs.append({"listener": sub, "session": ses, "segment": f"{k}|{a // SEG}", "isc": float(iw[i].sum())})
        stack = np.stack(per)
        m = np.nanmean(stack, 0)
        for i, sub in enumerate(L1):
            rows.append({"listener": sub, "session": ses, "sync": float(m[i].sum()), **{f"sync_c{c + 1}": float(m[i, c]) for c in range(N_COMP)},
                         "n_files": int(np.isfinite(stack[:, i, 0]).sum())})
    pd.DataFrame(rows).to_csv(C.TAB / "meg_masc_sync_listeners.csv", index=False)
    pd.DataFrame(segs).to_csv(C.TAB / "meg_masc_sync_segments.csv", index=False)
    # all usable listeners of each session
    for ses in (0, 1):
        Ls = sorted(t[(t.session == ses) & t.usable].listener)
        for sub in Ls:
            residualise(sub, ses)
        masks = {k: list_masks(k, ns[k]) for k in keys}
        for source in ("meg", "resid"):
            W = spaces(Ls, ses, keys, ns, source)
            YL, OL, YS, OS = [], [], [], []
            for k in keys:
                X, V = file_data(Ls, ses, k, ns[k], source)
                Y = np.matmul(W[k].T, X)
                lm, sm = masks[k]
                YL.append(Y), OL.append(V & lm[None]), YS.append(Y), OS.append(V & sm[None])
            il, is_ = pooled_isc(YL, OL), pooled_isc(YS, OS)
            for i, sub in enumerate(Ls):
                lists.append({"listener": sub, "session": ses, "source": source, "isc_list": float(il[i].sum()), "isc_sentence": float(is_[i].sum()),
                              "n_list_samples": int(sum(o[i].sum() for o in OL)), "n_sentence_samples": int(sum(o[i].sum() for o in OS))})
    pd.DataFrame(lists).to_csv(C.TAB / "meg_masc_sync_lists.csv", index=False)
