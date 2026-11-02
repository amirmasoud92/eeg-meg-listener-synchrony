"""Removal of the acoustic response from the EEG before synchrony is computed.

A leave-one-out ridge model (lags 0-594 ms in 31-ms steps; the grid includes no removal) predicts the EEG from the
stimulus features of each level, the held-out prediction is subtracted, and synchrony is recomputed on the residuals.
The mismatched-stimulus model is the control.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from preprocessing import cnd_recordings as cnd
from synchrony import intact_reversed_speech as b

LAG_STEP, N_LAGS = 2, 20                 # 0-594 ms at 64 Hz in 31 ms steps
LEVELS = {"acoustic": 34, "phonetic": 53, "word": 54, "mismatched": 53}   # nested: first k features;
RIDGE_GRID = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, np.inf]   # x mean eigenvalue; inf = no removal


def stim_features(cond: str) -> tuple[list[np.ndarray], dict]:
    """Per run: features x samples at 64 Hz (1-4 Hz), ordered envelope, envelope onset, spectrogram (16),
    spectrogram onsets (16), phonetic (19), word onsets."""
    import scipy.io as sio
    load = lambda c: sio.loadmat(b.cond_dir(c) / "dataCND" / "dataStim.mat", squeeze_me=True, struct_as_record=False)["stim"]
    fwd, own = load("forward"), load(cond)
    names_f = list(fwd.names)
    Df, Do = np.atleast_2d(fwd.data), np.atleast_2d(own.data)
    names_o = list(own.names)
    runs, audit = [], {"word_onsets_equal_reversed_forward": []}
    for r in range(Df.shape[1]):
        env = np.asarray(Do[names_o.index("Speech Envelope Vectors"), r], float)
        spec = np.asarray(Df[names_f.index("Spectrogram"), r], float)
        phon = np.asarray(Df[names_f.index("Phonetic Features"), r], float)
        word = np.asarray(Do[names_o.index("Word Onset Vectors"), r], float)
        if cond == "reversed":
            spec, phon = spec[::-1], phon[::-1]
            wf = np.asarray(Df[names_f.index("Word Onset Vectors"), r], float)
            audit["word_onsets_equal_reversed_forward"].append(bool(np.array_equal(wf[::-1][: word.size] > 0, word[: wf.size] > 0)))
        n = min(env.size, spec.shape[0], phon.shape[0], word.size)
        env, spec, phon, word = env[:n], spec[:n], phon[:n], word[:n]
        on = lambda x: np.maximum(np.diff(x, axis=0, prepend=x[:1]), 0)
        F = np.column_stack([env, on(env), spec, on(spec), phon, word]).T
        runs.append(cnd.band_64(F, 128.0, *b.BAND))
    return runs, audit


def lagged(F: np.ndarray, T: int, step: int = 1) -> np.ndarray:
    """samples x (features * lags); column f * N_LAGS + k is feature f delayed by k * step samples."""
    nf = F.shape[0]
    X = np.zeros((T, nf * N_LAGS), dtype=np.float32)
    for k in range(N_LAGS):
        d = k * step
        if d < T:
            X[d:, k::N_LAGS] = F[:, : T - d].T
    return X


def residuals_one(i: int, arr: dict, feats: list[np.ndarray]) -> dict:
    """Leave-one-run-out residual EEG of listener i for every level. The ridge parameter is chosen per listener and
    level from RIDGE_GRID by the total held-out explained variance over the same folds; the grid includes no
    removal, so the removal cannot add variance in cross-validation."""
    a64 = int(b.START_S * 64)
    runs = sorted(arr)
    Xs, Gs, Cs, Ys, Vs = {}, {}, {}, {}, {}
    for j, r in enumerate(runs):
        Y = arr[r][0][i]
        v = arr[r][1][i]
        T = Y.shape[1]
        F = feats[r][:, a64:a64 + T]
        other = feats[runs[(j + 1) % len(runs)]][34:53, a64:a64 + T]
        if other.shape[1] < T:
            other = np.pad(other, ((0, 0), (0, T - other.shape[1])))
        F = np.vstack([F, other])                      # rows 54-72: mismatched phonetic features
        F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)
        X = lagged(F, T, LAG_STEP)
        Xs[r], Ys[r], Vs[r] = X, Y, v
        Xv = X[v].astype(np.float64)
        Gs[r], Cs[r] = Xv.T @ Xv, Xv.T @ Y[:, v].T
    G, C = sum(Gs.values()), sum(Cs.values())
    cols = {"acoustic": np.arange(34 * N_LAGS), "phonetic": np.arange(53 * N_LAGS),
            "word": np.arange(54 * N_LAGS),
            "mismatched": np.r_[np.arange(34 * N_LAGS), np.arange(54 * N_LAGS, 73 * N_LAGS)]}
    out, info = {}, {}
    for lv, idx in cols.items():
        preds = {lam: {} for lam in RIDGE_GRID}
        for r in runs:
            Gt = (G - Gs[r])[np.ix_(idx, idx)]
            e, V = np.linalg.eigh(Gt)
            VC = V.T @ (C - Cs[r])[idx]
            Xr = Xs[r][:, idx].astype(np.float64) @ V
            m = e.mean()
            for lam in RIDGE_GRID:
                preds[lam][r] = np.zeros_like(Ys[r]) if np.isinf(lam) else (Xr @ (VC / (e + lam * m)[:, None])).T
        ev = {lam: sum(float(((Ys[r][:, Vs[r]] - preds[lam][r][:, Vs[r]]) ** 2).sum()) for r in runs) for lam in RIDGE_GRID}
        best = min(ev, key=ev.get)
        tot = sum(float((Ys[r][:, Vs[r]] ** 2).sum()) for r in runs)
        out[lv] = {r: (Ys[r] - preds[best][r]).astype(np.float32) for r in runs}
        info[lv] = {"lambda_x_mean_eig": float(best), "frac_removed": 1 - ev[best] / tot}
    return {"resid": out, "info": info}


def ensure_residuals(arr: dict, feats: list[np.ndarray], n_jobs: int, cache: Path) -> dict:
    """Compute (once) and cache each listener's residuals, one file per listener and level; return the variance
    removed per level (mean over listeners) and the chosen ridges."""
    n = arr[sorted(arr)[0]][0].shape[0]
    done = lambda i: all((cache / f"listener{i:02d}_{lv}.npz").exists() for lv in LEVELS) and (cache / f"listener{i:02d}_info.json").exists()

    def one(i):
        r = residuals_one(i, arr, feats)
        for lv in LEVELS:
            np.savez(cache / f"listener{i:02d}_{lv}.npz", **{f"r{k}": v for k, v in r["resid"][lv].items()})
        (cache / f"listener{i:02d}_info.json").write_text(json.dumps(r["info"]))
        return i
    Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(n) if not done(i))
    infos = [json.loads((cache / f"listener{i:02d}_info.json").read_text()) for i in range(n)]
    return {lv: {"mean_frac_removed": float(np.mean([x[lv]["frac_removed"] for x in infos])),
                 "lambdas": [x[lv]["lambda_x_mean_eig"] for x in infos]} for lv in LEVELS}


def level_arrays(arr: dict, cache: Path, lv: str) -> dict:
    n = arr[sorted(arr)[0]][0].shape[0]
    zs = [np.load(cache / f"listener{i:02d}_{lv}.npz") for i in range(n)]
    return {r: (np.stack([z[f"r{r}"] for z in zs]), arr[r][1]) for r in sorted(arr)}


def synchrony_on_residuals(F: dict, R: dict, nF: int, nR: int, seed: int, n_jobs: int) -> pd.DataFrame:
    """b01.test_a on given arrays (pooled and forward-only filters)."""
    runs = sorted(set(F) & set(R))
    size = min(nF, nR)
    tF = {r: b.pair_terms(*F[r]) for r in runs}
    tR = {r: b.pair_terms(*R[r]) for r in runs}
    g = np.random.default_rng(seed)
    subsets = [np.sort(g.choice(nF, size, replace=False)) for _ in range(b.N_SUBSETS if nF > size else 1)]
    idxR = np.arange(nR)[:size]
    fits = {"pooled": ("F", "R"), "forward_only": ("F",)}
    totF = (sum(tF[q][0] for q in runs), sum(tF[q][1] for q in runs))
    totR = (sum(tR[q][0] for q in runs), sum(tR[q][1] for q in runs))
    jobs = [(fit, parts, r, (totF[0] - tF[r][0], totF[1] - tF[r][1]),
             b.subset_terms(totR[0] - tR[r][0], totR[1] - tR[r][1], idxR), F[r], R[r]) for fit, parts in fits.items() for r in runs]
    rows = Parallel(n_jobs=min(n_jobs, 3))(delayed(b._run_one)(*j, subsets, idxR, size, nF) for j in jobs)
    return pd.DataFrame(rows).drop(columns=["subject_forward", "subject_reversed"])
