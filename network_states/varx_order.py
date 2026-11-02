"""Selection of the recurrent order of the VARX model by BIC, on random groups of trials."""
from __future__ import annotations

import numpy as np
import pandas as pd

from network_states.varx_model import _fit, myxcorr

def record_stats(y: np.ndarray, x: np.ndarray, na_max: int, nb: int, maxlag: int):
    pred = np.hstack([y[:-1], x[1:]])
    lags = np.r_[np.full(y.shape[1], na_max), np.full(x.shape[1], nb)]
    return myxcorr(pred, y[1:], lags, maxlag)


def _columns(ydim: int, na_max: int, na: int, n_input: int) -> np.ndarray:
    rec = [i * na_max + k for i in range(ydim) for k in range(na)]
    return np.r_[rec, ydim * na_max + np.arange(n_input)].astype(int)


def select_order(Y: list[np.ndarray], X: list[np.ndarray], na_grid: list[int], nb: int, lam: float,
                 maxlag: int) -> pd.DataFrame:
    na_max = max(na_grid)
    ydim = Y[0].shape[1]
    stats = [record_stats(y, x, na_max, nb, maxlag) for y, x in zip(Y, X)]
    Rxx = sum(s[0] for s in stats)
    Rxy = sum(s[1] for s in stats)
    ryy = sum(s[2] for s in stats)
    T = sum(s[3] for s in stats)
    rows = []
    for na in na_grid:
        cols = _columns(ydim, na_max, na, X[0].shape[1] * nb)
        n_par = cols.size
        gamma = lam / np.sqrt(T - n_par) if lam else 0.0
        _, s2, _ = _fit(Rxx[np.ix_(cols, cols)], Rxy[cols], ryy, gamma)
        bic = float(np.sum(np.log(T) * n_par + T * np.log(s2 / T)))
        aic = float(np.sum(2 * n_par + T * np.log(s2 / T)))
        cv = []
        for k, (rk, ck, yk, tk) in enumerate(stats):
            tr_T = T - tk
            g = lam / np.sqrt(tr_T - n_par) if lam else 0.0
            h, _, _ = _fit((Rxx - rk)[np.ix_(cols, cols)], (Rxy - ck)[cols], ryy - yk, g)
            r = rk[np.ix_(cols, cols)]
            sse = (h * (r @ h)).sum(axis=0) - 2 * (h * ck[cols]).sum(axis=0) + yk
            cv.append(float(np.mean(sse / yk)))
        rows.append({"na": na, "bic": bic, "aic": aic, "cv_nmse": float(np.mean(cv)),
                     "cv_nmse_sem": float(np.std(cv, ddof=1) / np.sqrt(len(cv)))})
    return pd.DataFrame(rows)
