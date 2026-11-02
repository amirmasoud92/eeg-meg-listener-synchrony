"""Inter-subject correlation by correlated components analysis (Dmochowski et al., 2012; Parra et al., 2019).

Projections W maximise the ratio of between-subject to within-subject covariance of stimulus-locked multichannel data.
The ISC of a listener for a stimulus is the correlation of that listener's component time course with the mean of the
other listeners', summed over the leading components. W is fitted without the scored stimulus.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sl

def _cov_terms(data: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Within (Rw) and between (Rb) subject covariance for one stimulus.

    data: subjects x channels x time; valid: subjects x time (NaN-free samples usable)."""
    N, D, T = data.shape
    X = data - np.nanmean(data, axis=2, keepdims=True)
    if valid is not None:
        X = np.where(valid[:, None, :], X, 0.0)
    X = np.nan_to_num(X)
    Rw = np.einsum("ndt,net->de", X, X)
    S = X.sum(0)
    Rt = S @ S.T
    return Rw, Rt - Rw


def corrca_from_terms(Rw: np.ndarray, Rb: np.ndarray, N: int, shrinkage: float = 0.5,
                      n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """CorrCA from summed within/between covariance (shrinkage of Rw towards its mean eigenvalue)."""
    D = Rw.shape[0]
    Rw_s = (1 - shrinkage) * Rw + shrinkage * np.trace(Rw) / D * np.eye(D)
    evals, W = sl.eigh(Rb, Rw_s)
    order = np.argsort(evals)[::-1][:n_components]
    W = W[:, order]
    isc = np.diag(W.T @ Rb @ W) / np.diag(W.T @ Rw @ W) / (N - 1)
    return W, isc


def subject_isc(data: np.ndarray, W: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Leave-one-subject-out ISC (subjects x components) of one stimulus.

    data: subjects x channels x time. Each subject's component is correlated with the mean
    of the other subjects' components over samples valid for that subject and for all others'
    mean (others' invalid samples are excluded from their mean)."""
    X = data if not np.isnan(data).any() else np.nan_to_num(data)
    Y = np.matmul(W.T, X)
    N = Y.shape[0]
    ok = np.ones((N, Y.shape[2]), bool) if valid is None else valid
    okf = ok.astype(float)
    total = np.einsum("nkt,nt->kt", Y, okf)
    count = okf.sum(0)
    out = np.full((N, W.shape[1]), np.nan)
    for n in range(N):
        wsum = count - okf[n]
        mean_o = (total - Y[n] * okf[n]) / np.maximum(wsum, 1)
        sel = ok[n] & (wsum >= max(1, (N - 1) // 2))
        if not sel.any():
            continue
        a, b = Y[n][:, sel], mean_o[:, sel]
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        out[n] = np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1.0), np.nan)
    return out


def forward_model(Rw: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Scalp projections of the components (Parra et al., 2019; Haufe et al., 2014): Rw W (W' Rw W)^-1."""
    return Rw @ W @ np.linalg.inv(W.T @ Rw @ W)


def pair_cov(Ya: np.ndarray, oka: np.ndarray | None = None, Yb: np.ndarray | None = None,
             okb: np.ndarray | None = None, min_samples: int = 128) -> np.ndarray:
    """Covariance of component time courses for every ordered subject pair (subjects x subjects x
    components): entry [i, j] is cov(subject i in a, subject j in b) over the samples valid in both.
    The diagonal and pairs with fewer than ``min_samples`` joint samples are NaN."""
    Yb_, okb_ = (Ya, oka) if Yb is None else (Yb, okb)
    N, K, T = Ya.shape
    oka = np.ones((N, T), bool) if oka is None else oka
    okb_ = np.ones((N, T), bool) if okb_ is None else okb_
    out = np.full((N, N, K), np.nan)
    for i in range(N):
        for j in range(N):
            m = oka[i] & okb_[j]
            if i == j or m.sum() < min_samples:
                continue
            a, b = Ya[i][:, m], Yb_[j][:, m]
            out[i, j] = ((a - a.mean(1, keepdims=True)) * (b - b.mean(1, keepdims=True))).mean(1)
    return out


def subject_shared_cov(P: np.ndarray) -> np.ndarray:
    """Per-subject share of a pair_cov matrix: mean over j != i of (P[i, j] + P[j, i]) / 2 (subjects x components)."""
    S = (P + np.transpose(P, (1, 0, 2))) / 2
    return np.nanmean(S, axis=1)
