"""Vector autoregressive model with exogenous input, a port of the MATLAB toolbox of Parra et al. (2025; github.com/lcparra/varx)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import chi2

def _valid_mask(x: np.ndarray, y: np.ndarray, Q: int) -> np.ndarray:
    """Samples with a finite target and Q finite samples of predictor history (covariance method)."""
    finite_x = np.isfinite(x).all(axis=1)
    run = np.convolve(finite_x.astype(int), np.ones(Q, dtype=int), mode="full")[: x.shape[0]]
    ok = run == Q
    ok[: Q - 1] = False
    return ok & np.isfinite(y).all(axis=1)


def myxcorr(x: np.ndarray, y: np.ndarray, lags: np.ndarray, maxlag: int | None = None):
    lags = np.asarray(lags, dtype=int)
    Q = int(max(lags.max(), maxlag or 0))
    valid = _valid_mask(x, y, Q)
    T = int(valid.sum())
    x = x - x[valid].mean(axis=0)
    y = y - y[valid].mean(axis=0)
    X = np.zeros((x.shape[0], int(lags.sum())))
    col = 0
    for i, L in enumerate(lags):
        xi = np.nan_to_num(x[:, i])
        X[:, col:col + L] = toeplitz(xi, np.r_[xi[0], np.zeros(L - 1)])
        col += L
    Xv, yv = X[valid], y[valid]
    return Xv.T @ Xv, Xv.T @ yv, (yv**2).sum(axis=0), T


def _fit(Rxx, Rxy, ryy, gamma):
    Gamma = gamma * np.diag(np.diag(Rxx))
    h = np.linalg.solve(Rxx + Gamma, Rxy)
    Rxyest = Rxx @ h
    s2 = (h * Rxyest).sum(axis=0) - 2 * (h * Rxy).sum(axis=0) + ryy
    if gamma > 0:
        resid = Rxy - Rxyest
        bias = (resid * np.linalg.solve(Rxx, resid)).sum(axis=0) / s2 / 2
    else:
        bias = np.zeros_like(s2)
    return h, s2, bias


@dataclass
class VarxModel:
    A: np.ndarray                 # na x ydim(out) x ydim(in)
    B: np.ndarray                 # nb x ydim(out) x xdim
    s2: np.ndarray
    T: int
    n_params: int
    aic: float
    bic: float
    A_deviance: np.ndarray | None = field(default=None, repr=False)
    B_deviance: np.ndarray | None = field(default=None, repr=False)
    A_pval: np.ndarray | None = field(default=None, repr=False)
    B_pval: np.ndarray | None = field(default=None, repr=False)

    @property
    def A_R(self) -> np.ndarray:
        return np.sqrt(1 - np.exp(-self.A_deviance / self.T))

    @property
    def B_R(self) -> np.ndarray:
        return np.sqrt(1 - np.exp(-self.B_deviance / self.T))


def varx(Y: list[np.ndarray], na: int, X: list[np.ndarray] | None = None, nb: int = 0, lam: float = 0.0,
         maxlag: int | None = None, granger: bool = True) -> VarxModel:
    """Fit a VARX model to a list of records (time x ydim outputs, time x xdim inputs)."""
    Y = [np.asarray(y, dtype=float) for y in Y]
    if X is None or nb == 0:
        X, nb = [None] * len(Y), 0
    ydim = Y[0].shape[1]
    xdim = 0 if X[0] is None else np.asarray(X[0]).shape[1]
    lags = np.r_[np.full(ydim, na), np.full(xdim, nb)] if nb else np.full(ydim, na)
    Rxx = Rxy = ryy = 0
    T = 0
    for y, x in zip(Y, X):
        pred = y[:-1]
        if nb:
            pred = np.hstack([pred, np.asarray(x, dtype=float)[1:]])
        r1, r2, r3, t = myxcorr(pred, y[1:], lags, maxlag)
        Rxx, Rxy, ryy, T = Rxx + r1, Rxy + r2, ryy + r3, T + t
    gamma = 0.0 if not lam else lam / np.sqrt(T - lags.sum())
    AB, s2, bias = _fit(Rxx, Rxy, ryy, gamma)
    A = AB[: ydim * na].reshape(ydim, na, ydim).transpose(1, 2, 0)
    B = AB[ydim * na:].reshape(xdim, nb, ydim).transpose(1, 2, 0) if nb else np.zeros((0, ydim, 0))
    n_params = int(lags.sum())
    loglik = T * np.log(s2 / T)
    model = VarxModel(A=A, B=B, s2=s2 / T, T=T, n_params=n_params,
                      aic=float(np.sum(2 * n_params + loglik)), bic=float(np.sum(np.log(T) * n_params + loglik)))
    if granger:
        dev = np.zeros((ydim, lags.size))
        df = T - lags.sum()
        starts = np.r_[0, np.cumsum(lags)[:-1]]
        for i in range(lags.size):
            keep = np.setdiff1d(np.arange(lags.sum()), np.arange(starts[i], starts[i] + lags[i]))
            _, s2r, biasr = _fit(Rxx[np.ix_(keep, keep)], Rxy[keep], ryy, gamma)
            dev[:, i] = np.maximum(df * np.log(s2r / s2) - T * biasr + T * bias, 0)
        pval = 1 - chi2.cdf(dev, lags[None, :])
        model.A_deviance, model.B_deviance = dev[:, :ydim], dev[:, ydim:]
        model.A_pval, model.B_pval = pval[:, :ydim], pval[:, ydim:]
    return model


def varx_filter(A: np.ndarray, B: np.ndarray, x: np.ndarray, e: np.ndarray | None = None) -> np.ndarray:
    """Output of the VARX recursion driven by input x (and optional innovation e), zero initial state."""
    na, ydim = A.shape[0], A.shape[1]
    nb = B.shape[0]
    T = x.shape[0]
    Q = max(na, nb)
    xp = np.vstack([np.zeros((Q, x.shape[1])), x])
    drive = np.zeros((T + Q, ydim))
    for l in range(nb):
        drive[Q:] += xp[Q - l: Q - l + T] @ B[l].T
    if e is not None:
        drive[Q:] += e
    y = np.zeros((T + Q, ydim))
    # y[t] = sum_k A_k y[t-1-k] + drive[t]; history stacked newest first
    A_flat = np.concatenate(list(A), axis=1)                 # ydim x (na*ydim)
    hist = np.zeros(na * ydim)
    for t in range(Q, T + Q):
        yt = A_flat @ hist + drive[t]
        y[t] = yt
        hist = np.concatenate([yt, hist[:-ydim]]) if na > 1 else yt
    return y[Q:]


def decompose(model: VarxModel, y: np.ndarray, x: np.ndarray, y_mean: np.ndarray, x_mean: np.ndarray):
    """Stimulus-driven component H(z)x and the remainder (innovations propagated by A)."""
    ext = varx_filter(model.A, model.B, x - x_mean)
    return ext, (y - y_mean) - ext


def stable(A: np.ndarray) -> float:
    """Spectral radius of the companion matrix (< 1 for a stable recursion)."""
    na, n, _ = A.shape
    C = np.zeros((na * n, na * n))
    C[:n] = np.hstack(list(A))
    C[n:, :-n] = np.eye((na - 1) * n)
    return float(np.abs(np.linalg.eigvals(C)).max())
