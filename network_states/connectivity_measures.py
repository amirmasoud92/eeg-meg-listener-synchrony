"""Leakage-robust connectivity: debiased weighted phase-lag index (Vinck et al., 2011) and orthogonalised amplitude-envelope correlation (Hipp et al., 2012; Colclough et al., 2015)."""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

def pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def epoch_fourier(x: np.ndarray, sfreq: float, epoch_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann-tapered Fourier coefficients per non-overlapping epoch: (n_epochs, n_signals, n_freqs)."""
    n = int(round(epoch_s * sfreq))
    k = x.shape[-1] // n
    ep = x[:, : k * n].reshape(x.shape[0], k, n).transpose(1, 0, 2)
    ep = ep - ep.mean(axis=-1, keepdims=True)
    F = np.fft.rfft(ep * ss.windows.hann(n, sym=False), axis=-1)
    return np.fft.rfftfreq(n, 1.0 / sfreq), F


def imag_cross(F: np.ndarray, freq_idx: np.ndarray) -> np.ndarray:
    """Im(S_ij) per epoch for all pairs i<j at the selected bins: (n_epochs, n_pairs, n_bins)."""
    i, j = pair_indices(F.shape[1])
    Fs = F[:, :, freq_idx]
    return np.imag(Fs[:, i, :] * np.conj(Fs[:, j, :]))


def dwpli_from_sums(s_im: np.ndarray, s_abs: np.ndarray, s_sq: np.ndarray) -> np.ndarray:
    """Debiased squared WPLI from epoch sums of Im, |Im| and Im^2."""
    num = s_im**2 - s_sq
    den = s_abs**2 - s_sq
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, 0.0)
    return out


def dwpli_bootstrap(im: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    """Band-averaged dwPLI averaged over epoch subsets.

    im: (n_epochs, n_pairs, n_bins) imaginary cross-spectra; subsets: (n_boot, n_keep) epoch indices.
    Returns (n_pairs,).
    """
    B = np.zeros((subsets.shape[0], im.shape[0]))
    np.put_along_axis(B, subsets, 1.0, axis=1)
    flat = im.reshape(im.shape[0], -1)
    s_im = (B @ flat).reshape(-1, *im.shape[1:])
    s_abs = (B @ np.abs(flat)).reshape(-1, *im.shape[1:])
    s_sq = (B @ flat**2).reshape(-1, *im.shape[1:])
    return dwpli_from_sums(s_im, s_abs, s_sq).mean(axis=2).mean(axis=0)


def closest_orthogonal(Z: np.ndarray, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Colclough et al. (2015) 'closest' orthogonalisation; returns L (n x n) with O = Z @ L.

    Z: samples x signals (demeaned). Iterates P = polar(Z D), D = diag(Z' P) until the
    squared error stabilises; O = P D has mutually orthogonal columns closest to Z.
    """
    n = Z.shape[1]
    d = np.ones(n)
    prev = np.inf
    L = np.eye(n)
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(Z * d, full_matrices=False)
        P = U @ Vt
        d_new = np.einsum("ti,ti->i", Z, P)
        # U = Z D V S^-1, hence O = P D_new = Z (D V S^-1 V' D_new)
        L = (d[:, None] * (Vt.T / s) @ Vt) * d_new[None, :]
        err = np.sum((Z - P * d_new) ** 2)
        d = d_new
        if abs(prev - err) <= tol * max(err, 1.0):
            break
        prev = err
    return L


def envelope_stats(env: np.ndarray, sfreq: float, epoch_s: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-epoch sufficient statistics of amplitude envelopes: sums (E x n) and cross-products (E x n x n)."""
    n = int(round(epoch_s * sfreq))
    k = env.shape[-1] // n
    ep = env[:, : k * n].reshape(env.shape[0], k, n).transpose(1, 0, 2)
    return ep.sum(axis=-1), np.einsum("ein,ejn->eij", ep, ep), n


def aec_bootstrap(sums: np.ndarray, cross: np.ndarray, n_per_epoch: int, subsets: np.ndarray) -> np.ndarray:
    """Envelope correlation matrix averaged over epoch subsets (upper triangle, n_pairs)."""
    i, j = pair_indices(sums.shape[1])
    out = np.zeros(i.size)
    for sub in subsets:
        N = n_per_epoch * sub.size
        s = sums[sub].sum(axis=0)
        c = cross[sub].sum(axis=0)
        cov = c / N - np.outer(s, s) / N**2
        d = np.sqrt(np.diag(cov))
        out += (cov / np.outer(d, d))[i, j]
    return out / subsets.shape[0]
