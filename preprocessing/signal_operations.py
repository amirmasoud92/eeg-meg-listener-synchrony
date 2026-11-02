"""Signal operations shared by the EEG preprocessing chains.

Robust polynomial detrending (de Cheveigne & Arzounian, 2018), Hamming windowed-sinc FIR filters (one pass, delay
compensated, cutoff at -6 dB), bad-channel detection on the log standard deviation, spherical-spline interpolation with
average reference as a linear operator, and extended-Infomax ICA against anterior-channel proxies.
"""
from __future__ import annotations

from functools import lru_cache

import mne
import numpy as np
import scipy.signal as ss

from network_states.source_model import make_info
from readers import etard_release as etard

@lru_cache(maxsize=32)
def fir_kernel(cutoff_hz: float, sfreq: float, order: int, highpass: bool) -> np.ndarray:
    """Hamming windowed-sinc FIR of the given order (order + 1 taps), cutoff at -6 dB."""
    if order % 2:
        raise ValueError("FIR order must be even for delay compensation")
    return ss.firwin(order + 1, cutoff_hz, window="hamming", pass_zero=not highpass, fs=sfreq)


def fir_filter(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Zero-delay FIR filtering along the last axis with edge-value padding (EEGLAB firfilt)."""
    half = (b.size - 1) // 2
    xp = np.concatenate([np.repeat(x[..., :1], half, axis=-1), x, np.repeat(x[..., -1:], half, axis=-1)], axis=-1)
    return ss.oaconvolve(xp, b[(None,) * (x.ndim - 1)], mode="valid", axes=-1)


def robust_detrend(x: np.ndarray, order: int, threshold_sd: float, n_iter: int = 3) -> np.ndarray:
    """Weighted polynomial detrending with iterative outlier down-weighting (NoiseTools nt_detrend)."""
    n = x.shape[-1]
    t = np.linspace(-1.0, 1.0, n)
    basis = np.polynomial.legendre.legvander(t, order)
    out = np.empty_like(x, dtype=float)
    for c in range(x.shape[0]):
        w = np.ones(n, dtype=bool)
        y = np.asarray(x[c], float)
        for _ in range(n_iter):
            coef, *_ = np.linalg.lstsq(basis[w], y[w], rcond=None)
            r = y - basis @ coef
            sd = r[w].std()
            w_new = np.abs(r) < threshold_sd * sd
            if w_new.sum() < basis.shape[1] * 10 or np.array_equal(w_new, w):
                break
            w = w_new
        out[c] = r
    return out


def raw_part(root, subject: str, condition: str, part: int) -> tuple[etard.Recording, np.ndarray]:
    rec = etard.recording(root, subject, condition, part)
    idx = [i for i, c in enumerate(rec.header.channels) if c != "Sound"]
    res = np.asarray(rec.header.resolutions, float)[idx, None]
    return rec, np.asarray(rec.data()[:, idx], float).T * res


def bad_channels(broadband: list[np.ndarray], ch_names: list[str], robust_z: float) -> list[str]:
    """Channels whose log SD over the condition deviates by more than ``robust_z`` robust SDs, or are flat."""
    sd = np.concatenate(broadband, axis=1).std(axis=1)
    lsd = np.log(np.maximum(sd, 1e-12))
    med = np.median(lsd)
    mad = 1.4826 * np.median(np.abs(lsd - med))
    bad = (np.abs(lsd - med) / max(mad, 1e-12) > robust_z) | (sd < 1e-3)
    return [c for c, b in zip(ch_names, bad) if b]


def spatial_operator(ch_names: list[str], bads: list[str], sfreq: float) -> tuple[np.ndarray, list[str]]:
    """Linear map from recorded channels to average-referenced channels including PO4, with the
    listed bad channels replaced by spherical-spline interpolation."""
    names = ch_names + ["PO4"]
    info = make_info(names, sfreq)
    eye = np.vstack([np.eye(len(ch_names)), np.zeros((1, len(ch_names)))])
    raw = mne.io.RawArray(eye, info, verbose="error")
    raw.info["bads"] = list(bads)
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    op = raw.get_data()
    op = op - op.mean(axis=0, keepdims=True)
    return op, names


def fit_ica(broadband: np.ndarray, names: list[str], sfreq: float, rank: int, cfg: dict, seed: int) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Extended-Infomax ICA on the cleaned broadband data; returns the projection that removes the
    ocular components, their indices and their EOG-proxy scores."""
    info = make_info(names, sfreq)
    raw = mne.io.RawArray(broadband * 1e-6, info, verbose="error")
    ica = mne.preprocessing.ICA(n_components=rank, method="infomax", fit_params={"extended": cfg["extended"]},
                                random_state=seed, max_iter="auto", verbose="error")
    ica.fit(raw, verbose="error")
    proxies = [c for c in cfg["eog_proxies"] if c in names]
    idx, scores = ica.find_bads_eog(raw, ch_name=proxies, threshold=cfg["eog_z"], measure="zscore", verbose="error")
    score = np.max(np.abs(np.atleast_2d(scores)), axis=0)
    idx = sorted(idx, key=lambda k: -score[k])[: cfg["max_components"]]
    ica.exclude = list(idx)
    n = len(names)
    probe = mne.io.RawArray(np.hstack([np.eye(n), np.zeros((n, 1))]) * 1e-6, info, verbose="error")
    out = ica.apply(probe, verbose="error").get_data()
    proj = (out[:, :n] - out[:, n:]) / 1e-6
    return proj, list(idx), score


def rank_after(op: np.ndarray) -> int:
    s = np.linalg.svd(op, compute_uv=False)
    return int((s > s[0] * 1e-8).sum())
