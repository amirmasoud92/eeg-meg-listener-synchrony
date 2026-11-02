"""Filters of the Aalborg release, reproduced for envelope alignment."""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

def release_bandpass_sos(sfreq: float, band: tuple[float, float] = (0.5, 8.0), order: int = 10) -> np.ndarray:
    """Butterworth band-pass equivalent to MATLAB designfilt('bandpassiir', FilterOrder=order)."""
    if order % 2:
        raise ValueError("band-pass order must be even")
    return ss.butter(order // 2, band, btype="bandpass", fs=sfreq, output="sos")


def zero_phase_envelope(envelope: np.ndarray, sfreq: float, pad_s: float = 10.0) -> np.ndarray:
    """Cancel the phase of the release's causal envelope filter.

    Passing the causally filtered envelope backwards through the same filter yields
    the zero-phase response |H|^2 (the filtfilt equivalent). The backward pass
    has no future samples, so the end is extended by odd reflection of ``pad_s``
    seconds, the scheme used by filtfilt.
    """
    env = np.asarray(envelope, dtype=float)
    sos = release_bandpass_sos(sfreq)
    n = int(round(pad_s * sfreq))
    if n >= env.shape[-1]:
        raise ValueError("padding longer than signal")
    tail = 2 * env[..., -1:] - env[..., -2:-n - 2:-1]
    ext = np.concatenate([env, tail], axis=-1)
    back = ss.sosfilt(sos, ext[..., ::-1], axis=-1)[..., ::-1]
    return back[..., : env.shape[-1]]
