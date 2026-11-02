"""Compressed broadband speech envelope (Etard & Reichenbach, 2019)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import scipy.signal as ss

from preprocessing.signal_operations import fir_filter, fir_kernel

def audio_lowpass_order(sfreq: float, transition_hz: float) -> int:
    """Hamming-window order rule of EEGLAB firwsord: 3.3 / (df / fs), rounded up to even."""
    m = int(np.ceil(3.3 / (transition_hz / sfreq)))
    return m + (m % 2)


def broadband_envelope(audio: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Compressed envelope at 1 kHz."""
    b = fir_kernel(float(cfg["audio_lowpass_hz"]), float(fs), audio_lowpass_order(fs, cfg["audio_transition_hz"]), False)
    x = fir_filter(np.asarray(audio, float)[None], b)[0]
    fr = Fraction(int(cfg["intermediate_sfreq"]), int(round(fs))).limit_denominator(10000)
    x = ss.resample_poly(x, fr.numerator, fr.denominator)
    e = np.abs(x) ** float(cfg["compression"])
    return ss.resample_poly(e, 1000, int(cfg["intermediate_sfreq"]))
