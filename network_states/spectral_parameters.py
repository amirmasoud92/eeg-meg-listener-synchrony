"""Regional power spectra and their periodic and aperiodic parameters (specparam; Donoghue et al., 2020)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

def epoch_psd(x: np.ndarray, sfreq: float, epoch_s: float, epochs_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed periodograms of non-overlapping epochs averaged over the clean ones: (n_signals, n_freqs)."""
    n = int(round(epoch_s * sfreq))
    k = x.shape[-1] // n
    ep = x[:, : k * n].reshape(x.shape[0], k, n).transpose(1, 0, 2)[epochs_ok[:k]]
    f, p = ss.periodogram(ep, fs=sfreq, window="hann", detrend="constant", axis=-1)
    return f, p.mean(axis=0)


@dataclass(frozen=True)
class SpecparamSettings:
    freq_range: tuple[float, float]
    peak_width_limits: tuple[float, float]
    max_n_peaks: int
    min_peak_height: float
    peak_threshold: float
    aperiodic_mode: str


@dataclass
class SpectralFit:
    exponent: np.ndarray
    offset: np.ndarray
    knee: np.ndarray | None
    r_squared: np.ndarray
    error: np.ndarray
    freqs: np.ndarray
    flattened: np.ndarray

    def band_power(self, band: tuple[float, float], closed: bool) -> np.ndarray:
        lo, hi = band
        sel = (self.freqs >= lo) & ((self.freqs <= hi) if closed else (self.freqs < hi))
        return self.flattened[:, sel].mean(axis=1)


def fit_spectra(freqs: np.ndarray, psd: np.ndarray, settings: SpecparamSettings) -> SpectralFit:
    from specparam import SpectralModel

    lo, hi = settings.freq_range
    sel = (freqs >= lo) & (freqs <= hi)
    f = freqs[sel]
    n = psd.shape[0]
    exp_, off, knee, r2, err = (np.full(n, np.nan) for _ in range(5))
    flat = np.full((n, f.size), np.nan)
    for i in range(n):
        fm = SpectralModel(peak_width_limits=list(settings.peak_width_limits), max_n_peaks=settings.max_n_peaks,
                           min_peak_height=settings.min_peak_height, peak_threshold=settings.peak_threshold,
                           aperiodic_mode=settings.aperiodic_mode, verbose=False)
        fm.fit(freqs, psd[i], list(settings.freq_range))
        ap = fm.results.get_params("aperiodic")
        if settings.aperiodic_mode == "fixed":
            off[i], exp_[i] = ap
            ap_fit = off[i] - exp_[i] * np.log10(f)
        else:
            off[i], knee[i], exp_[i] = ap
            ap_fit = off[i] - np.log10(knee[i] + f ** exp_[i])
        r2[i] = fm.results.get_metrics("gof")
        err[i] = fm.results.get_metrics("error")
        flat[i] = np.log10(psd[i, sel]) - ap_fit
    return SpectralFit(exponent=exp_, offset=off, knee=knee if settings.aperiodic_mode != "fixed" else None,
                       r_squared=r2, error=err, freqs=f, flattened=flat)
