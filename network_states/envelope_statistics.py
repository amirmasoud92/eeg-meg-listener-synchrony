"""Trial-level envelope statistics, used as covariates and for the stimulus benchmark of the language-capture index.

Log modulation power in octave bands 0.5-1, 1-2, 2-4 and 4-8 Hz, log frequency of the modulation peak and log envelope
standard deviation; covariates are the leading principal components up to 95 % explained variance.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

OCTAVES = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))


def envelope_statistics(envelope: np.ndarray, sfreq: float, window_s: float = 8.0) -> np.ndarray:
    f, p = ss.welch(envelope, fs=sfreq, nperseg=int(window_s * sfreq))
    sel = (f >= OCTAVES[0][0]) & (f <= OCTAVES[-1][1])
    bands = [np.log10(p[(f >= a) & (f < b)].mean()) for a, b in OCTAVES]
    return np.r_[bands, np.log10(f[sel][np.argmax(p[sel])]), np.log10(envelope.std())]


class AcousticCovariates:
    def __init__(self, stimulus_envelopes: dict[str, np.ndarray], sfreq: float, variance: float = 0.95,
                 window_s: float = 8.0):
        self.labels = sorted(stimulus_envelopes)
        S = np.array([envelope_statistics(stimulus_envelopes[k], sfreq, window_s) for k in self.labels])
        self.mean, self.sd = S.mean(axis=0), S.std(axis=0)
        Zs = (S - self.mean) / self.sd
        ev, vec = np.linalg.eigh(np.cov(Zs.T))
        ev, vec = ev[::-1], vec[:, ::-1]
        self.k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), variance) + 1)
        self.explained = (ev / ev.sum())[: self.k]
        self.components = vec[:, : self.k]
        self.statistics = dict(zip(self.labels, S))
        self.scores = dict(zip(self.labels, Zs @ self.components))

    def __call__(self, labels: list[str]) -> np.ndarray:
        return np.array([self.scores[l] for l in labels])
