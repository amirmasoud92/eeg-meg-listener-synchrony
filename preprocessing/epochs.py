"""Trial container, epoch views and the Aalborg trial table."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from readers.aalborg_release import load_preprocessed, preprocessed_path

@dataclass
class Trial:
    subject: int
    condition: str
    excerpt: int
    label: str
    presentation_index: int
    engagement: float
    familiarity: float
    sfreq: float
    t0: float
    data: np.ndarray = field(repr=False)

    @property
    def key(self) -> str:
        return f"S{self.subject}_{self.condition}{self.excerpt}"


def average_reference(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def crop(x: np.ndarray, sfreq: float, window_s: tuple[float, float]) -> np.ndarray:
    i0, i1 = int(round(window_s[0] * sfreq)), int(round(window_s[1] * sfreq))
    if i1 > x.shape[-1]:
        raise ValueError(f"window {window_s} exceeds trial length {x.shape[-1] / sfreq:.2f} s")
    return x[..., i0:i1]


def load_participant(preprocessed_dir: Path, trial_table: pd.DataFrame, subject: int,
                     window_s: tuple[float, float], sfreq: float) -> list[Trial]:
    """Average-referenced trials of one participant cropped to ``window_s`` (seconds from onset)."""
    raw = load_preprocessed(preprocessed_path(preprocessed_dir, subject), subject)
    if raw.info.eeg_sfreq != sfreq:
        raise ValueError(f"S{subject}: sampling rate {raw.info.eeg_sfreq} != {sfreq}")
    meta = trial_table[trial_table.subject == subject].set_index("label")
    trials = []
    for t in raw.trials:
        m = meta.loc[t.label]
        x = average_reference(t.eeg.astype(np.float64))
        trials.append(Trial(subject=subject, condition=t.condition, excerpt=t.excerpt, label=t.label,
                            presentation_index=int(m.presentation_index), engagement=t.engagement,
                            familiarity=t.familiarity, sfreq=sfreq, t0=window_s[0],
                            data=crop(x, sfreq, window_s)))
    return trials


def epoch_view(x: np.ndarray, sfreq: float, length_s: float) -> np.ndarray:
    """Non-overlapping epochs, shape (n_epochs, n_channels, n_samples_epoch); trailing remainder dropped."""
    n = int(round(length_s * sfreq))
    k = x.shape[-1] // n
    return x[..., : k * n].reshape(x.shape[0], k, n).transpose(1, 0, 2)


def coarse_mask(fine_ok: np.ndarray, factor: int) -> np.ndarray:
    """Validity of long epochs built from ``factor`` consecutive short epochs."""
    k = fine_ok.size // factor
    return fine_ok[: k * factor].reshape(k, factor).all(axis=1)
