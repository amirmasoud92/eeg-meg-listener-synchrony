"""Residual-artefact rejection and epoch equalisation.

FASTER epoch criteria (Nolan, Whelan & Reilly, 2010): for every epoch of a listener, with all conditions pooled, the
channel-averaged amplitude range, variance and deviation of the channel means are z-scored across epochs, and epochs
with any |z| > 3 are rejected. An epoch is also rejected when any channel's log peak-to-peak amplitude exceeds a robust
z of 5 (median/MAD; Bigdely-Shamlo et al., 2015). A trial is excluded when more than one third of its epochs are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocessing.epochs import Trial, epoch_view

@dataclass(frozen=True)
class RejectionResult:
    threshold_uv: float
    epoch_ok: dict[str, np.ndarray]
    trial_ok: dict[str, bool]


def faster_epoch_properties(epochs: np.ndarray, grand_channel_mean: np.ndarray) -> np.ndarray:
    """Nolan et al. (2010) epoch properties: amplitude range, variance, channel deviation."""
    amp_range = (epochs.max(axis=-1) - epochs.min(axis=-1)).mean(axis=1)
    variance = epochs.var(axis=-1).mean(axis=1)
    deviation = np.abs(epochs.mean(axis=-1) - grand_channel_mean[None, :]).mean(axis=1)
    return np.column_stack([amp_range, variance, deviation])


def robust_channel_outliers(epochs: np.ndarray, reference: np.ndarray, z_threshold: float) -> np.ndarray:
    """Epochs in which any channel's log peak-to-peak exceeds a robust z (per channel) of ``z_threshold``."""
    from scipy.stats import median_abs_deviation

    ref = np.log(reference.max(axis=-1) - reference.min(axis=-1))
    med, scale = np.median(ref, axis=0), 1.4826 * median_abs_deviation(ref, axis=0)
    lp = np.log(epochs.max(axis=-1) - epochs.min(axis=-1))
    return ((lp - med) / scale > z_threshold).any(axis=1)


def reject(trials: list[Trial], ch_names: list[str], length_s: float, seed: int,
           max_bad_fraction: float = 1 / 3, method: str = "faster", z_threshold: float = 3.0,
           channel_robust_z: float | None = 5.0) -> RejectionResult:
    views = {t.key: epoch_view(t.data, t.sfreq, length_s) for t in trials}
    epoch_ok, trial_ok = {}, {}
    if method == "faster":
        pooled = np.concatenate(list(views.values()))
        grand = pooled.mean(axis=(0, 2))
        props = faster_epoch_properties(pooled, grand)
        mu, sd = props.mean(axis=0), props.std(axis=0)
        for key, ep in views.items():
            z = (faster_epoch_properties(ep, grand) - mu) / sd
            bad = (np.abs(z) > z_threshold).any(axis=1)
            if channel_robust_z is not None:
                bad |= robust_channel_outliers(ep, pooled, channel_robust_z)
            epoch_ok[key] = ~bad
        thr = float("nan")
    else:
        raise ValueError(f"unknown rejection method '{method}'")
    for key, ok in epoch_ok.items():
        trial_ok[key] = bool((~ok).mean() <= max_bad_fraction)
    return RejectionResult(threshold_uv=thr, epoch_ok=epoch_ok, trial_ok=trial_ok)


def equalised_subsets(available: np.ndarray, n_keep: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap draws (without replacement within a draw) of ``n_keep`` clean epoch indices.

    Returns an array (n_boot, n_keep); when every clean epoch is kept, all draws coincide and a
    single row is returned.
    """
    idx = np.flatnonzero(available)
    if idx.size < n_keep:
        raise ValueError(f"{idx.size} clean epochs < {n_keep} requested")
    if idx.size == n_keep:
        return idx[None, :]
    return np.stack([np.sort(rng.choice(idx, n_keep, replace=False)) for _ in range(n_boot)])
