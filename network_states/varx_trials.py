"""VARX inputs: regional signals at 100 Hz and phase-corrected envelopes, condition fits and component features."""
from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
import scipy.signal as ss

from network_states.connectivity_measures import aec_bootstrap, closest_orthogonal, dwpli_bootstrap, envelope_stats, epoch_fourier, imag_cross
from network_states.network_features import Band
from network_states.varx_model import VarxModel, decompose, stable, varx
from preprocessing.epoch_rejection import equalised_subsets
from preprocessing.release_filters import zero_phase_envelope

@dataclass
class VarxTrial:
    key: str
    subject: int
    condition: str
    y: np.ndarray = field(repr=False)          # samples x regions (100 Hz, 5-65 s)
    x: np.ndarray = field(repr=False)          # samples x 1 (phase-corrected envelope)
    ok: np.ndarray = field(repr=False)         # clean 2-s epochs within the window


def resample(x: np.ndarray, sfreq_in: float, sfreq_out: float, axis: int = -1) -> np.ndarray:
    from fractions import Fraction

    fr = Fraction(sfreq_out / sfreq_in).limit_denominator(1000)
    return ss.resample_poly(x, fr.numerator, fr.denominator, axis=axis)


def prepare_trial(regional_512: np.ndarray, pad_s: float, envelope_128: np.ndarray, ok_2s_trial: np.ndarray,
                  trial_start_s: float, window_s: tuple[float, float], sfreq_in: float, sfreq_out: float,
                  env_sfreq: float, env_pad_s: float, epoch_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regional signals and envelope on the 100 Hz grid, plus the clean-epoch mask of that window.

    ``regional_512`` starts ``pad_s`` seconds before ``trial_start_s``; it is resampled as a whole so
    that resampling transients fall outside the window. The envelope spans the window exactly.
    """
    y = resample(regional_512, sfreq_in, sfreq_out, axis=-1)
    t0 = trial_start_s - pad_s
    i0, i1 = int(round((window_s[0] - t0) * sfreq_out)), int(round((window_s[1] - t0) * sfreq_out))
    y = y[:, i0:i1].T
    env = zero_phase_envelope(envelope_128, env_sfreq, pad_s=env_pad_s)
    x = resample(env, env_sfreq, sfreq_out)[:, None]
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"envelope ({x.shape[0]}) and EEG ({y.shape[0]}) lengths differ")
    first = int(round((window_s[0] - trial_start_s) / epoch_s))
    n_ep = int(round((window_s[1] - window_s[0]) / epoch_s))
    return y, x, ok_2s_trial[first:first + n_ep]


def nan_bad_epochs(y: np.ndarray, ok: np.ndarray, epoch_n: int) -> np.ndarray:
    out = y.copy()
    for k in np.flatnonzero(~ok):
        out[k * epoch_n:(k + 1) * epoch_n] = np.nan
    return out


def equalise_fit_epochs(trials: list[VarxTrial], n_keep: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Clean-epoch masks thinned at random so the condition contributes ``n_keep`` epochs in total."""
    pool = [(i, k) for i, t in enumerate(trials) for k in np.flatnonzero(t.ok)]
    keep = set(map(tuple, np.array(pool)[rng.choice(len(pool), n_keep, replace=False)].tolist()))
    return [np.array([(i, k) in keep for k in range(t.ok.size)]) for i, t in enumerate(trials)]


@dataclass
class ConditionFit:
    model: VarxModel
    y_mean: np.ndarray
    x_mean: np.ndarray
    spectral_radius: float
    explained_by_input: float


def fit_condition(trials: list[VarxTrial], fit_masks: list[np.ndarray], na: int, nb: int, lam: float,
                  epoch_n: int, granger: bool = True, diagnostics: bool = True) -> ConditionFit:
    Y = [nan_bad_epochs(t.y, m, epoch_n) for t, m in zip(trials, fit_masks)]
    X = [t.x for t in trials]
    model = varx(Y, na, X, nb, lam, granger=granger)
    ym = np.nanmean(np.vstack(Y), axis=0)
    xm = np.mean(np.vstack(X), axis=0)
    explained = float("nan")
    if diagnostics:
        ext = [decompose(model, t.y, t.x, ym, xm)[0] for t in trials]
        var_y = np.mean([np.nanvar(y, axis=0) for y in Y], axis=0)
        var_e = np.mean([e.var(axis=0) for e in ext], axis=0)
        explained = float(np.mean(var_e / var_y))
    return ConditionFit(model=model, y_mean=ym, x_mean=xm, spectral_radius=stable(model.A),
                        explained_by_input=explained)


def component_features(sig: np.ndarray, ok: np.ndarray, n_equalised: int, sfreq: float, bands: list[Band],
                       epoch_s: float, spectral_epoch_s: float, n_boot: int, rng: np.random.Generator) -> dict:
    """dwPLI, orthogonalised AEC and log band power (no aperiodic model) for regions x samples."""
    subsets = equalised_subsets(ok, n_equalised, n_boot, rng)
    freqs, F = epoch_fourier(sig, sfreq, epoch_s)
    F = F[: ok.size]
    dw = np.stack([dwpli_bootstrap(imag_cross(F, b.select(freqs)), subsets) for b in bands])
    n_ep = int(round(epoch_s * sfreq))
    clean = np.zeros(sig.shape[1], dtype=bool)
    clean[: ok.size * n_ep] = np.repeat(ok, n_ep)
    aec = []
    for b in bands:
        z = mne.filter.filter_data(sig, sfreq, b.lo, b.hi, method="fir", phase="zero", fir_design="firwin",
                                   verbose=False)
        L = closest_orthogonal((z[:, clean] - z[:, clean].mean(axis=1, keepdims=True)).T)
        env = np.abs(ss.hilbert((z.T @ L).T, axis=-1))
        sums, cross, n = envelope_stats(env, sfreq, epoch_s)
        aec.append(aec_bootstrap(sums[: ok.size], cross[: ok.size], n, subsets))
    ok4 = ok[: 2 * (ok.size // 2)].reshape(-1, 2).all(axis=1)
    n4 = int(round(spectral_epoch_s * sfreq))
    k4 = sig.shape[1] // n4
    ep = sig[:, : k4 * n4].reshape(sig.shape[0], k4, n4).transpose(1, 0, 2)[ok4[:k4]]
    f, p = ss.periodogram(ep, fs=sfreq, window="hann", detrend="constant", axis=-1)
    p = p.mean(axis=0)
    power = np.stack([np.log10(p[:, b.select(f)].mean(axis=1)) for b in bands])
    return {"dwpli": dw, "aec": np.stack(aec), "logpower": power}
