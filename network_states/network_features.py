"""Trial-level network-state features and their network-summarised representation."""
from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
import scipy.signal as ss

from network_states.connectivity_measures import aec_bootstrap, closest_orthogonal, dwpli_bootstrap, envelope_stats, epoch_fourier, imag_cross, pair_indices
from network_states.spectral_parameters import SpecparamSettings, epoch_psd, fit_spectra
from preprocessing.epoch_rejection import equalised_subsets

BAND_NAMES = ("delta", "theta", "alpha", "beta")
NETWORKS = {
    "AUD": ("TTG", "STG"),
    "TLG": ("MTG",),
    "IFM": ("IFG", "PreCG", "RMFG"),
    "PAR": ("SMG", "IPL"),
    "MPC": ("PCun",),
}


@dataclass(frozen=True)
class Band:
    name: str
    lo: float
    hi: float
    closed: bool

    def select(self, freqs: np.ndarray) -> np.ndarray:
        upper = freqs <= self.hi if self.closed else freqs < self.hi
        return np.flatnonzero((freqs >= self.lo) & upper)


def bands_from_config(spec: dict) -> list[Band]:
    return [Band(n, float(spec[n]["range"][0]), float(spec[n]["range"][1]), bool(spec[n]["closed"])) for n in BAND_NAMES]


@dataclass
class TrialFeatures:
    key: str
    dwpli: np.ndarray            # bands x pairs
    aec: np.ndarray              # bands x pairs
    power: np.ndarray            # bands x regions (aperiodic-removed log10 power)
    exponent: np.ndarray         # regions
    offset: np.ndarray           # regions
    r_squared: np.ndarray = field(repr=False)
    fit_error: np.ndarray = field(repr=False)


def extract(regional: np.ndarray, pad: int, ok_2s: np.ndarray, n_equalised: int, sfreq: float,
            bands: list[Band], spectral_bands: list[Band], spec: SpecparamSettings, epoch_s: float,
            spectral_epoch_s: float, n_boot: int, rng: np.random.Generator, key: str = "") -> TrialFeatures:
    """Features of one trial from its regional time courses.

    ``regional`` holds ``pad`` samples preceding the analysis window; band-pass filtering is
    applied to the padded signal so that filter onset transients fall outside the window.
    """
    x = regional[:, pad:]
    subsets = equalised_subsets(ok_2s, n_equalised, n_boot, rng)

    freqs, F = epoch_fourier(x, sfreq, epoch_s)
    F = F[: ok_2s.size]
    dwpli = np.stack([dwpli_bootstrap(imag_cross(F, b.select(freqs)), subsets) for b in bands])

    n_ep = int(round(epoch_s * sfreq))
    clean = np.zeros(x.shape[1], dtype=bool)
    clean[: ok_2s.size * n_ep] = np.repeat(ok_2s, n_ep)
    aec = []
    for b in bands:
        z = mne.filter.filter_data(regional, sfreq, b.lo, b.hi, method="fir", phase="zero", fir_design="firwin",
                                   verbose=False)[:, pad:]
        L = closest_orthogonal((z[:, clean] - z[:, clean].mean(axis=1, keepdims=True)).T)
        o = (z.T @ L).T
        env = np.abs(ss.hilbert(o, axis=-1))
        sums, cross, n = envelope_stats(env, sfreq, epoch_s)
        aec.append(aec_bootstrap(sums[: ok_2s.size], cross[: ok_2s.size], n, subsets))
    aec = np.stack(aec)

    ok_4s = ok_2s[: 2 * (ok_2s.size // 2)].reshape(-1, 2).all(axis=1)
    f, psd = epoch_psd(x, sfreq, spectral_epoch_s, ok_4s)
    fit = fit_spectra(f, psd, spec)
    power = np.stack([fit.band_power((b.lo, b.hi), b.closed) for b in spectral_bands])
    return TrialFeatures(key=key, dwpli=dwpli, aec=aec, power=power, exponent=fit.exponent, offset=fit.offset,
                         r_squared=fit.r_squared, fit_error=fit.error)


def network_membership(region_names: list[str]) -> np.ndarray:
    base = [r.split("_")[0] for r in region_names]
    lookup = {reg: k for k, (net, regs) in enumerate(NETWORKS.items()) for reg in regs}
    return np.array([lookup[b] for b in base])


def network_edge_means(values: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Average pairwise values (..., n_pairs) into within- then between-network means (..., 15)."""
    i, j = pair_indices(membership.size)
    ni, nj = membership[i], membership[j]
    k = len(NETWORKS)
    blocks = [(a, a) for a in range(k)] + [(a, b) for a in range(k) for b in range(a + 1, k)]
    out = []
    for a, b in blocks:
        sel = ((ni == a) & (nj == b)) | ((ni == b) & (nj == a))
        out.append(values[..., sel].mean(axis=-1))
    return np.stack(out, axis=-1)


def network_node_means(values: np.ndarray, membership: np.ndarray) -> np.ndarray:
    return np.stack([values[..., membership == a].mean(axis=-1) for a in range(len(NETWORKS))], axis=-1)


def band_vector(feat: dict[str, np.ndarray], band: int, summarised: bool, membership: np.ndarray) -> np.ndarray:
    """Pattern vector of one band for arrays stacked over trials (trials x features)."""
    if summarised:
        parts = [network_edge_means(feat["dwpli"][:, band], membership),
                 network_edge_means(feat["aec"][:, band], membership),
                 network_node_means(feat["power"][:, band], membership),
                 network_node_means(feat["exponent"], membership),
                 network_node_means(feat["offset"], membership)]
    else:
        parts = [feat["dwpli"][:, band], feat["aec"][:, band], feat["power"][:, band], feat["exponent"], feat["offset"]]
    return np.concatenate(parts, axis=1)


def stacked_vector(feat: dict[str, np.ndarray], bands: list[int], summarised: bool, membership: np.ndarray) -> np.ndarray:
    """Multiband pattern: band-specific blocks for the listed bands plus the aperiodic block once."""
    blocks = []
    for b in bands:
        v = band_vector(feat, b, summarised, membership)
        n_ap = 2 * (len(NETWORKS) if summarised else membership.size)
        blocks.append(v[:, :-n_ap])
    blocks.append(v[:, -n_ap:])
    return np.concatenate(blocks, axis=1)
