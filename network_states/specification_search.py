"""Specification space of network-state representations and search-wide inference.

A specification combines a feature family, a set of frequency bands and a granularity. Inference controls the
family-wise error over the whole space with the max-statistic sign-flip test (Westfall & Young, 1993; Nichols & Holmes,
2002), and selection is by a hypothesis-neutral criterion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from network_states.network_features import BAND_NAMES, network_edge_means, network_node_means

FAMILIES = {
    "dwpli": ("dwpli",),
    "aec": ("aec",),
    "power": ("power",),
    "connectivity": ("dwpli", "aec"),
    "spectral": ("power", "aperiodic"),
    "all": ("dwpli", "aec", "power", "aperiodic"),
}
BANDSETS = {
    "delta": ("delta",), "theta": ("theta",), "alpha": ("alpha",), "beta": ("beta",),
    "delta_theta": ("delta", "theta"), "alpha_beta": ("alpha", "beta"),
    "all_bands": ("delta", "theta", "alpha", "beta"),
}


@dataclass(frozen=True)
class Spec:
    family: str
    bands: str
    summarised: bool

    @property
    def name(self) -> str:
        return f"{'net' if self.summarised else 'full'}|{self.family}|{self.bands}"


def specifications() -> list[Spec]:
    out = []
    for summ in (True, False):
        for fam in FAMILIES:
            for bs in BANDSETS:
                out.append(Spec(fam, bs, summ))
        out.append(Spec("aperiodic", "none", summ))
    return out


def build(feat: dict, spec: Spec, membership: np.ndarray) -> np.ndarray:
    """Pattern matrix (trials x features) of one specification."""
    parts = []
    members = FAMILIES.get(spec.family, ("aperiodic",))
    if spec.bands != "none":
        for band in BANDSETS[spec.bands]:
            b = BAND_NAMES.index(band)
            for m in members:
                if m in ("dwpli", "aec"):
                    v = feat[m][:, b]
                    parts.append(network_edge_means(v, membership) if spec.summarised else v)
                elif m == "power":
                    v = feat["power"][:, b]
                    parts.append(network_node_means(v, membership) if spec.summarised else v)
    if "aperiodic" in members:
        for k in ("exponent", "offset"):
            v = feat[k]
            parts.append(network_node_means(v, membership) if spec.summarised else v)
    return np.concatenate(parts, axis=1)


def signflip_maxt(S: np.ndarray, n_perm: int | None = None, seed: int = 0, alternative: str = "greater",
                  exact_max_n: int = 18, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uncorrected and family-wise corrected p-values for one-sample t tests of each column.

    S is participants x specifications (NaN-free). The same sign flip is applied to all columns of a
    participant, preserving the dependence between specifications. Returns (t, p_unc, p_fwe).
    """
    S = np.asarray(S, float)
    n, V = S.shape
    sumsq = (S ** 2).sum(0)

    def tstat(means):
        var = (sumsq - n * means ** 2) / (n - 1)
        return means / np.sqrt(np.maximum(var, 1e-300) / n)

    t_obs = tstat(S.mean(0))
    sign = {"greater": 1.0, "less": -1.0, "two-sided": None}[alternative]
    stat = (lambda t: sign * t) if sign is not None else np.abs
    obs = stat(t_obs)
    if n <= exact_max_n:
        codes = np.arange(2 ** n, dtype=np.int64)
        flips_iter = (1 - 2 * ((codes[i:i + chunk, None] >> np.arange(n)) & 1) for i in range(0, codes.size, chunk))
        total = codes.size
    else:
        g = np.random.default_rng(seed)
        total = n_perm or 20000
        flips_iter = (g.choice([-1.0, 1.0], size=(min(chunk, total - i), n)) for i in range(0, total, chunk))
    ge_unc = np.zeros(V)
    max_null = []
    for F in flips_iter:
        T = stat(tstat(F.astype(float) @ S / n))
        ge_unc += (T >= obs[None, :] - 1e-12).sum(0)
        max_null.append(T.max(1))
    max_null = np.concatenate(max_null)
    p_unc = ge_unc / total
    p_fwe = (max_null[:, None] >= obs[None, :] - 1e-12).mean(0)
    return t_obs, p_unc, p_fwe
