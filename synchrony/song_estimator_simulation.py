"""Simulation of the speech-specific excess L under a natural-voice null and with a lyric source.

Leave-one-song-out spaces with songs as items, calibrated to the observed ISC levels. Sung versions carry the hummed
sources with a larger shared gain; under the alternative they also carry a lyric source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from statistical_tests.inference import summarise
from synchrony.correlated_components import _cov_terms, corrca_from_terms, subject_isc

def mean_isc(X, W, ok=None) -> float:
    return float(np.nanmean(subject_isc(X, W, ok).sum(1)))


def _smooth(g, T):
    x = np.convolve(g.normal(size=T + 40), np.hanning(21), "same")[20:T + 20]
    return x / x.std()


def simulate(g, lyric_in_sung: float, S=18, N=18, D=32, T=5000, voice_gain=1.8, noise=48.0) -> pd.DataFrame:
    """Calibrated to the observed values: hummed ISC ~0.10 and sung ~0.18 (3 components) with voice_gain 1.8, noise 48."""
    a_sh = g.normal(size=D)
    a_mel = 0.6 * a_sh + 0.8 * g.normal(size=D)
    a_ly = 0.6 * a_sh + 0.8 * g.normal(size=D)
    mix = g.normal(size=(D, D)) / np.sqrt(D)

    def version(pats, gains):
        src = [_smooth(g, T) for _ in pats]
        X = sum(gn * p[None, :, None] * s[None, None, :] for gn, p, s in zip(gains, pats, src))
        return X + noise * (np.einsum("de,net->ndt", mix, g.normal(size=(N, D, T))) + 0.5 * g.normal(size=(N, D, T)))

    songs = [{"spoken": version([a_sh, a_ly], [voice_gain, 1.0]), "hummed": version([a_sh, a_mel], [1.0, 1.0]),
              "sung": version([a_sh, a_mel, a_ly], [voice_gain, 1.0, lyric_in_sung])} for _ in range(S)]
    terms = [{v: _cov_terms(s[v]) for v in ("spoken", "hummed")} for s in songs]
    rows = []
    for i, s in enumerate(songs):
        tot = {v: [sum(t[v][j] for k, t in enumerate(terms) if k != i) for j in (0, 1)] for v in ("spoken", "hummed")}
        row = {"song": i}
        for k in (1, 3, 5):
            Wsp = corrca_from_terms(*tot["spoken"], N, 0.5, k)[0]
            Wm = corrca_from_terms(*tot["hummed"], N, 0.5, k)[0]
            isc = {(sp, v): mean_isc(s[v], W) for sp, W in (("sp", Wsp), ("mel", Wm)) for v in ("sung", "hummed")}
            row[f"L_sep_{k}"] = (isc[("sp", "sung")] - isc[("sp", "hummed")]) - (isc[("mel", "sung")] - isc[("mel", "hummed")])
            if k == 3:
                row |= {f"isc_{sp}_{v}": val for (sp, v), val in isc.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def run_sim(seed: int, n_rep: int = 30) -> dict:
    out = {}
    for name, lyric in (("null_voice_gain_only", 0.0), ("alternative_lyric_source", 1.0)):
        stats, levels = {}, []
        for r in range(n_rep):
            df = simulate(np.random.default_rng([seed, r, int(lyric * 10)]), lyric)
            levels.append(df.filter(like="isc_").mean())
            for c in ("L_sep_1", "L_sep_3", "L_sep_5"):
                sm = summarise(df[c].to_numpy(), "greater")
                stats.setdefault(c, []).append((sm["mean"], sm["dz"], sm["p_signflip"] < 0.05))
        out[name] = {c: {"mean": float(np.mean([m for m, _, _ in v])), "mean_dz": float(np.mean([d for _, d, _ in v])),
                         "rejection_rate_alpha05": float(np.mean([r for _, _, r in v]))} for c, v in stats.items()}
        out[name]["isc_levels_3comp"] = pd.concat(levels, axis=1).mean(1).round(4).to_dict()
    out["n_replications"], out["songs"], out["listeners"] = n_rep, 18, 18
    return out
