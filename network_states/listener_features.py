"""Network-state features of a listener's trials, with epochs equalised across trials."""
from __future__ import annotations

import numpy as np

from common.settings import rng
from network_states.network_features import bands_from_config, extract
from network_states.spectral_parameters import SpecparamSettings
from preprocessing.aalborg_trials import CleanDataset

def listener_features(subject, paths, cfg, source, window=None):
    c4 = cfg.section("network_features")
    c2 = cfg.section("preprocessing")
    ds = CleanDataset(paths, cfg)
    W = np.load(paths.derived_root / "source_model" / f"filters_S{subject}.npz")[source]
    bands = bands_from_config(c4["bands"])
    sbands = bands_from_config(c4["spectral_bands"])
    spec = SpecparamSettings(**{k: tuple(v) if isinstance(v, list) else v for k, v in c4["specparam"].items()})
    rows, feats = [], []
    trials = ds.participant(subject, pad_s=c4["filter_pad_s"])
    a1 = c2["analysis_window_s"][0]
    ep = c2["connectivity_epoch_s"]
    if window is not None:
        e0, e1 = int(round((window[0] - a1) / ep)), int(round((window[1] - a1) / ep))
        n_eq = min(int(ct.ok_2s[e0:e1].sum()) for ct in trials)
    for ct in trials:
        t = ct.trial
        data, pad, ok, n_eq_t = W @ t.data, ct.pad, ct.ok_2s, ct.n_equalised_2s
        if window is not None:
            shift = int(round((window[0] - a1) * t.sfreq))
            data = data[:, : pad + int(round((window[1] - a1) * t.sfreq))]
            pad, ok, n_eq_t = pad + shift, ok[e0:e1], n_eq
        f = extract(data, pad, ok, n_eq_t, t.sfreq, bands, sbands, spec,
                    c2["connectivity_epoch_s"], c2["spectral_epoch_s"], c4["n_bootstraps"],
                    rng(cfg.project.seed, "equalise", t.key), key=t.key)
        feats.append(f)
        rows.append({"key": t.key, "subject": t.subject, "condition": t.condition, "excerpt": t.excerpt,
                     "presentation_index": t.presentation_index, "engagement": t.engagement,
                     "familiarity": t.familiarity, "n_clean_2s": int(ct.ok_2s.sum()),
                     "n_equalised_2s": ct.n_equalised_2s})
    return rows, feats
