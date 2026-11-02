"""Envelope tracking in the MEG attention dataset: attended and ignored talker, true and mismatched chapters."""
from __future__ import annotations

import numpy as np

from encoding_models import ridge_encoding as E
from readers import meg_attention as C

def envelope_rows(f: str, k: int, stream: str, T: int, shift: int) -> np.ndarray:
    k2 = (k - 1 + shift) % 5 + 1
    raw = C.features(f, k2)[f"{stream}_acoustic_raw"][:2]
    idx = np.arange(T) % raw.shape[1]
    return C.bz(raw[:, idx])


def listener_tracking(p: int) -> dict:
    if not C.included(p):
        return {"listener": p, "included": False}
    trials = C.listener_trials(p)
    row = {"listener": p, "included": True, "n_chapters": len(trials)}
    masks = [tr["v"] for tr in trials]
    for stream in C.STREAMS:
        for s in range(5):
            for tr in trials:
                tr["F"]["env"] = envelope_rows(tr["file"], tr["chapter"], stream, tr["T"], s)
            preds, lam = E.fit_predict(trials, ["env"], None, C.N_LAGS)
            tag = "true" if s == 0 else f"mis{s}"
            row[f"r_{stream}_{tag}"] = E.r_mean(trials, preds, masks, None)
            row[f"lam_{stream}_{tag}"] = lam
    return row
