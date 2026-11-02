"""Envelope tracking in MEG-MASC: a forward model with the envelope and its onsets, 1-8 Hz, lags 0-688 ms, leave-one-sound-file-out."""
from __future__ import annotations

from encoding_models import gpu_encoding as WG
from readers import meg_masc as C

def heldout_r(ff: WG.GpuFolds) -> float:
    preds = [(fd["XaUv"] @ (fd["UtCa"] / (fd["e"] + ff.lam * fd["e"].mean())[:, None])).T[None] for fd in ff.folds]
    return float(ff.all_valid.stats(preds)["pooled"][0].mean())


def fit_r(trials, names) -> float:
    ff = WG.GpuFolds(trials, names, C.N_LAGS)
    r = heldout_r(ff)
    return r


def session_tracking(sub: str, ses: int) -> dict:
    trials = C.listener_trials(sub, ses)
    for tr in trials:
        st, T = tr["st"], tr["T"]
        tr["F"]["env"] = C.bz(st["acoustic_raw"][:2, :T])
    out = {"listener": sub, "session": ses, "n_files": len(trials), "c1_r": fit_r(trials, ["env"])}
    return out
