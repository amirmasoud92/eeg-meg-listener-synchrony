"""Display data of the illustrative panels: the CantiSong components of Fig. 1a and Fig. 6a and the MEG sensor layout.

Computed from the preprocessed CantiSong recordings (1-8 Hz, 64 Hz) and the header of one MEG attention recording, and
written to figure_data/illustrations.npz, which the figure code reads.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import scipy.signal as ss

from common.settings import PROJECT_ROOT, dataset_root, load_paths
from figures import design as V

FIGURE_DATA = PROJECT_ROOT / "figure_data"


def _song01_component():
    """CantiSong song 01, sung: the first correlated component (delta band), fitted on the other sung songs; its forward
    model (scalp pattern), the component in every listener and each listener's ISC with the mean of the others (display
    only)."""
    from synchrony import correlated_components as ISC
    fs = 64
    chans = pd.read_csv(FIGURE_DATA / "topographies.csv").channel.tolist()
    Z = [np.load(f) for f in sorted(glob.glob(str(load_paths().derived_root / "cantisong_1to8" / "sub*.npz")))]
    idx = [i for i, z in enumerate(Z) if "1-8|song|01" in z]
    b, a = ss.butter(2, [1 / (fs / 2), 4 / (fs / 2)], btype="band")

    def stim(song):
        n = min(Z[i][f"1-8|song|{song}"].shape[1] for i in idx)
        return np.array([ss.filtfilt(b, a, Z[i][f"1-8|song|{song}"][:, :n], axis=1) for i in idx])
    songs = sorted({k.split("|")[2] for z in Z for k in z.keys() if k.startswith("1-8|song|")} - {"01"})
    train = [stim(s) for s in songs if all(f"1-8|song|{s}" in Z[i] for i in idx)]
    Rw = Rb = 0.0
    for d in train:
        w_, b_ = ISC._cov_terms(d)
        Rw, Rb = Rw + w_, Rb + b_
    W, _ = ISC.corrca_from_terms(Rw, Rb, train[0].shape[0], 0.5, 1)
    A = (Rw @ W @ np.linalg.inv(W.T @ Rw @ W))[:, 0]
    X = stim("01")
    Y = np.einsum("dc,ndt->nt", W, X)
    isc = ISC.subject_isc(X, W)[:, 0]
    if A[chans.index("Cz")] < 0:                                   # the sign of a component is arbitrary: Cz positive
        A, Y = -A, -Y
    Y = (Y - Y.mean(1, keepdims=True)) / Y.std(1, keepdims=True)
    return chans, A, Y, isc, np.arange(Y.shape[1]) / fs + 5.0


def _space_pattern(cond):
    """Display only: the first correlated component (δ band) fitted on all 18 CantiSong songs in one version (speech:
    the spoken lyrics; melody: the hummed melody), and its forward model, Cz positive (the sign is arbitrary). The analysis
    fits each song's spaces on the other songs, with three components."""
    from synchrony import correlated_components as ISC
    fs = 64
    chans = pd.read_csv(FIGURE_DATA / "topographies.csv").channel.tolist()
    Z = [np.load(f) for f in sorted(glob.glob(str(load_paths().derived_root / "cantisong_1to8" / "sub*.npz")))]
    b, a = ss.butter(2, [1 / (fs / 2), 4 / (fs / 2)], btype="band")
    songs = sorted({k.split("|")[2] for z in Z for k in z.keys() if k.startswith(f"1-8|{cond}|")})
    Rw = Rb = 0.0
    for s in songs:
        idx = [i for i, z in enumerate(Z) if f"1-8|{cond}|{s}" in z]
        n = min(Z[i][f"1-8|{cond}|{s}"].shape[1] for i in idx)
        d = np.array([ss.filtfilt(b, a, Z[i][f"1-8|{cond}|{s}"][:, :n], axis=1) for i in idx])
        w_, b_ = ISC._cov_terms(d)
        Rw, Rb = Rw + w_, Rb + b_
    W, _ = ISC.corrca_from_terms(Rw, Rb, len(Z), 0.5, 1)
    A = (Rw @ W @ np.linalg.inv(W.T @ Rw @ W))[:, 0]
    return chans, (A if A[chans.index("Cz")] > 0 else -A)


def _meg_layout():
    import mne
    info = mne.io.read_info(dataset_root("meg_attention") / "1" / "LP_speaker_attended" / "data_meg.fif", verbose="error")    # header only
    return V.meg_xy([c["loc"][:3] for c in info["chs"]])


def write_illustration_data():
    chans, A, Y, isc, t = _song01_component()
    arrays = {"song01|chans": np.array(chans), "song01|A": A, "song01|Y": Y, "song01|isc": isc, "song01|t": t}
    for cond in ("speech", "melody"):
        ch, a = _space_pattern(cond)
        arrays[f"space_{cond}|chans"], arrays[f"space_{cond}|A"] = np.array(ch), a
    arrays["meg_layout|xy"] = np.asarray(_meg_layout())
    np.savez_compressed(FIGURE_DATA / "illustrations.npz", **arrays)
