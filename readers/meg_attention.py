"""MEG attention dataset: listener chapters and per-chapter regressors in the trial format of the encoding models."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from preprocessing.cnd_recordings import band_64

ROOT = Path(__file__).parents[1]
DER = ROOT / "data" / "derived" / "meg_attention"
TAB = ROOT / "results"
FILES = {"LP": {"attended": 1, "ignored": 4}, "HP": {"attended": 3, "ignored": 2}}
CHAPTERS = [(f, k) for f in FILES for k in range(1, 6)]
FS, N_LAGS, WIN_S = 64, 23, 0.8
STREAMS = ("attended", "ignored")


def rng(key: str) -> np.random.Generator:
    return np.random.default_rng([7, zlib.crc32(key.encode())])


def bz(F: np.ndarray) -> np.ndarray:
    """1-8 Hz on the 64 Hz grid (band_64), then z-scored per row over the chapter."""
    B = band_64(np.atleast_2d(F), FS, 1.0, 8.0)
    return (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)


def qc(p: int) -> dict:
    return json.loads((TAB / "qc" / f"P{p:02d}.json").read_text(encoding="utf-8"))


def kept_chapters(p: int) -> list[tuple[str, int]]:
    q = qc(p)
    return [(c["file"], c["chapter"]) for c in q["chapters"] if c["kept"]]


def included(p: int) -> bool:
    return bool(qc(p)["included"])


def features(f: str, k: int) -> dict:
    with np.load(DER / "features" / f"{f}_{k}.npz") as z:
        return {key: z[key] for key in z.files}


def surprisal_impulse(st: dict, stream: str, values: np.ndarray, keep: np.ndarray | None, n: int) -> np.ndarray:
    """Raw impulse train (1 x n) at the stream's onsets with the given values; words outside `keep` get 0."""
    idx = st[f"{stream}_onset_idx"]
    v = np.asarray(values, float).copy()
    if keep is not None:
        v = np.where(keep, v, 0.0)
    R = np.zeros((1, n))
    ok = (idx >= 0) & (idx < n)
    np.add.at(R[0], idx[ok], v[ok])
    return R


def listener_trials(p: int, band: str = "d18") -> list[dict]:
    """Kept chapters of a listener: {'file', 'chapter', 'Y' (channels x T, float64), 'v', 'T', 'F': {}, 'st'}."""
    out = []
    with np.load(DER / "meg" / f"P{p:02d}.npz") as z:
        for f, k in kept_chapters(p):
            Y = z[f"{f}_{k}_{band}"].astype(np.float64)
            v = z[f"{f}_{k}_valid"]
            st = features(f, k)
            T = min(Y.shape[1], int(st["n"]))
            out.append({"file": f, "chapter": k, "Y": Y[:, :T], "v": v[:T].copy(), "T": T, "F": {}, "st": st})
    return out


def add_base(trials: list[dict]) -> list[str]:
    """Base regressors (both streams): acoustic (34), word onsets (1), controls (4), band-passed and z-scored."""
    names = []
    for s in STREAMS:
        for part in ("acoustic", "word", "controls"):
            nm = f"{s}_{part}"
            for tr in trials:
                tr["F"][nm] = bz(tr["st"][f"{s}_{part}_raw"][:, : tr["T"]])
            names.append(nm)
    return names


def word_masks(trials: list[dict], stream: str, keep_key: str | None = None) -> list[np.ndarray]:
    """0-800 ms after each word of the stream (optionally kept words only), within the valid samples."""
    w = int(round(WIN_S * FS))
    out = []
    for tr in trials:
        idx = tr["st"][f"{stream}_onset_idx"]
        if keep_key is not None:
            idx = idx[tr["st"][f"{stream}_{keep_key}"]]
        m = np.zeros(tr["T"], bool)
        for i in idx:
            if i + w > 0 and i < tr["T"]:
                m[max(i, 0): min(i + w, tr["T"])] = True
        out.append(m & tr["v"])
    return out
