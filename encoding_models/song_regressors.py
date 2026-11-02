"""Acoustic, phonetic and word regressors for CantiSong and the shared lagged design."""
from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as ss

from common.settings import load_paths
from encoding_models import speech_envelope as tk
from preprocessing import cnd_recordings as cnd
from readers import etard_release as etard
from synchrony import song_versions as c

FS = 64
BAND = (1.0, 8.0)
LAG_STEP, N_LAGS = 2, 23                  # 0-688 ms
WORD_WIN = int(0.8 * FS)
N_SHUF = 10
RIDGE_GRID = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
AUDIO = {"speech": "speech", "song": "song", "melody": "melody"}
TIMING = {"speech": "speech", "song": "song", "melody": "song"}      # hummed = sung timing
PHON = {  # 19 features: manner 5, place 7, voiced, vowel, front, central, back, high, low
    "p": "plosive bilabial", "b": "plosive bilabial voiced", "t": "plosive alveolar", "d": "plosive alveolar voiced",
    "k": "plosive velar", "g": "plosive velar voiced", "ch": "affricate postalveolar", "jh": "affricate postalveolar voiced",
    "f": "fricative labiodental", "v": "fricative labiodental voiced", "th": "fricative dental", "dh": "fricative dental voiced",
    "s": "fricative alveolar", "z": "fricative alveolar voiced", "sh": "fricative postalveolar", "zh": "fricative postalveolar voiced",
    "hh": "fricative glottal", "m": "nasal bilabial voiced", "n": "nasal alveolar voiced", "ng": "nasal velar voiced",
    "l": "approximant alveolar voiced", "r": "approximant postalveolar voiced", "w": "approximant bilabial voiced",
    "y": "approximant postalveolar voiced",
    "iy": "vowel voiced front high", "ih": "vowel voiced front high", "ey": "vowel voiced front", "eh": "vowel voiced front",
    "ae": "vowel voiced front low", "aa": "vowel voiced back low", "ao": "vowel voiced back", "ow": "vowel voiced back",
    "uh": "vowel voiced back high", "uw": "vowel voiced back high", "ah": "vowel voiced central", "er": "vowel voiced central",
    "aw": "vowel voiced central low", "ay": "vowel voiced central low", "oy": "vowel voiced back"}
PHON_NAMES = ["plosive", "fricative", "affricate", "nasal", "approximant", "bilabial", "labiodental", "dental", "alveolar",
              "postalveolar", "velar", "glottal", "voiced", "vowel", "front", "central", "back", "high", "low"]


def acoustic(song: str, version: str) -> np.ndarray:
    """34 x samples at 64 Hz: envelope, envelope onset, 16-band spectrogram, spectrogram onsets."""
    fs, x = wavfile.read(c.ROOT / "Stimuli" / "sounds" / f"{song}_{AUDIO[version]}.wav")
    x = np.asarray(x, float)
    x = x[:, 0] if x.ndim > 1 else x
    e1k = tk.broadband_envelope(x, fs, etard.load_config().section("envelope"))
    fr = Fraction(FS, 1000)
    env = ss.resample_poly(e1k, fr.numerator, fr.denominator)
    y = ss.resample_poly(x, 16000, fs) if fs != 16000 else x
    f, t, Z = ss.stft(y, 16000, nperseg=512, noverlap=512 - 250, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    edges = 700 * (10 ** (np.linspace(np.log10(1 + 100 / 700), np.log10(1 + 7000 / 700), 18)) - 1)
    bands = np.array([P[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(edges[:-2], edges[2:])]) ** 0.3
    n = min(env.size, bands.shape[1])
    env, bands = env[:n], bands[:, :n]
    on = lambda a: np.maximum(np.diff(a, axis=-1, prepend=a[..., :1]), 0)
    return np.vstack([env, on(env), bands, on(bands)])


def annotation_offset(song: str, timing: str) -> float:
    """Seconds to add to the released phone annotation of (song, timing): the stimulus-only check in
    song_annotation.py flags only song 09 sung, +2.88 s."""
    e = json.loads((load_paths().derived_root / "cantisong_words" / "annotation_offsets.json").read_text())[song][timing]
    return float(e["offset_s"]) if e["flag"] else 0.0


def load_words(song: str) -> pd.DataFrame:
    """The word table of a song, with annotation offsets applied."""
    w = pd.read_csv(load_paths().derived_root / "cantisong_words" / f"{song}_words.csv")
    for t in ("song", "speech"):
        o = annotation_offset(song, t)
        if o:
            w[[f"onset_{t}", f"offset_{t}"]] += o
    return w


def phonetic(song: str, version: str, n: int) -> np.ndarray:
    F = np.zeros((19, n))
    off = annotation_offset(song, TIMING[version])
    for ln in (c.ROOT / "Stimuli" / "ph_annotations" / f"{song}_{TIMING[version]}.txt").read_text().splitlines():
        if not ln.strip():
            continue
        a, b_, p = ln.split()
        if p in PHON:
            for name in PHON[p].split():
                F[PHON_NAMES.index(name), int((float(a) + off) * FS): int((float(b_) + off) * FS)] = 1.0
    return F


def lexical(words: pd.DataFrame, version: str, n: int, anchored_only: bool = False) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """word-onset impulses, surprisal impulses, onset sample indices."""
    col = TIMING[version]
    w = words.dropna(subset=[f"onset_{col}"])
    if anchored_only:
        w = w[w[f"anchored_{col}"]]
    idx = (w[f"onset_{col}"].to_numpy() * FS).round().astype(int)
    keep = idx < n
    idx, s = idx[keep], w.surprisal_bits.to_numpy()[keep]
    on, su = np.zeros(n), np.zeros(n)
    np.add.at(on, idx, 1.0)
    np.add.at(su, idx, s)
    return on, su, idx.tolist()


def stimulus(song: str, version: str, words: pd.DataFrame, anchored_only: bool) -> dict:
    A = acoustic(song, version)
    n = A.shape[1]
    on, su, idx = lexical(words, version, n, anchored_only)
    parts = {"acoustic": A, "word": on[None], "surprisal": su[None]}
    if version != "melody":
        parts["phonetic"] = phonetic(song, version, n)
    return {k: bz(v) for k, v in parts.items()} | {"word_idx": idx, "raw_surprisal": su}


def bz(F: np.ndarray) -> np.ndarray:
    """1-8 Hz on the 64 Hz grid, then z-scored per row (per song)."""
    B = cnd.band_64(F, FS, *BAND)
    return (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)


def lagged(F: np.ndarray, T: int) -> np.ndarray:
    X = np.zeros((T, F.shape[0] * N_LAGS))
    for k in range(N_LAGS):
        d = k * LAG_STEP
        if d < T:
            X[d:, k::N_LAGS] = F[:, : T - d].T
    return X


def fit_predict(blocks: list[dict], names: list[str], lam: float | None) -> tuple[list[np.ndarray], float]:
    """Leave-one-song-out ridge; blocks[j] = {'X': {name: lagged}, 'Y': C x T, 'v': mask}. Returns held-out
    predictions and the ridge (x mean eigenvalue) used; if lam is None it is chosen from RIDGE_GRID by the mean
    over channels of r on all held-out valid samples."""
    Xs = [np.hstack([bl["X"][nm] for nm in names]) for bl in blocks]
    Gs = [X[bl["v"]].T @ X[bl["v"]] for X, bl in zip(Xs, blocks)]
    Cs = [X[bl["v"]].T @ bl["Y"][:, bl["v"]].T for X, bl in zip(Xs, blocks)]
    G, C = sum(Gs), sum(Cs)
    grid = RIDGE_GRID if lam is None else [lam]
    preds = {g: [] for g in grid}
    for j in range(len(blocks)):
        Gt, Ct = G - Gs[j], C - Cs[j]
        e, V = np.linalg.eigh(Gt)
        VC, XV = V.T @ Ct, Xs[j] @ V
        for g in grid:
            preds[g].append((XV @ (VC / (e + g * e.mean())[:, None])).T)
    if lam is None:
        score = {g: r_mean(blocks, preds[g], None) for g in grid}
        lam = max(score, key=score.get)
    return preds[lam], lam


def r_mean(blocks: list[dict], preds: list[np.ndarray], masks: list[np.ndarray] | None) -> float:
    Y = np.hstack([bl["Y"][:, (bl["v"] if masks is None else masks[j])] for j, bl in enumerate(blocks)])
    P = np.hstack([p[:, (bl["v"] if masks is None else masks[j])] for j, (p, bl) in enumerate(zip(preds, blocks))])
    Y = Y - Y.mean(1, keepdims=True)
    P = P - P.mean(1, keepdims=True)
    return float(np.mean((Y * P).sum(1) / np.sqrt((Y * Y).sum(1) * (P * P).sum(1) + 1e-300)))
